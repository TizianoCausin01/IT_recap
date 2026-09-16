"""
Boost the GRU decoder with a spatial readout of a convolutional feature map.

The time-bin search left every architecture at the same test MSE while their
stim_r differed, which says the decoders miss a component of the response that
squared error barely prices. This script asks whether that component is spatial:
the GRU keeps the pooled I-JEPA features it won with, and a small head reads the
*residual* it leaves off the complete last-conv map of a CNN (AlexNet conv5 or
ConvNeXt stage 4), giving every IT site its own spatial weighting.

Two heads are trained on the identical residual. The factorized readout gives
each site a spatial mask; the pooled control averages the map over space first.
Their difference isolates spatial information from the mere addition of CNN
features.

The residuals the head is trained on are out-of-fold: the GRU is refitted K
times, each fold predicted by a GRU that never saw it, so the head is not
fitting the base model's own overfitting. The GRU used at prediction time is
the one trained on the whole fit split.

The script also reports why MSE and stim_r disagree, by decomposing the test
MSE into the part no model can remove (trial noise, fixed by the noise ceiling)
and the part that is predictable, and by measuring how much each prediction is
shrunk relative to its target.
"""

import argparse
import copy
import json
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset


sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "python_scripts" / "src")
)

from IT_recap.neural_prediction_training import (  # noqa: E402
    stimulus_correlation,
)
from IT_recap.tvsd_experiments import (  # noqa: E402
    PROJECT_ROOT,
    decompose_test_mse,
    score_and_decompose,
    load_project_paths,
    make_tensor_loader,
    prepare_search_data,
    resolve_device,
    train_cached_feature_decoder,
)
from model_classes.spatial_readout_models import (  # noqa: E402
    SPATIAL_READOUT_CLASSES,
    build_spatial_readout,
)
from model_classes.timebin_models import build_timebin_model  # noqa: E402


# Only the repetition mean is scored; it is the aggregation with a defined
# Spearman-Brown noise ceiling.
REDUCER = "mean"

# Hyperparameters of the base decoder, read out of the search results.
BASE_HYPERPARAMETERS = (
    "hidden_dim",
    "time_embedding_dim",
    "dropout",
    "input_noise_std",
    "temporal_noise_std",
    "learning_rate",
    "weight_decay",
)


@dataclass
class Cfg:
    # Caches and the finished search this experiment builds on.
    env: str | None = None
    mua_file_name: str = "f_THINGS_MUA_trials.mat"
    feature_archive_name: str = "tvsd_monkeyF_ijepa_vith14_1k_224_features.npz"
    search_results_dir: str | None = None
    output_dir: str | None = None

    # The cached convolutional map, written by extract_tvsd_spatial_features.py.
    spatial_stem: str = "tvsd_monkeyF_alexnet_features_11_spatial"

    # Target identity and window, identical to the architecture search.
    area: str = "IT"
    target_fs: int = 100
    time_start_ms: float = 0.0
    time_end_ms: float = 200.0
    window_start_ms: float = 80.0
    window_end_ms: float = 180.0
    timebin_ms: float = 20.0
    model_name: str = "ijepa_vith14_1k"
    layer_names: list[str] = field(
        default_factory=lambda: [
            "encoder.layer.4.output.dense",
            "encoder.layer.17.output.dense",
            "encoder.layer.27.output.dense",
        ]
    )
    validation_fraction: float = 0.1
    random_seed: int = 0

    # Base decoder. The architecture and its winning configuration come from
    # the search; only the fold count is specific to this experiment.
    base_architecture: str = "gru"
    cross_fit_folds: int = 3

    # Residual head.
    readouts: str = ",".join(SPATIAL_READOUT_CLASSES)
    hidden_dim: int = 64
    dropout: float = 0.3
    mask_temperature: float = 1.0
    head_learning_rate: float = 1e-3
    head_weight_decay: float = 1e-2
    # The spatial masks only see gradient through the feature weights, so at a
    # shared learning rate they barely move within the epoch budget and the
    # factorized head silently degenerates into the pooled control.
    mask_learning_rate_scale: float = 10.0

    # Optimization, shared by the base decoder and the heads.
    batch_size: int = 256
    num_workers: int = 0
    epochs: int = 50
    minimum_epochs: int = 10
    patience: int = 10
    gradient_clip: float = 1.0
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    selection_metric: str = "stim_r"

    # Evaluation and bookkeeping.
    noise_ceiling_resamples: int = 40
    device: str = "auto"
    smoke_test: bool = False


"""
parse_args
Parse command-line overrides into a configuration object.

OUTPUT:
    - cfg: Cfg -> caches, base decoder, head, and optimization settings
"""
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    default_layer_names = Cfg.__dataclass_fields__["layer_names"].default_factory()
    for field_name, field_definition in Cfg.__dataclass_fields__.items():
        if field_name == "layer_names":
            parser.add_argument(
                "--layer_names",
                default=",".join(default_layer_names),
                help="Comma-separated hooked ANN layers, in model order.",
            )
            continue
        # end if the layer selection needs list parsing
        default = field_definition.default
        argument_name = f"--{field_name}"
        if isinstance(default, bool):
            parser.add_argument(
                argument_name,
                action=argparse.BooleanOptionalAction,
                default=default,
            )
        elif default is None:
            parser.add_argument(argument_name, default=default)
        else:
            parser.add_argument(
                argument_name, type=type(default), default=default
            )
        # end if boolean, optional string, or typed argument
    # end for configuration field
    arguments = vars(parser.parse_args())
    arguments["layer_names"] = [
        name.strip() for name in arguments["layer_names"].split(",") if name.strip()
    ]
    return Cfg(**arguments)
# EOF


class SpatialResidualDataset(Dataset):
    """
    Pair each presentation with its stimulus feature map and its target rows.

    The map cache is float16 and several gigabytes, so it stays memory-mapped
    and one presentation is converted at a time. Presentations of the same
    image simply index the same row.

    INPUT (__getitem__):
        - index: int -> position within this subset

    OUTPUT:
        - feature_map: torch.Tensor -> [channels, height, width]
        - residual: torch.Tensor -> [time, neurons] target minus base prediction
    """

    """
    __init__
    Store the memory-mapped maps, the stimulus row of every presentation, and
    the residual it should explain.

    INPUT:
        - feature_maps: np.ndarray -> [stimuli, channels, height, width] cache
        - image_rows: np.ndarray -> zero-based cache row of each presentation
        - residuals: np.ndarray -> [presentations, time, neurons]

    OUTPUT:
        - None
    """
    def __init__(self, feature_maps, image_rows, residuals):
        self.feature_maps = feature_maps
        self.image_rows = np.asarray(image_rows, dtype=int)
        self.residuals = np.asarray(residuals, dtype=np.float32)
        if len(self.image_rows) != len(self.residuals):
            raise ValueError("image_rows and residuals must align.")
        # end if the presentation axes disagree
        if self.image_rows.max(initial=0) >= len(feature_maps):
            raise IndexError("A presentation indexes a missing feature map.")
        # end if a stimulus is absent from the cache

    def __len__(self):
        return len(self.image_rows)
    # EOF

    def __getitem__(self, index):
        feature_map = np.asarray(
            self.feature_maps[self.image_rows[index]], dtype=np.float32
        )
        return torch.from_numpy(feature_map), torch.from_numpy(
            self.residuals[index]
        )
    # EOF
# EOC


"""
load_base_configuration
Read the winning configuration of the base architecture out of the search.

INPUT:
    - cfg: Cfg -> architecture name and results location

OUTPUT:
    - configuration: dict -> the searched hyperparameters of that architecture
    - results_dir: Path -> the search directory the configuration came from
"""
def load_base_configuration(cfg):
    results_dir = Path(
        cfg.search_results_dir
        or PROJECT_ROOT
        / "results"
        / f"tvsd_timebin_architecture_search_{cfg.model_name}"
    ).expanduser()
    results_path = results_dir / "search_results.json"
    if not results_path.is_file():
        raise FileNotFoundError(
            f"{results_path} is missing. Run "
            "run_tvsd_timebin_architecture_search.py first."
        )
    # end if the search has not been run
    with open(results_path, "r") as results_file:
        best_rows = json.load(results_file)["best_per_architecture"]
    # end with search results
    if cfg.base_architecture not in best_rows:
        raise KeyError(
            f"The search holds no {cfg.base_architecture!r} result; it has "
            f"{sorted(best_rows)}."
        )
    # end if the requested base architecture was not searched
    row = best_rows[cfg.base_architecture]
    configuration = {
        name: row[name] for name in BASE_HYPERPARAMETERS if name in row
    }
    return configuration, results_dir
# EOF


"""
train_base_decoder
Fit one base decoder on a subset of the fit split and return it.

INPUT:
    - cfg: Cfg -> optimization settings
    - configuration: dict -> the base decoder's hyperparameters
    - shapes: dict -> data shapes for the decoder
    - loaders: dict -> train and validation loaders
    - device: torch.device -> compute device

OUTPUT:
    - model: nn.Module -> the restored best checkpoint
    - best_epoch: int -> selected epoch
"""
def train_base_decoder(cfg, configuration, shapes, loaders, device):
    model_kwargs = {
        name: value
        for name, value in configuration.items()
        if name not in {"learning_rate", "weight_decay"}
    }
    torch.manual_seed(cfg.random_seed)
    model = build_timebin_model(
        cfg.base_architecture, **shapes, **model_kwargs
    ).to(device)
    run_cfg = replace(
        cfg,
        learning_rate=configuration["learning_rate"],
        weight_decay=configuration["weight_decay"],
    )
    _, best_epoch, _, _ = train_cached_feature_decoder(
        model, loaders, run_cfg, device, verbose=False
    )
    return model, best_epoch
# EOF


"""
predict_with_base
Run a trained base decoder over a feature/target array pair.

INPUT:
    - model: nn.Module -> trained decoder
    - features: np.ndarray -> [presentations, layers, embedding]
    - targets: np.ndarray -> [presentations, time, neurons]
    - cfg: Cfg -> loader settings
    - device: torch.device -> compute device

OUTPUT:
    - predictions: np.ndarray -> [presentations, time, neurons]
"""
def predict_with_base(model, features, targets, cfg, device):
    loader = make_tensor_loader(features, targets, cfg, shuffle=False)
    model.eval()
    batches = []
    with torch.no_grad():
        for batch_features, _ in loader:
            batches.append(model(batch_features.to(device))[0].cpu())
        # end for evaluation batch
    # end with no gradient tracking
    return torch.cat(batches).numpy()
# EOF


"""
cross_fit_base_predictions
Predict every fit presentation with a decoder that never saw it.

Training the residual head on in-sample residuals would let it chase whatever
the base decoder overfitted. K-fold refitting costs K extra base runs and makes
the residual an honest estimate of what the base model misses.

INPUT:
    - cfg: Cfg -> fold count and optimization settings
    - configuration: dict -> the base decoder's hyperparameters
    - shapes: dict -> data shapes for the decoder
    - data: dict -> subset features and targets
    - device: torch.device -> compute device

OUTPUT:
    - out_of_fold: np.ndarray -> [fit presentations, time, neurons]
"""
def cross_fit_base_predictions(cfg, configuration, shapes, data, device):
    fit_features = data["subset_features"]["train"]
    fit_targets = data["subset_targets"]["train"]
    fold_rng = np.random.default_rng(cfg.random_seed)
    fold_of = fold_rng.permutation(len(fit_features)) % cfg.cross_fit_folds
    out_of_fold = np.empty_like(fit_targets)

    for fold in range(cfg.cross_fit_folds):
        held_out = np.flatnonzero(fold_of == fold)
        kept = np.flatnonzero(fold_of != fold)
        loaders = {
            "train": make_tensor_loader(
                fit_features[kept], fit_targets[kept], cfg, shuffle=True
            ),
            # The shared validation split selects the checkpoint of every fold,
            # exactly as it does for the full-fit decoder.
            "validation": data["loaders"]["validation"],
        }
        fold_start = time.perf_counter()
        model, best_epoch = train_base_decoder(
            cfg, configuration, shapes, loaders, device
        )
        out_of_fold[held_out] = predict_with_base(
            model, fit_features[held_out], fit_targets[held_out], cfg, device
        )
        print(
            f"  fold {fold + 1}/{cfg.cross_fit_folds}: {len(kept):,} fit, "
            f"{len(held_out):,} predicted, epoch {best_epoch:02d}, "
            f"{time.perf_counter() - fold_start:.0f}s"
        )
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        # end if the device caches its allocations
    # end for cross-fitting fold
    return out_of_fold
# EOF


"""
evaluate_head
Run a residual head over a loader and return its predictions in order.

INPUT:
    - head: nn.Module -> residual readout
    - loader: DataLoader -> evaluation batches
    - device: torch.device -> compute device

OUTPUT:
    - predictions: np.ndarray -> [presentations, time, neurons]
"""
def evaluate_head(head, loader, device):
    head.eval()
    batches = []
    with torch.no_grad():
        for feature_maps, _ in loader:
            batches.append(head(feature_maps.to(device))[0].cpu())
        # end for evaluation batch
    # end with no gradient tracking
    return torch.cat(batches).numpy()
# EOF


"""
train_residual_head
Fit one head on the base decoder's residual, selecting on the combined model.

The head minimizes squared error on the residual, but the checkpoint is chosen
by the stim_r of base + head on the validation split: that is the quantity the
experiment reports, and MSE on a residual is dominated by trial noise.

INPUT:
    - cfg: Cfg -> optimization settings
    - head: nn.Module -> residual readout to fit
    - loaders: dict -> train and validation residual loaders
    - base_validation: np.ndarray -> base predictions on the validation split
    - validation_targets: np.ndarray -> standardized validation targets
    - device: torch.device -> compute device

OUTPUT:
    - history: list[dict] -> per-epoch residual MSE and combined validation stim_r
    - best_epoch: int -> selected epoch
    - best_validation_stim_r: float -> combined validation stim_r at that epoch
"""
def train_residual_head(
    cfg, head, loaders, base_validation, validation_targets, device
):
    # Weight decay would pull the mask logits back towards a flat mask, which
    # is the very hypothesis under test, so that group is left undecayed.
    mask_parameters = [
        parameter
        for name, parameter in head.named_parameters()
        if name == "spatial_logits"
    ]
    other_parameters = [
        parameter
        for name, parameter in head.named_parameters()
        if name != "spatial_logits"
    ]
    parameter_groups = [
        {
            "params": other_parameters,
            "lr": cfg.head_learning_rate,
            "weight_decay": cfg.head_weight_decay,
        }
    ]
    if mask_parameters:
        parameter_groups.append(
            {
                "params": mask_parameters,
                "lr": cfg.head_learning_rate * cfg.mask_learning_rate_scale,
                "weight_decay": 0.0,
            }
        )
    # end if this head owns spatial masks
    optimizer = torch.optim.AdamW(parameter_groups)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(3, cfg.patience // 3)
    )
    cost_function = torch.nn.MSELoss()

    # The zero-initialized head is a valid checkpoint: it scores exactly the
    # base decoder, so a head that never helps cannot make the result worse.
    best_state = copy.deepcopy(head.state_dict())
    best_stim_r = float(
        np.nanmean(stimulus_correlation(base_validation, validation_targets))
    )
    base_validation_stim_r = best_stim_r
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        head.train()
        train_squared_error, train_values = 0.0, 0
        for feature_maps, residuals in loaders["train"]:
            feature_maps = feature_maps.to(device)
            residuals = residuals.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions, _ = head(feature_maps)
            loss = cost_function(predictions, residuals)
            loss.backward()
            if cfg.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    head.parameters(), cfg.gradient_clip
                )
            # end if gradient clipping is enabled
            optimizer.step()
            train_squared_error += loss.item() * residuals.numel()
            train_values += residuals.numel()
        # end for training batch

        head_validation = evaluate_head(head, loaders["validation"], device)
        combined_validation = base_validation + head_validation
        validation_stim_r = float(
            np.nanmean(
                stimulus_correlation(combined_validation, validation_targets)
            )
        )
        scheduler.step(-validation_stim_r)
        history.append(
            {
                "epoch": epoch,
                "residual_train_mse": train_squared_error / train_values,
                "combined_validation_stim_r": validation_stim_r,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"    epoch {epoch:03d}/{cfg.epochs:03d} | residual MSE "
            f"{history[-1]['residual_train_mse']:.6f} | combined validation "
            f"stim_r {validation_stim_r:.4f} (base {base_validation_stim_r:.4f})"
        )

        if validation_stim_r > best_stim_r + 1e-8:
            best_stim_r = validation_stim_r
            best_epoch = epoch
            best_state = copy.deepcopy(head.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        # end if the combined model improved

        if (
            epoch >= cfg.minimum_epochs
            and epochs_without_improvement >= cfg.patience
        ):
            print(f"    early stop after {epoch} epochs")
            break
        # end if early-stopping patience is exhausted
    # end for optimization epoch

    head.load_state_dict(best_state)
    return history, best_epoch, best_stim_r
# EOF


"""
load_spatial_cache
Open the cached feature maps of both stimulus splits.

INPUT:
    - cfg: Cfg -> cache stem
    - paths: dict -> active project paths

OUTPUT:
    - caches: dict -> split name to memory-mapped [stimuli, C, H, W] array
    - feature_shape: tuple -> cached [channels, height, width]
"""
def load_spatial_cache(cfg, paths):
    cache_dir = Path(paths["data_path"]) / "models"
    caches = {}
    for split_name in ("train", "test"):
        cache_path = cache_dir / f"{cfg.spatial_stem}_{split_name}.npy"
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"Missing {cache_path}. Run extract_tvsd_spatial_features.py "
                "for this backbone first."
            )
        # end if the cache is absent
        caches[split_name] = np.load(cache_path, mmap_mode="r")
    # end for stimulus split
    feature_shape = tuple(caches["train"].shape[1:])
    if tuple(caches["test"].shape[1:]) != feature_shape:
        raise ValueError("The train and test caches have different geometry.")
    # end if the two splits disagree
    return caches, feature_shape
# EOF


def main():
    cfg = parse_args()
    if cfg.smoke_test:
        cfg.epochs = min(cfg.epochs, 2)
        cfg.minimum_epochs = 1
        cfg.patience = 1
        cfg.cross_fit_folds = 2
        cfg.noise_ceiling_resamples = 4
    # end if a smoke run was requested

    paths = load_project_paths(cfg)
    device = resolve_device(cfg.device)
    configuration, search_dir = load_base_configuration(cfg)
    backbone_label = cfg.spatial_stem.replace("tvsd_monkeyF_", "").replace(
        "_spatial", ""
    )
    output_dir = Path(
        cfg.output_dir
        or PROJECT_ROOT / "results" / f"tvsd_spatial_residual_{backbone_label}"
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = prepare_search_data(cfg, paths, device)
    scoring, shapes = data["scoring"], data["shapes"]
    subset_targets, subset_features = data["subset_targets"], data["subset_features"]
    caches, feature_shape = load_spatial_cache(cfg, paths)
    print(
        f"spatial cache {cfg.spatial_stem}: {feature_shape} per stimulus | "
        f"base decoder {cfg.base_architecture} from {search_dir.name}"
    )

    # Rows of the map cache follow the one-based stimulus identifiers, so a
    # presentation reads the row of the image it showed.
    allmat, indices = data["allmat"], data["indices"]
    image_rows = {
        "train": allmat[indices["train"], 1] - 1,
        "validation": allmat[indices["validation"], 1] - 1,
        "test": allmat[indices["test"], 2] - 1,
    }
    cache_of = {"train": "train", "validation": "train", "test": "test"}

    # --- base decoder: one fit on everything, plus out-of-fold predictions ---
    print(f"training the base {cfg.base_architecture} on the full fit split")
    base_model, base_epoch = train_base_decoder(
        cfg, configuration, shapes, data["loaders"], device
    )
    base_predictions = {
        subset_name: predict_with_base(
            base_model,
            subset_features[subset_name],
            subset_targets[subset_name],
            cfg,
            device,
        )
        for subset_name in ("validation", "test")
    }
    print(f"  best epoch {base_epoch:02d}")
    print(f"cross-fitting the base decoder over {cfg.cross_fit_folds} folds")
    base_predictions["train"] = cross_fit_base_predictions(
        cfg, configuration, shapes, data, device
    )

    residuals = {
        subset_name: subset_targets[subset_name] - base_predictions[subset_name]
        for subset_name in ("train", "validation", "test")
    }
    print(
        "residual variance (fit split): "
        f"{residuals['train'].var():.4f} of target variance "
        f"{subset_targets['train'].var():.4f}"
    )

    result_rows, site_correlations = [], {}
    base_row, base_site_r = score_and_decompose(
        base_predictions["test"], subset_targets["test"], scoring,
        f"{cfg.base_architecture} (base)", REDUCER,
    )
    result_rows.append(base_row)
    site_correlations["base"] = base_site_r

    # --- residual heads ---
    readout_names = [
        name.strip() for name in cfg.readouts.split(",") if name.strip()
    ]
    head_histories = {}
    for readout_name in readout_names:
        print(f"training the {readout_name} residual head")
        loaders = {
            subset_name: DataLoader(
                SpatialResidualDataset(
                    caches[cache_of[subset_name]],
                    image_rows[subset_name],
                    residuals[subset_name],
                ),
                batch_size=cfg.batch_size,
                shuffle=subset_name == "train",
                num_workers=cfg.num_workers,
                generator=torch.Generator().manual_seed(cfg.random_seed),
            )
            for subset_name in ("train", "validation", "test")
        }
        torch.manual_seed(cfg.random_seed)
        head = build_spatial_readout(
            readout_name,
            n_channels=feature_shape[0],
            spatial_size=feature_shape[1:],
            n_timepoints=shapes["n_timepoints"],
            n_neurons=shapes["n_neurons"],
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
            mask_temperature=cfg.mask_temperature,
        ).to(device)
        history, best_epoch, best_stim_r = train_residual_head(
            cfg,
            head,
            loaders,
            base_predictions["validation"],
            subset_targets["validation"],
            device,
        )
        head_histories[readout_name] = history
        head_test = evaluate_head(head, loaders["test"], device)
        row, site_r = score_and_decompose(
            base_predictions["test"] + head_test,
            subset_targets["test"],
            scoring,
            f"{cfg.base_architecture} + {readout_name}",
            REDUCER,
        )
        row["head_parameters"] = sum(
            parameter.numel()
            for parameter in head.parameters()
            if parameter.requires_grad
        )
        row["head_best_epoch"] = best_epoch
        row["combined_validation_stim_r"] = round(best_stim_r, 4)
        result_rows.append(row)
        site_correlations[readout_name] = site_r
        if readout_name == "factorized_spatial":
            masks = head.spatial_masks().detach().cpu().numpy()
            np.save(output_dir / "site_spatial_masks.npy", masks)
            # A mask concentrated on few positions means the site reads a
            # location; a flat one means it does not use space at all.
            entropy = -(masks * np.log(masks + 1e-12)).sum(axis=(1, 2))
            uniform_entropy = float(np.log(masks[0].size))
            print(
                f"  spatial masks: mean entropy {entropy.mean():.3f} of "
                f"{uniform_entropy:.3f} for a flat mask"
            )
            row["mean_mask_entropy"] = round(float(entropy.mean()), 3)
            row["uniform_mask_entropy"] = round(uniform_entropy, 3)
        # end if the factorized masks are available
        del head
        if device.type == "mps":
            torch.mps.empty_cache()
        # end if the device caches its allocations
    # end for residual readout

    with open(output_dir / "config.json", "w") as config_file:
        json.dump(
            {
                **asdict(cfg),
                "base_configuration": configuration,
                "feature_shape": list(feature_shape),
                "bin_edges_ms": data["bin_edges_ms"].tolist(),
            },
            config_file,
            indent=2,
        )
    # end with saved configuration
    with open(output_dir / "results.json", "w") as results_file:
        json.dump(
            {"rows": result_rows, "histories": head_histories},
            results_file,
            indent=2,
        )
    # end with saved metrics
    np.savez_compressed(
        output_dir / "site_stim_r.npz",
        ceiling=scoring["ceiling"],
        **site_correlations,
    )

    header = (
        f"{'model':<34}{'test r':>9}{'frac ceil':>11}{'test MSE':>10}"
        f"{'floor':>9}{'above floor':>13}{'scale':>8}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in result_rows:
        print(
            f"{row['model']:<34}{row['mean_stim_r_response']:>9.4f}"
            f"{row['fraction_of_ceiling']:>11.3f}{row['test_mse']:>10.5f}"
            f"{row['irreducible_mse']:>9.5f}{row['mse_above_floor']:>13.5f}"
            f"{row['mean_optimal_scale']:>8.3f}"
        )
    # end for scored model
    print(
        "\nfloor is the MSE that trial noise alone leaves at this noise "
        "ceiling; scale is the factor each prediction would need to match its "
        "target's spread."
    )
    print(f"\nsaved to {output_dir}")
# EOF


if __name__ == "__main__":
    main()
# EOC
