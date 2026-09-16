"""
Train one decoder on pooled I-JEPA layers and the AlexNet conv map together.

The residual experiment froze the GRU and patched what it missed, which leaves
the pooled branch fitted to a target the spatial branch was never allowed to
help with. Here the two paths are trained jointly: the GRU reads the three
I-JEPA depths, a factorized readout gives every IT site its own mask over the
conv5 map, and their predictions are summed before a single MSE.

Three variants run under one protocol so the comparison is exact -- the pooled
branch alone, the spatial branch alone, and the two together -- against the
ridge reference and the two-stage residual result. Each reports test stim_r,
the MSE decomposition that explains why raw squared error disagrees with it,
and whether the run was still improving inside its epoch budget.
"""

import argparse
import copy
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "python_scripts" / "src")
)

from IT_recap.neural_prediction_training import stimulus_correlation  # noqa: E402
from IT_recap.tvsd_experiments import (  # noqa: E402
    PROJECT_ROOT,
    load_project_paths,
    prepare_search_data,
    resolve_device,
    score_and_decompose,
)
from model_classes.spatial_readout_models import (  # noqa: E402
    JointSpatialTemporalDecoder,
)


REDUCER = "mean"

# Hyperparameters of the pooled branch, read out of the finished search.
TEMPORAL_HYPERPARAMETERS = (
    "hidden_dim",
    "time_embedding_dim",
    "dropout",
    "input_noise_std",
    "temporal_noise_std",
)

# Which branches each named variant enables.
VARIANT_BRANCHES = {
    "temporal_only": (True, False),
    "spatial_only": (False, True),
    "joint": (True, True),
}

# Epochs used to judge whether a run had stopped improving.
TREND_WINDOW = 10


@dataclass
class Cfg:
    # Caches and the finished search this experiment builds on.
    env: str | None = None
    mua_file_name: str = "f_THINGS_MUA_trials.mat"
    feature_archive_name: str = "tvsd_monkeyF_ijepa_vith14_1k_224_features.npz"
    search_results_dir: str | None = None
    output_dir: str | None = None
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

    # Branches. The pooled branch keeps the architecture and configuration it
    # won the search with; only its learning rate is set here.
    variants: str = ",".join(VARIANT_BRANCHES)
    temporal_architecture: str = "gru"
    spatial_readout: str = "factorized_spatial"
    spatial_hidden_dim: int = 64
    spatial_dropout: float = 0.3
    mask_temperature: float = 1.0

    # Optimization. The three parameter groups follow the two experiments this
    # one merges: the searched decoder settings and the residual head's.
    batch_size: int = 256
    num_workers: int = 0
    epochs: int = 50
    minimum_epochs: int = 10
    patience: int = 12
    gradient_clip: float = 1.0
    temporal_learning_rate: float = 3e-4
    temporal_weight_decay: float = 1e-2
    spatial_learning_rate: float = 1e-3
    spatial_weight_decay: float = 1e-2
    # The masks only see gradient through the feature weights, so at a shared
    # learning rate they stay flat and the spatial branch loses its point.
    mask_learning_rate_scale: float = 10.0

    # Evaluation and bookkeeping.
    noise_ceiling_resamples: int = 40
    selection_metric: str = "stim_r"
    device: str = "auto"
    smoke_test: bool = False


"""
parse_args
Parse command-line overrides into a configuration object.

OUTPUT:
    - cfg: Cfg -> caches, branches, and optimization settings
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


class JointFeatureDataset(Dataset):
    """
    Serve the pooled layer features and the conv map of one presentation.

    The map cache is float16 and about 2 GB, so it stays memory-mapped and one
    presentation is converted at a time; the pooled features are already in RAM.

    INPUT (__getitem__):
        - index: int -> position within this subset

    OUTPUT:
        - layer_features: torch.Tensor -> [layers, embedding]
        - feature_map: torch.Tensor -> [channels, height, width]
        - target: torch.Tensor -> [time, neurons] standardized response
    """

    def __init__(
        self, layer_features, feature_maps, image_rows, targets, load_maps=True
    ):
        # A variant without a spatial branch would otherwise page two gigabytes
        # of conv maps through memory for nothing.
        self.load_maps = bool(load_maps)
        self.layer_features = np.asarray(layer_features, dtype=np.float32)
        self.feature_maps = feature_maps
        self.image_rows = np.asarray(image_rows, dtype=int)
        self.targets = np.asarray(targets, dtype=np.float32)
        lengths = {
            len(self.layer_features), len(self.image_rows), len(self.targets)
        }
        if len(lengths) != 1:
            raise ValueError("Features, stimulus rows, and targets must align.")
        # end if the presentation axes disagree
        if self.image_rows.max(initial=0) >= len(feature_maps):
            raise IndexError("A presentation indexes a missing feature map.")
        # end if a stimulus is absent from the map cache

    def __len__(self):
        return len(self.targets)
    # EOF

    def __getitem__(self, index):
        if self.load_maps:
            feature_map = torch.from_numpy(
                np.asarray(
                    self.feature_maps[self.image_rows[index]], dtype=np.float32
                )
            )
        else:
            feature_map = torch.zeros(0)
        # end if this variant reads the conv map
        return (
            torch.from_numpy(self.layer_features[index]),
            feature_map,
            torch.from_numpy(self.targets[index]),
        )
    # EOF
# EOC


"""
load_temporal_configuration
Read the winning configuration of the pooled branch out of the search results.

INPUT:
    - cfg: Cfg -> architecture name and results location

OUTPUT:
    - configuration: dict -> that architecture's searched hyperparameters
    - results_dir: Path -> the directory it came from
"""
def load_temporal_configuration(cfg):
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
    row = best_rows[cfg.temporal_architecture]
    return {
        name: row[name] for name in TEMPORAL_HYPERPARAMETERS if name in row
    }, results_dir
# EOF


"""
build_parameter_groups
Give the pooled branch, the spatial branch, and the masks their own settings.

INPUT:
    - model: JointSpatialTemporalDecoder -> the model being optimized
    - cfg: Cfg -> per-branch learning rates and weight decays

OUTPUT:
    - groups: list[dict] -> optimizer parameter groups
"""
def build_parameter_groups(model, cfg):
    mask_names = set(model.mask_parameter_names())
    groups, seen = [], set()
    for prefix, learning_rate, weight_decay in (
        ("temporal_branch", cfg.temporal_learning_rate, cfg.temporal_weight_decay),
        ("spatial_branch", cfg.spatial_learning_rate, cfg.spatial_weight_decay),
    ):
        parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if name.startswith(prefix) and name not in mask_names
        ]
        if parameters:
            groups.append(
                {
                    "params": parameters,
                    "lr": learning_rate,
                    "weight_decay": weight_decay,
                }
            )
            seen.update(
                name
                for name, _ in model.named_parameters()
                if name.startswith(prefix) and name not in mask_names
            )
        # end if this branch is present
    # end for branch

    mask_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name in mask_names
    ]
    if mask_parameters:
        # Weight decay on the mask logits pulls them back to a flat mask, which
        # is exactly the hypothesis under test.
        groups.append(
            {
                "params": mask_parameters,
                "lr": cfg.spatial_learning_rate * cfg.mask_learning_rate_scale,
                "weight_decay": 0.0,
            }
        )
    # end if the model owns spatial masks
    return groups
# EOF


"""
evaluate_joint
Run the joint decoder over a loader and collect predictions and targets.

INPUT:
    - model: nn.Module -> joint decoder
    - loader: DataLoader -> evaluation batches
    - device: torch.device -> compute device

OUTPUT:
    - predictions: np.ndarray -> [presentations, time, neurons]
    - targets: np.ndarray -> [presentations, time, neurons]
"""
def evaluate_joint(model, loader, device):
    model.eval()
    prediction_batches, target_batches = [], []
    with torch.no_grad():
        for layer_features, feature_maps, targets in loader:
            predictions, _ = model(
                layer_features.to(device), feature_maps.to(device)
            )
            prediction_batches.append(predictions.cpu())
            target_batches.append(targets)
        # end for evaluation batch
    # end with no gradient tracking
    return (
        torch.cat(prediction_batches).numpy(),
        torch.cat(target_batches).numpy(),
    )
# EOF


"""
train_joint_decoder
Optimize one variant under MSE and restore its best validation checkpoint.

INPUT:
    - cfg: Cfg -> optimization settings
    - model: nn.Module -> joint decoder on the device
    - loaders: dict -> train and validation loaders
    - device: torch.device -> compute device

OUTPUT:
    - history: list[dict] -> per-epoch train MSE and validation stim_r
    - best_epoch: int -> selected checkpoint epoch
    - best_validation_stim_r: float -> validation stim_r at that epoch
"""
def train_joint_decoder(cfg, model, loaders, device):
    optimizer = torch.optim.AdamW(build_parameter_groups(model, cfg))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(3, cfg.patience // 3)
    )
    cost_function = torch.nn.MSELoss()

    best_state = copy.deepcopy(model.state_dict())
    best_validation_stim_r = -np.inf
    best_epoch, epochs_without_improvement = 0, 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_squared_error, train_values = 0.0, 0
        for layer_features, feature_maps, targets in loaders["train"]:
            layer_features = layer_features.to(device)
            feature_maps = feature_maps.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions, _ = model(layer_features, feature_maps)
            loss = cost_function(predictions, targets)
            loss.backward()
            if cfg.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.gradient_clip
                )
            # end if gradient clipping is enabled
            optimizer.step()
            train_squared_error += loss.item() * targets.numel()
            train_values += targets.numel()
        # end for training batch

        validation_predictions, validation_targets = evaluate_joint(
            model, loaders["validation"], device
        )
        validation_stim_r = float(
            np.nanmean(
                stimulus_correlation(validation_predictions, validation_targets)
            )
        )
        scheduler.step(-validation_stim_r)
        history.append(
            {
                "epoch": epoch,
                "train_mse": train_squared_error / train_values,
                "validation_mse": float(
                    np.mean((validation_predictions - validation_targets) ** 2)
                ),
                "validation_stim_r": validation_stim_r,
            }
        )
        print(
            f"    epoch {epoch:03d}/{cfg.epochs:03d} | train "
            f"{history[-1]['train_mse']:.6f} | validation stim_r "
            f"{validation_stim_r:.4f}"
        )

        if validation_stim_r > best_validation_stim_r + 1e-8:
            best_validation_stim_r = validation_stim_r
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        # end if this epoch improved the selection criterion

        if (
            epoch >= cfg.minimum_epochs
            and epochs_without_improvement >= cfg.patience
        ):
            print(f"    early stop after {epoch} epochs")
            break
        # end if early-stopping patience is exhausted
    # end for optimization epoch

    model.load_state_dict(best_state)
    return history, best_epoch, float(best_validation_stim_r)
# EOF


"""
describe_optimization
Summarize whether a run had converged inside its epoch budget.

INPUT:
    - history: list[dict] -> per-epoch records
    - best_epoch: int -> selected checkpoint epoch
    - cfg: Cfg -> epoch budget

OUTPUT:
    - description: dict -> epochs run, trailing slopes, and a still-improving flag
"""
def describe_optimization(history, best_epoch, cfg):
    trailing = history[-TREND_WINDOW:]
    epochs = np.arange(len(trailing), dtype=float)
    slopes = {}
    for field_name in ("train_mse", "validation_stim_r"):
        values = np.array([entry[field_name] for entry in trailing], dtype=float)
        slopes[field_name] = (
            float(np.polyfit(epochs, values, 1)[0]) if len(values) > 1 else float("nan")
        )
    # end for tracked quantity
    return {
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "stopped_early": len(history) < cfg.epochs,
        "train_mse_percent_per_epoch": round(
            100.0 * slopes["train_mse"] / history[-1]["train_mse"], 4
        ),
        "validation_stim_r_slope_per_epoch": round(
            slopes["validation_stim_r"], 6
        ),
        "still_improving_at_budget": bool(
            len(history) == cfg.epochs
            and best_epoch >= len(history) - 2
            and slopes["validation_stim_r"] > 0.0
        ),
    }
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
    return caches, tuple(caches["train"].shape[1:])
# EOF


"""
save_finished_variant
Store one finished variant so a later run reuses it instead of repeating it.

INPUT:
    - variant_dir: Path -> directory holding one file per finished variant
    - variant_name: str -> the variant's name
    - row: dict -> its metrics
    - history: list[dict] -> its per-epoch record
    - site_correlations: np.ndarray -> [time, neurons] test stim_r

OUTPUT:
    - None: writes the variant's record and its per-site correlations
"""
def save_finished_variant(variant_dir, variant_name, row, history, site_correlations):
    variant_dir.mkdir(parents=True, exist_ok=True)
    np.save(variant_dir / f"{variant_name}_site_stim_r.npy", site_correlations)
    with open(variant_dir / f"{variant_name}.json", "w") as variant_file:
        json.dump({"row": row, "history": history}, variant_file, indent=2)
    # end with saved variant record
# EOF


"""
load_finished_variant
Read one stored variant back, or report that it has not been run yet.

INPUT:
    - variant_dir: Path -> directory holding one file per finished variant
    - variant_name: str -> the variant to look for

OUTPUT:
    - finished: tuple | None -> (row, history, site_correlations), or None
"""
def load_finished_variant(variant_dir, variant_name):
    record_path = variant_dir / f"{variant_name}.json"
    site_path = variant_dir / f"{variant_name}_site_stim_r.npy"
    if not (record_path.is_file() and site_path.is_file()):
        return None
    # end if this variant has not finished before
    with open(record_path, "r") as variant_file:
        record = json.load(variant_file)
    # end with stored variant record
    return record["row"], record["history"], np.load(site_path)
# EOF


def main():
    cfg = parse_args()
    if cfg.smoke_test:
        cfg.epochs = min(cfg.epochs, 2)
        cfg.minimum_epochs = 1
        cfg.patience = 1
        cfg.noise_ceiling_resamples = 4
    # end if a smoke run was requested

    paths = load_project_paths(cfg)
    device = resolve_device(cfg.device)
    temporal_configuration, search_dir = load_temporal_configuration(cfg)
    backbone_label = cfg.spatial_stem.replace("tvsd_monkeyF_", "").replace(
        "_spatial", ""
    )
    output_dir = Path(
        cfg.output_dir
        or PROJECT_ROOT / "results" / f"tvsd_joint_spatial_{backbone_label}"
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = prepare_search_data(cfg, paths, device)
    scoring, shapes = data["scoring"], data["shapes"]
    subset_targets, subset_features = data["subset_targets"], data["subset_features"]
    caches, feature_shape = load_spatial_cache(cfg, paths)
    allmat, indices = data["allmat"], data["indices"]

    # Rows of the map cache follow the one-based stimulus identifiers.
    image_rows = {
        "train": allmat[indices["train"], 1] - 1,
        "validation": allmat[indices["validation"], 1] - 1,
        "test": allmat[indices["test"], 2] - 1,
    }
    cache_of = {"train": "train", "validation": "train", "test": "test"}

    def build_loaders(load_maps):
        """Build the three loaders, with or without the conv maps."""
        return {
            subset_name: DataLoader(
                JointFeatureDataset(
                    subset_features[subset_name],
                    caches[cache_of[subset_name]],
                    image_rows[subset_name],
                    subset_targets[subset_name],
                    load_maps=load_maps,
                ),
                batch_size=cfg.batch_size,
                shuffle=subset_name == "train",
                num_workers=cfg.num_workers,
                generator=torch.Generator().manual_seed(cfg.random_seed),
            )
            for subset_name in ("train", "validation", "test")
        }
    # EOF
    print(
        f"pooled {cfg.model_name} {len(cfg.layer_names)} depths + spatial "
        f"{backbone_label} {feature_shape} | {cfg.temporal_architecture} "
        f"configuration from {search_dir.name}"
    )

    temporal_kwargs = {**shapes, **temporal_configuration}
    spatial_kwargs = {
        "n_channels": feature_shape[0],
        "spatial_size": feature_shape[1:],
        "n_timepoints": shapes["n_timepoints"],
        "n_neurons": shapes["n_neurons"],
        "hidden_dim": cfg.spatial_hidden_dim,
        "dropout": cfg.spatial_dropout,
        "mask_temperature": cfg.mask_temperature,
    }

    variant_names = [
        name.strip() for name in cfg.variants.split(",") if name.strip()
    ]
    unknown = sorted(set(variant_names) - set(VARIANT_BRANCHES))
    if unknown:
        raise KeyError(f"Unknown variants: {unknown}.")
    # end if a variant name is invalid

    # Each finished variant is written before the next one starts, so an
    # interrupted comparison resumes instead of repeating what it has.
    variant_dir = output_dir / "variants"
    result_rows, histories, site_correlations = [], {}, {}
    for variant_name in variant_names:
        use_temporal, use_spatial = VARIANT_BRANCHES[variant_name]
        finished = load_finished_variant(variant_dir, variant_name)
        if finished is not None:
            row, history, site_r = finished
            result_rows.append(row)
            histories[variant_name] = history
            site_correlations[variant_name] = site_r
            print(
                f"\nreusing {variant_name}: test stim_r "
                f"{row['mean_stim_r_response']:.4f}"
            )
            continue
        # end if this variant has already been run
        loaders = build_loaders(use_spatial)
        print(f"\ntraining {variant_name}")
        torch.manual_seed(cfg.random_seed)
        model = JointSpatialTemporalDecoder(
            cfg.temporal_architecture,
            temporal_kwargs,
            cfg.spatial_readout,
            spatial_kwargs,
            use_temporal=use_temporal,
            use_spatial=use_spatial,
        ).to(device)
        run_start = time.perf_counter()
        history, best_epoch, best_validation_stim_r = train_joint_decoder(
            cfg, model, loaders, device
        )
        train_seconds = time.perf_counter() - run_start

        predictions, targets = evaluate_joint(model, loaders["test"], device)
        row, site_r = score_and_decompose(
            predictions, targets, scoring, variant_name, REDUCER
        )
        row.update(
            {
                "trainable_parameters": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
                "validation_stim_r": round(best_validation_stim_r, 4),
                "train_seconds": round(train_seconds, 1),
                **describe_optimization(history, best_epoch, cfg),
            }
        )
        if use_spatial:
            masks = model.spatial_branch.spatial_masks().detach().cpu().numpy()
            np.save(output_dir / f"site_spatial_masks_{variant_name}.npy", masks)
            entropy = -(masks * np.log(masks + 1e-12)).sum(axis=(1, 2))
            row["mean_mask_entropy"] = round(float(entropy.mean()), 3)
            row["uniform_mask_entropy"] = round(float(np.log(masks[0].size)), 3)
            print(
                f"  spatial masks: mean entropy {entropy.mean():.3f} of "
                f"{np.log(masks[0].size):.3f} for a flat mask"
            )
        # end if this variant owns spatial masks
        result_rows.append(row)
        histories[variant_name] = history
        site_correlations[variant_name] = site_r
        save_finished_variant(variant_dir, variant_name, row, history, site_r)
        print(
            f"  test stim_r {row['mean_stim_r_response']:.4f} | frac ceiling "
            f"{row['fraction_of_ceiling']:.3f} | epoch {best_epoch:02d}/"
            f"{row['epochs_run']:02d} | {train_seconds:.0f}s"
        )
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        # end if the device caches its allocations
    # end for variant

    with open(output_dir / "config.json", "w") as config_file:
        json.dump(
            {
                **asdict(cfg),
                "temporal_configuration": temporal_configuration,
                "feature_shape": list(feature_shape),
                "bin_edges_ms": data["bin_edges_ms"].tolist(),
            },
            config_file,
            indent=2,
        )
    # end with saved configuration
    with open(output_dir / "results.json", "w") as results_file:
        json.dump({"rows": result_rows, "histories": histories}, results_file, indent=2)
    # end with saved metrics
    np.savez_compressed(
        output_dir / "site_stim_r.npz",
        ceiling=scoring["ceiling"],
        **site_correlations,
    )

    header = (
        f"{'variant':<16}{'test r':>9}{'frac ceil':>11}{'MSE':>10}"
        f"{'MSE rescaled':>14}{'scale':>8}{'params':>12}{'epoch':>8}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in result_rows:
        print(
            f"{row['model']:<16}{row['mean_stim_r_response']:>9.4f}"
            f"{row['fraction_of_ceiling']:>11.3f}{row['test_mse']:>10.5f}"
            f"{row['mse_after_rescaling']:>14.5f}{row['mean_optimal_scale']:>8.3f}"
            f"{row['trainable_parameters']:>12,}"
            f"{row['best_epoch']:>4d}/{row['epochs_run']:<3d}"
        )
    # end for scored variant
    print("\nWas the loss still decreasing at the epoch budget?")
    for row in result_rows:
        verdict = (
            "still improving"
            if row["still_improving_at_budget"]
            else ("converged" if row["stopped_early"] else "flat at budget")
        )
        print(
            f"  {row['model']:<16}{verdict:<17}train MSE "
            f"{row['train_mse_percent_per_epoch']:+.3f}%/epoch | validation "
            f"stim_r {row['validation_stim_r_slope_per_epoch']:+.5f}/epoch"
        )
    # end for scored variant
    print(f"\nsaved to {output_dir}")
# EOF


if __name__ == "__main__":
    main()
# EOC
