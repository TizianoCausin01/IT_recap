"""
Search for a nonlinear TVSD decoder that beats the joint spatial baseline.

The reference is the `joint` variant of run_tvsd_joint_spatial_decoder.py: a
GRU over three pooled I-JEPA depths summed with a factorized AlexNet conv5
readout, trained under MSE, 0.5207 test stim_r against a 0.774 ceiling. Every
variant here reuses that protocol exactly -- same window, same 20 ms bins, same
320 IT sites, same split seed, same validation-stim_r selection -- and changes
one thing, so the comparison is a controlled one and the numbers are directly
comparable to the published table.

No new features are extracted: the four frozen caches this repository already
holds (pooled I-JEPA, pooled DINOv3, the AlexNet conv5 map, the ConvNeXt stage-4
map) are the whole input space. What varies is the objective, the noise and
augmentation applied to those caches, the readout's own nonlinearity, and what
else the decoder is asked to predict on the side.

Variants are written to disk as they finish, so an interrupted sweep resumes.
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
from torch.utils.data import DataLoader, Dataset


sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "python_scripts" / "src")
)

from IT_recap.neural_prediction_training import stimulus_correlation  # noqa: E402
from IT_recap.tvsd_experiments import (  # noqa: E402
    PROJECT_ROOT,
    load_cached_targets,
    load_project_paths,
    prepare_search_data,
    prepare_timebin_data,
    resolve_device,
    score_and_decompose,
)
from IT_recap.tvsd_objectives import (  # noqa: E402
    WeightAverage,
    add_relative_noise,
    apply_output_calibration,
    build_objective,
    drop_feature_groups,
    fit_output_calibration,
    masked_prediction_loss,
    mixup_batch,
    random_translate_maps,
)
from model_classes.tvsd_nonlinear_models import (  # noqa: E402
    MultiBranchTimebinDecoder,
)


REDUCER = "mean"
TREND_WINDOW = 10

# Hyperparameters of the pooled branch, read out of the finished search.
TEMPORAL_HYPERPARAMETERS = (
    "hidden_dim",
    "time_embedding_dim",
    "dropout",
    "input_noise_std",
    "temporal_noise_std",
)

ALEXNET_STEM = "tvsd_monkeyF_alexnet_features_11_spatial"
CONVNEXT_STEM = "tvsd_monkeyF_convnext_tiny_features_7_spatial"


@dataclass
class Cfg:
    # Caches and the finished runs this experiment builds on.
    env: str | None = None
    mua_file_name: str = "f_THINGS_MUA_trials.mat"
    feature_archive_name: str = "tvsd_monkeyF_ijepa_vith14_1k_224_features.npz"
    search_results_dir: str | None = None
    output_dir: str | None = None
    spatial_stems: str = ALEXNET_STEM

    # Target identity and window, identical to the joint spatial experiment.
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

    # Branches.
    temporal_architecture: str = "gru"
    spatial_hidden_dim: int = 64
    spatial_dropout: float = 0.3
    mask_temperature: float = 1.0
    n_masks: int = 1
    core_layers: int = 0
    core_dim: int = 64

    # Objective. Sites are laid out [IT, auxiliary]; only IT is ever scored.
    objective: str = "mse"
    correlation_weight: float = 0.5
    huber_delta: float = 1.0
    auxiliary_area: str = ""
    auxiliary_weight: float = 0.3
    reconstruction_weight: float = 0.0
    reconstruction_layer: int = 2

    # Noise and augmentation applied to the frozen caches.
    mixup_alpha: float = 0.0
    map_shift: float = 0.0
    map_noise_std: float = 0.0
    group_dropout: float = 0.0
    ema_decay: float = 0.0

    # Optimization, following the joint spatial experiment.
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
    mask_learning_rate_scale: float = 10.0

    # Which variants to run, and bookkeeping.
    variants: str = ""
    noise_ceiling_resamples: int = 40
    device: str = "auto"
    smoke_test: bool = False


# One controlled change each, on top of the reproduced `joint` reference.
VARIANTS = {
    # --- reference ---
    "joint_mse": {},
    # --- objectives ---
    "obj_correlation": {"objective": "correlation"},
    "obj_mse_corr": {"objective": "mse_correlation", "correlation_weight": 0.5},
    "obj_mse_corr_light": {
        "objective": "mse_correlation", "correlation_weight": 0.2
    },
    "obj_huber": {"objective": "huber", "huber_delta": 1.0},
    # --- auxiliary objectives ---
    "aux_v4": {"auxiliary_area": "V4", "auxiliary_weight": 0.3},
    "aux_v4_heavy": {"auxiliary_area": "V4", "auxiliary_weight": 1.0},
    "aux_reconstruct": {"reconstruction_weight": 0.1},
    # --- noise and augmentation ---
    "aug_mixup": {"mixup_alpha": 0.4},
    "aug_map_shift": {"map_shift": 1.0},
    "aug_map_noise": {"map_noise_std": 0.25},
    "aug_group_dropout": {"group_dropout": 0.2},
    "reg_ema": {"ema_decay": 0.999},
    # --- readout nonlinearity ---
    "arch_multimask": {"n_masks": 4},
    "arch_core1": {"core_layers": 1, "core_dim": 128},
    "arch_core2": {"core_layers": 2, "core_dim": 128},
    "arch_two_maps": {"spatial_stems": f"{ALEXNET_STEM},{CONVNEXT_STEM}"},
}

# Wave two: combinations of the single changes that survived wave one, plus
# seed replicates. A trained decoder on this task moves by ~0.005 between
# seeds, so nothing below that margin is called a result off one run.
COMBINATION_VARIANTS = {
    "combo_corr_v4": {"objective": "correlation", "auxiliary_area": "V4"},
    "combo_corr_noise": {"objective": "correlation", "map_noise_std": 0.25},
    "combo_corr_v4_noise": {
        "objective": "correlation", "auxiliary_area": "V4", "map_noise_std": 0.25,
    },
    "combo_corr_v4_noise_shift": {
        "objective": "correlation", "auxiliary_area": "V4",
        "map_noise_std": 0.25, "map_shift": 1.0,
    },
    "combo_v4_noise": {"auxiliary_area": "V4", "map_noise_std": 0.25},
}
VARIANTS.update(COMBINATION_VARIANTS)

# Seed replicates of the reference and of every combination, so a winning
# margin can be read against the spread it has to clear.
for _seeded in ["joint_mse", *COMBINATION_VARIANTS]:
    for _seed in (1, 2):
        VARIANTS[f"{_seeded}_s{_seed}"] = {
            **VARIANTS[_seeded], "random_seed": _seed
        }
    # end for replicate seed
# end for replicated variant


"""
parse_args
Parse command-line overrides into a configuration object.

OUTPUT:
    - cfg: Cfg -> caches, branches, objective, and optimization settings
"""
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    default_layers = Cfg.__dataclass_fields__["layer_names"].default_factory()
    for field_name, field_definition in Cfg.__dataclass_fields__.items():
        if field_name == "layer_names":
            parser.add_argument("--layer_names", default=",".join(default_layers))
            continue
        # end if the layer selection needs list parsing
        default = field_definition.default
        argument_name = f"--{field_name}"
        if isinstance(default, bool):
            parser.add_argument(
                argument_name, action=argparse.BooleanOptionalAction, default=default
            )
        elif default is None:
            parser.add_argument(argument_name, default=default)
        else:
            parser.add_argument(argument_name, type=type(default), default=default)
        # end if boolean, optional string, or typed argument
    # end for configuration field
    arguments = vars(parser.parse_args())
    arguments["layer_names"] = [
        name.strip() for name in arguments["layer_names"].split(",") if name.strip()
    ]
    return Cfg(**arguments)
# EOF


class MultiMapDataset(Dataset):
    """
    Serve pooled layer features, one or more conv maps, and the target.

    The map caches are float16 and about two gigabytes each, so they stay
    memory-mapped and one presentation is converted at a time.

    INPUT (__getitem__):
        - index: int -> position within this subset

    OUTPUT:
        - layer_features: torch.Tensor -> [layers, embedding]
        - feature_maps: list[torch.Tensor] -> one [channels, height, width] each
        - target: torch.Tensor -> [time, sites] standardized response
        - image_row: int -> the stimulus this presentation shows, so an
          auxiliary target defined per stimulus survives shuffling
    """

    def __init__(self, layer_features, map_caches, image_rows, targets):
        self.layer_features = np.asarray(layer_features, dtype=np.float32)
        self.map_caches = list(map_caches)
        self.image_rows = np.asarray(image_rows, dtype=int)
        self.targets = np.asarray(targets, dtype=np.float32)
        if len({len(self.layer_features), len(self.image_rows), len(self.targets)}) != 1:
            raise ValueError("Features, stimulus rows, and targets must align.")
        # end if the presentation axes disagree
        for cache in self.map_caches:
            if self.image_rows.max(initial=0) >= len(cache):
                raise IndexError("A presentation indexes a missing feature map.")
            # end if a stimulus is absent from a map cache
        # end for map cache

    def __len__(self):
        return len(self.targets)
    # EOF

    def __getitem__(self, index):
        row = self.image_rows[index]
        maps = [
            torch.from_numpy(np.asarray(cache[row], dtype=np.float32))
            for cache in self.map_caches
        ]
        return (
            torch.from_numpy(self.layer_features[index]),
            maps,
            torch.from_numpy(self.targets[index]),
            int(row),
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
        or PROJECT_ROOT / "results" / f"tvsd_timebin_architecture_search_{cfg.model_name}"
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
    return {name: row[name] for name in TEMPORAL_HYPERPARAMETERS if name in row}, results_dir
# EOF


"""
load_auxiliary_targets
Bin, split, and standardize a second area's response on the primary's split.

The split is a deterministic function of the seed and the presentation
metadata, so running the shared preparation again for another area lines its
rows up with the primary target row for row.

INPUT:
    - cfg: Cfg -> window and split settings of the primary experiment
    - paths: dict -> active project paths
    - data: dict -> prepared primary data, reused for its features and indices
    - device: torch.device -> compute device, only reported

OUTPUT:
    - subset_targets: dict -> subset name to [presentations, time, sites]
"""
def load_auxiliary_targets(cfg, paths, data, device):
    auxiliary_cfg = replace(cfg, area=cfg.auxiliary_area)
    targets, allmat = load_cached_targets(auxiliary_cfg, paths)
    auxiliary = prepare_timebin_data(
        auxiliary_cfg,
        targets,
        data["train_features"],
        data["test_features"],
        allmat,
        device,
    )
    for subset_name, subset_indices in auxiliary["indices"].items():
        if not np.array_equal(subset_indices, data["indices"][subset_name]):
            raise ValueError(
                f"The {cfg.auxiliary_area} split does not match the "
                f"{cfg.area} split; the seed or window must differ."
            )
        # end if the two areas were split differently
    # end for subset
    return auxiliary["subset_targets"]
# EOF


"""
build_parameter_groups
Give the pooled branch, the spatial branches, and the masks their own settings.

INPUT:
    - model: MultiBranchTimebinDecoder -> the model being optimized
    - cfg: Cfg -> per-branch learning rates and weight decays

OUTPUT:
    - groups: list[dict] -> optimizer parameter groups
"""
def build_parameter_groups(model, cfg):
    mask_names = set(model.mask_parameter_names())
    groups = []
    for prefix, learning_rate, weight_decay in (
        ("temporal_branch", cfg.temporal_learning_rate, cfg.temporal_weight_decay),
        ("spatial_branches", cfg.spatial_learning_rate, cfg.spatial_weight_decay),
        ("reconstruction_head", cfg.spatial_learning_rate, cfg.spatial_weight_decay),
    ):
        parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if name.startswith(prefix) and name not in mask_names
        ]
        if parameters:
            groups.append(
                {"params": parameters, "lr": learning_rate, "weight_decay": weight_decay}
            )
        # end if this branch is present
    # end for branch

    mask_parameters = [
        parameter for name, parameter in model.named_parameters() if name in mask_names
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
evaluate_branches
Run a decoder over a loader and collect its primary-site predictions.

INPUT:
    - model: nn.Module -> decoder under evaluation
    - loader: DataLoader -> evaluation batches
    - device: torch.device -> compute device
    - n_primary: int -> sites the experiment is scored on

OUTPUT:
    - predictions: np.ndarray -> [presentations, time, n_primary]
    - targets: np.ndarray -> [presentations, time, n_primary]
"""
def evaluate_branches(model, loader, device, n_primary):
    model.eval()
    prediction_batches, target_batches = [], []
    with torch.no_grad():
        for layer_features, feature_maps, targets, _ in loader:
            predictions, _ = model(
                layer_features.to(device),
                [feature_map.to(device) for feature_map in feature_maps],
            )
            prediction_batches.append(predictions[..., :n_primary].cpu())
            target_batches.append(targets[..., :n_primary])
        # end for evaluation batch
    # end with no gradient tracking
    return torch.cat(prediction_batches).numpy(), torch.cat(target_batches).numpy()
# EOF


"""
train_variant
Optimize one variant and restore the checkpoint with the best validation stim_r.

INPUT:
    - cfg: Cfg -> objective, augmentation, and optimization settings
    - model: nn.Module -> decoder on the device
    - loaders: dict -> train and validation loaders
    - device: torch.device -> compute device
    - n_primary: int -> sites the experiment is scored on
    - reconstruction_targets: np.ndarray | None -> [stimuli, width] auxiliary

OUTPUT:
    - model: nn.Module -> the model to score, averaged weights when requested
    - history: list[dict] -> per-epoch train loss and validation stim_r
    - best_epoch: int -> selected checkpoint epoch
    - best_validation_stim_r: float -> validation stim_r at that epoch
"""
def train_variant(cfg, model, loaders, device, n_primary, reconstruction_targets):
    optimizer = torch.optim.AdamW(build_parameter_groups(model, cfg))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(3, cfg.patience // 3)
    )
    objective = build_objective(
        cfg.objective, cfg.correlation_weight, cfg.huber_delta
    )
    averager = WeightAverage(model, cfg.ema_decay) if cfg.ema_decay > 0.0 else None
    reconstruction = (
        torch.from_numpy(np.asarray(reconstruction_targets, dtype=np.float32))
        if reconstruction_targets is not None
        else None
    )

    best_state = copy.deepcopy(model.state_dict())
    best_validation_stim_r = -np.inf
    best_epoch, epochs_without_improvement = 0, 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss, n_batches = 0.0, 0
        for layer_features, feature_maps, targets, batch_rows in loaders["train"]:
            layer_features = layer_features.to(device)
            feature_maps = [feature_map.to(device) for feature_map in feature_maps]
            targets = targets.to(device)

            # Augmentation acts on the frozen caches, before any branch sees them.
            layer_features = drop_feature_groups(layer_features, cfg.group_dropout)
            feature_maps = [
                add_relative_noise(
                    random_translate_maps(feature_map, cfg.map_shift),
                    cfg.map_noise_std,
                )
                for feature_map in feature_maps
            ]
            mixed, targets = mixup_batch(
                [layer_features, *feature_maps], targets, cfg.mixup_alpha
            )
            layer_features, feature_maps = mixed[0], list(mixed[1:])

            optimizer.zero_grad(set_to_none=True)
            predictions, diagnostics = model(
                layer_features,
                feature_maps,
                return_diagnostics=reconstruction is not None,
            )
            loss = masked_prediction_loss(
                predictions, targets, n_primary, objective, cfg.auxiliary_weight
            )
            if reconstruction is not None and cfg.mixup_alpha <= 0.0:
                # Mixed inputs have no single frozen feature vector to match,
                # so the reconstruction term is only applied to clean batches.
                loss = loss + cfg.reconstruction_weight * torch.nn.functional.mse_loss(
                    diagnostics["reconstruction"], reconstruction[batch_rows].to(device)
                )
            # end if the reconstruction auxiliary is active
            loss.backward()
            if cfg.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            # end if gradient clipping is enabled
            optimizer.step()
            if averager is not None:
                averager.update(model)
            # end if weights are being averaged
            running_loss += float(loss.detach())
            n_batches += 1
        # end for training batch

        scored = averager.module if averager is not None else model
        validation_predictions, validation_targets = evaluate_branches(
            scored, loaders["validation"], device, n_primary
        )
        validation_stim_r = float(
            np.nanmean(stimulus_correlation(validation_predictions, validation_targets))
        )
        scheduler.step(-validation_stim_r)
        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / max(1, n_batches),
                "validation_mse": float(
                    np.mean((validation_predictions - validation_targets) ** 2)
                ),
                "validation_stim_r": validation_stim_r,
            }
        )
        print(
            f"    epoch {epoch:03d}/{cfg.epochs:03d} | train "
            f"{history[-1]['train_loss']:.6f} | validation stim_r "
            f"{validation_stim_r:.4f}",
            flush=True,
        )

        if validation_stim_r > best_validation_stim_r + 1e-8:
            best_validation_stim_r = validation_stim_r
            best_epoch = epoch
            best_state = copy.deepcopy(scored.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        # end if this epoch improved the selection criterion

        if epoch >= cfg.minimum_epochs and epochs_without_improvement >= cfg.patience:
            print(f"    early stop after {epoch} epochs")
            break
        # end if early-stopping patience is exhausted
    # end for optimization epoch

    scored = averager.module if averager is not None else model
    scored.load_state_dict(best_state)
    return scored, history, best_epoch, float(best_validation_stim_r)
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
    for field_name in ("train_loss", "validation_stim_r"):
        values = np.array([entry[field_name] for entry in trailing], dtype=float)
        slopes[field_name] = (
            float(np.polyfit(epochs, values, 1)[0]) if len(values) > 1 else float("nan")
        )
    # end for tracked quantity
    return {
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "stopped_early": len(history) < cfg.epochs,
        "validation_stim_r_slope_per_epoch": round(slopes["validation_stim_r"], 6),
        "still_improving_at_budget": bool(
            len(history) == cfg.epochs
            and best_epoch >= len(history) - 2
            and slopes["validation_stim_r"] > 0.0
        ),
    }
# EOF


"""
load_spatial_caches
Open the cached feature maps named by a configuration.

INPUT:
    - stems: list[str] -> cache stems under data_path/models
    - paths: dict -> active project paths

OUTPUT:
    - caches: dict -> stem to {split: memory-mapped [stimuli, C, H, W]}
    - shapes: list[tuple] -> cached [channels, height, width] per stem
"""
def load_spatial_caches(stems, paths):
    cache_dir = Path(paths["data_path"]) / "models"
    caches, shapes = {}, []
    for stem in stems:
        per_split = {}
        for split_name in ("train", "test"):
            cache_path = cache_dir / f"{stem}_{split_name}.npy"
            if not cache_path.is_file():
                raise FileNotFoundError(
                    f"Missing {cache_path}. Run extract_tvsd_spatial_features.py."
                )
            # end if the cache is absent
            per_split[split_name] = np.load(cache_path, mmap_mode="r")
        # end for stimulus split
        caches[stem] = per_split
        shapes.append(tuple(per_split["train"].shape[1:]))
    # end for cache stem
    return caches, shapes
# EOF


"""
run_variant
Build, train, and score one named variant under its configuration overrides.

INPUT:
    - cfg: Cfg -> the variant's fully resolved configuration
    - context: dict -> shared data, caches, and scoring material

OUTPUT:
    - row: dict -> metrics of this variant
    - history: list[dict] -> its per-epoch record
    - site_correlations: np.ndarray -> [time, neurons] test stim_r
"""
def run_variant(cfg, context):
    device, data = context["device"], context["data"]
    shapes, scoring = data["shapes"], data["scoring"]
    n_primary = shapes["n_neurons"]

    stems = [stem.strip() for stem in cfg.spatial_stems.split(",") if stem.strip()]
    caches, map_shapes = context["caches"], []
    for stem in stems:
        map_shapes.append(tuple(caches[stem]["train"].shape[1:]))
    # end for requested map

    # Auxiliary sites are appended to the target axis and dropped at scoring.
    subset_targets = dict(data["subset_targets"])
    if cfg.auxiliary_area:
        auxiliary = context["auxiliary_targets"][cfg.auxiliary_area]
        subset_targets = {
            name: np.concatenate([subset_targets[name], auxiliary[name]], axis=-1)
            for name in subset_targets
        }
    # end if this variant trains on a second area
    n_outputs = subset_targets["train"].shape[-1]

    cache_of = {"train": "train", "validation": "train", "test": "test"}
    loaders = {
        subset_name: DataLoader(
            MultiMapDataset(
                data["subset_features"][subset_name],
                [caches[stem][cache_of[subset_name]] for stem in stems],
                context["image_rows"][subset_name],
                subset_targets[subset_name],
            ),
            batch_size=cfg.batch_size,
            shuffle=subset_name == "train",
            num_workers=cfg.num_workers,
            generator=torch.Generator().manual_seed(cfg.random_seed),
        )
        for subset_name in ("train", "validation", "test")
    }

    temporal_kwargs = {
        **{k: v for k, v in shapes.items() if k != "n_neurons"},
        "n_neurons": n_outputs,
        **context["temporal_configuration"],
    }
    spatial_specs = [
        {
            "n_channels": map_shape[0],
            "spatial_size": map_shape[1:],
            "n_timepoints": shapes["n_timepoints"],
            "n_neurons": n_outputs,
            "hidden_dim": cfg.spatial_hidden_dim,
            "dropout": cfg.spatial_dropout,
            "mask_temperature": cfg.mask_temperature,
            "n_masks": cfg.n_masks,
            "core_layers": cfg.core_layers,
            "core_dim": cfg.core_dim,
        }
        for map_shape in map_shapes
    ]
    reconstruction_targets = None
    reconstruction_dim = 0
    if cfg.reconstruction_weight > 0.0:
        # Regressing a frozen pooled depth out of the conv core is an
        # auxiliary objective on the readout, not an extra input.
        reconstruction_targets = data["train_features"][:, cfg.reconstruction_layer]
        reconstruction_dim = reconstruction_targets.shape[1]
    # end if the reconstruction auxiliary is enabled

    torch.manual_seed(cfg.random_seed)
    model = MultiBranchTimebinDecoder(
        cfg.temporal_architecture or None,
        temporal_kwargs,
        spatial_specs,
        reconstruction_dim=reconstruction_dim,
    ).to(device)

    run_start = time.perf_counter()
    model, history, best_epoch, best_validation_stim_r = train_variant(
        cfg, model, loaders, device, n_primary, reconstruction_targets
    )
    train_seconds = time.perf_counter() - run_start

    predictions, targets = evaluate_branches(model, loaders["test"], device, n_primary)
    row, site_correlations = score_and_decompose(
        predictions, targets, scoring, cfg.variants, REDUCER
    )
    # An affine correction per (bin, site), fitted on validation images only,
    # makes raw MSE comparable between objectives that fix the output scale and
    # objectives that do not. It leaves stim_r untouched by construction.
    validation_predictions, validation_targets = evaluate_branches(
        model, loaders["validation"], device, n_primary
    )
    slope, intercept = fit_output_calibration(
        validation_predictions, validation_targets
    )
    calibrated_row, _ = score_and_decompose(
        apply_output_calibration(predictions, slope, intercept),
        targets, scoring, cfg.variants, REDUCER,
    )
    row.update(
        {
            "calibrated_test_mse": calibrated_row["test_mse"],
            "calibrated_stim_r": calibrated_row["mean_stim_r_response"],
            "calibration_slope": round(float(np.mean(slope)), 4),
        }
    )
    masks = model.spatial_branches[0].spatial_masks().detach().cpu().numpy()
    entropy = -(masks * np.log(masks + 1e-12)).sum(axis=(-2, -1))
    row.update(
        {
            # Counted without a requires_grad filter: an averaged copy of the
            # weights has none, and would otherwise be reported as zero.
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "validation_stim_r": round(best_validation_stim_r, 4),
            "train_seconds": round(train_seconds, 1),
            "mean_mask_entropy": round(float(entropy.mean()), 3),
            "uniform_mask_entropy": round(float(np.log(masks[0, 0].size)), 3),
            **describe_optimization(history, best_epoch, cfg),
        }
    )
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    # end if the device caches its allocations
    return row, history, site_correlations
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
    output_dir = Path(
        cfg.output_dir or PROJECT_ROOT / "results" / "tvsd_nonlinear_search"
    ).expanduser()
    variant_dir = output_dir / "variants"
    variant_dir.mkdir(parents=True, exist_ok=True)

    requested = [name.strip() for name in cfg.variants.split(",") if name.strip()]
    requested = requested or list(VARIANTS)
    unknown = sorted(set(requested) - set(VARIANTS))
    if unknown:
        raise KeyError(f"Unknown variants: {unknown}.")
    # end if a variant name is invalid

    data = prepare_search_data(cfg, paths, device)
    temporal_configuration, search_dir = load_temporal_configuration(cfg)
    allmat, indices = data["allmat"], data["indices"]
    image_rows = {
        "train": allmat[indices["train"], 1] - 1,
        "validation": allmat[indices["validation"], 1] - 1,
        "test": allmat[indices["test"], 2] - 1,
    }

    # Every cache any requested variant needs, opened once.
    stems = sorted(
        {
            stem.strip()
            for name in requested
            for stem in VARIANTS[name].get("spatial_stems", cfg.spatial_stems).split(",")
            if stem.strip()
        }
    )
    caches, _ = load_spatial_caches(stems, paths)
    auxiliary_areas = sorted(
        {
            VARIANTS[name]["auxiliary_area"]
            for name in requested
            if VARIANTS[name].get("auxiliary_area")
        }
    )
    auxiliary_targets = {}
    for area in auxiliary_areas:
        auxiliary_targets[area] = load_auxiliary_targets(
            replace(cfg, auxiliary_area=area), paths, data, device
        )
        print(
            f"auxiliary {area}: "
            f"{auxiliary_targets[area]['train'].shape[-1]} sites"
        )
    # end for auxiliary area

    context = {
        "device": device,
        "data": data,
        "caches": caches,
        "image_rows": image_rows,
        "temporal_configuration": temporal_configuration,
        "auxiliary_targets": auxiliary_targets,
    }
    print(
        f"pooled {cfg.model_name} {len(cfg.layer_names)} depths | "
        f"{cfg.temporal_architecture} configuration from {search_dir.name} | "
        f"{len(requested)} variants"
    )

    result_rows, histories, site_correlations = [], {}, {}
    for name in requested:
        record_path = variant_dir / f"{name}.json"
        site_path = variant_dir / f"{name}_site_stim_r.npy"
        if record_path.is_file() and site_path.is_file():
            with open(record_path, "r") as variant_file:
                record = json.load(variant_file)
            # end with stored variant record
            result_rows.append(record["row"])
            histories[name] = record["history"]
            site_correlations[name] = np.load(site_path)
            print(
                f"\nreusing {name}: test stim_r "
                f"{record['row']['mean_stim_r_response']:.4f}"
            )
            continue
        # end if this variant has already been run
        print(f"\ntraining {name}: {VARIANTS[name] or 'reference configuration'}")
        variant_cfg = replace(cfg, variants=name, **VARIANTS[name])
        row, history, site_r = run_variant(variant_cfg, context)
        result_rows.append(row)
        histories[name] = history
        site_correlations[name] = site_r
        np.save(site_path, site_r)
        with open(record_path, "w") as variant_file:
            json.dump(
                {"row": row, "history": history, "overrides": VARIANTS[name]},
                variant_file,
                indent=2,
            )
        # end with saved variant record
        print(
            f"  test stim_r {row['mean_stim_r_response']:.4f} | frac ceiling "
            f"{row['fraction_of_ceiling']:.3f} | MSE {row['test_mse']:.5f} | "
            f"{row['train_seconds']:.0f}s"
        )
    # end for requested variant

    with open(output_dir / "config.json", "w") as config_file:
        json.dump(
            {**asdict(cfg), "variant_overrides": VARIANTS,
             "bin_edges_ms": data["bin_edges_ms"].tolist()},
            config_file,
            indent=2,
        )
    # end with saved configuration
    with open(output_dir / "results.json", "w") as results_file:
        json.dump({"rows": result_rows, "histories": histories}, results_file, indent=2)
    # end with saved metrics
    np.savez_compressed(
        output_dir / "site_stim_r.npz",
        ceiling=data["scoring"]["ceiling"],
        **site_correlations,
    )

    ordered = sorted(
        result_rows, key=lambda row: row["mean_stim_r_response"], reverse=True
    )
    header = (
        f"{'variant':<22}{'test r':>9}{'frac ceil':>11}{'MSE':>10}"
        f"{'MSE calib':>11}{'val r':>9}{'params':>12}{'epoch':>9}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in ordered:
        print(
            f"{row['model']:<22}{row['mean_stim_r_response']:>9.4f}"
            f"{row['fraction_of_ceiling']:>11.3f}{row['test_mse']:>10.5f}"
            f"{row.get('calibrated_test_mse', float('nan')):>11.5f}"
            f"{row['validation_stim_r']:>9.4f}{row['trainable_parameters']:>12,}"
            f"{row['best_epoch']:>5d}/{row['epochs_run']:<3d}"
        )
    # end for scored variant
    print(f"\nsaved to {output_dir}")
# EOF


if __name__ == "__main__":
    main()
