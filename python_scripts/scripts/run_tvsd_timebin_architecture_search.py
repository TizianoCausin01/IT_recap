"""
Search architectures and hyperparameters for the 20 ms TVSD IT response.

The target is the baseline-corrected monkey F IT MUA in five 20 ms bins between
80 and 180 ms, obtained by averaging pairs of the cached 10 ms bins. Frozen
I-JEPA depths are the only input, and four decoder families are compared under
one protocol: the notebook's layer-attention baseline, a tiny transformer, a
GRU, and a linear dynamical system. RidgeCV from the same concatenated depths
to the same five-bin target is the reference; it sees the identical split and
runs through the identical scoring code, so every difference is the mapping.

Every architecture is searched over the same shared knobs -- learning rate,
weight decay, dropout, input-noise scale, and temporal-noise scale -- plus its
own width parameters. Each architecture starts from one noiseless reference
configuration, so the effect of noise is always read against a matched anchor.

No run exceeds 50 epochs by default, and every run reports whether it was still
improving when it stopped: the slope of the training MSE and of the validation
stim_r over the final epochs, and whether the selected checkpoint was among the
last epochs of the budget.

Every finished run, the ridge reference included, is written to its own file
under ``runs/`` before the next one starts, and a rerun of the script loads
those instead of repeating them. An interrupted search therefore resumes where
it stopped, and a completed one re-emits its tables and figure for free. Only
metrics are kept, not weights: the script is seeded, so the winning decoder is
reproduced by training its reported configuration again.
"""

import argparse
import csv
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "python_scripts" / "src")
)

from IT_recap.neural_prediction_training import (  # noqa: E402
    split_half_reliability,
    stimulus_correlation,
)
from IT_recap.tvsd_experiments import (  # noqa: E402
    N_TEST_IMAGES,
    PROJECT_ROOT,
    average_targets_over_bin_groups,
    build_datasets,
    evaluate_cached_feature_decoder,
    fit_ridge_map,
    gather_presentation_features,
    load_cached_data,
    load_project_paths,
    prepare_search_data,
    resolve_device,
    score_predictions,
    train_cached_feature_decoder,
)
from model_classes.timebin_models import (  # noqa: E402
    TIMEBIN_MODEL_CLASSES,
    build_timebin_model,
)


# One fixed colour per compared method, validated for colourblind separation
# against the light chart surface. Ridge keeps the blue it has in the other
# TVSD comparison figures.
METHOD_COLORS = {
    "ridge": "#2a78d6",
    "baseline": "#eb6834",
    "tiny_transformer": "#6a4bbc",
    "gru": "#1f9c6b",
    "lds": "#b23c8f",
}
METHOD_LABELS = {
    "ridge": "RidgeCV (linear)",
    "baseline": "Layer attention (baseline)",
    "tiny_transformer": "Tiny transformer",
    "gru": "GRU",
    "lds": "Linear dynamical system",
}
CEILING_COLOR = "#8a8a86"
GRID_COLOR = "#e3e3e0"

# Only the repetition mean is scored: it is the aggregation with a defined
# Spearman-Brown noise ceiling.
REDUCER = "mean"

# Knobs every architecture shares. Batch size is held fixed so that the compute
# per epoch is comparable across runs and the loaders can be built once.
SHARED_SEARCH_SPACE = {
    "learning_rate": [1e-3, 3e-4, 1e-4],
    "weight_decay": [1e-4, 1e-2, 1e-1],
    "dropout": [0.1, 0.3, 0.5],
    # Input noise is in units of the normalized feature scale, temporal noise
    # in units of the current spread of the learned time code.
    "input_noise_std": [0.0, 0.1, 0.25, 0.5, 1.0],
    "temporal_noise_std": [0.0, 0.1, 0.25, 0.5],
}

ARCHITECTURE_SEARCH_SPACES = {
    "baseline": {
        "time_embedding_dim": [32, 64, 128],
        "value_dim": [128, 256],
        "mlp_hidden_dim": [128, 256, 512],
    },
    "tiny_transformer": {
        "hidden_dim": [128, 256],
        "n_attention_heads": [4, 8],
        "n_transformer_layers": [1, 2],
    },
    "gru": {
        "hidden_dim": [128, 256, 512],
        "time_embedding_dim": [16, 32, 64],
    },
    "lds": {
        "state_dim": [32, 64, 128, 256],
    },
}

# The noiseless anchor of every architecture: mid-sized, moderately
# regularized, and identical in its shared knobs across the four families.
ARCHITECTURE_REFERENCE_CONFIGS = {
    "baseline": {"time_embedding_dim": 64, "value_dim": 256, "mlp_hidden_dim": 256},
    "tiny_transformer": {
        "hidden_dim": 256,
        "n_attention_heads": 8,
        "n_transformer_layers": 1,
    },
    "gru": {"hidden_dim": 256, "time_embedding_dim": 32},
    "lds": {"state_dim": 128},
}
REFERENCE_SHARED_CONFIG = {
    "learning_rate": 3e-4,
    "weight_decay": 1e-2,
    "dropout": 0.3,
    "input_noise_std": 0.0,
    "temporal_noise_std": 0.0,
}

# Everything else in a sampled configuration is a model constructor argument.
OPTIMIZER_HYPERPARAMETERS = ("learning_rate", "weight_decay")

# Epochs used to judge whether a run had stopped improving.
TREND_WINDOW = 10


@dataclass
class Cfg:
    # Caches produced by train_tvsd_baseline_marimo.py; nothing is recomputed.
    env: str | None = None
    mua_file_name: str = "f_THINGS_MUA_trials.mat"
    feature_archive_name: str = "tvsd_monkeyF_ijepa_vith14_1k_224_features.npz"
    output_dir: str | None = None

    # Identity of the cached target file, not the fitted window.
    area: str = "IT"
    target_fs: int = 100
    time_start_ms: float = 0.0
    time_end_ms: float = 200.0

    # The searched response window and its output bin width. Edges snap to the
    # cached 10 ms bins, so the covered range is reported at run time.
    window_start_ms: float = 80.0
    window_end_ms: float = 180.0
    timebin_ms: float = 20.0

    # Three I-JEPA depths, the same selection as the window-decoder experiment.
    model_name: str = "ijepa_vith14_1k"
    layer_names: list[str] = field(
        default_factory=lambda: [
            "encoder.layer.4.output.dense",
            "encoder.layer.17.output.dense",
            "encoder.layer.27.output.dense",
        ]
    )

    # Split, identical to every other TVSD experiment in this repository.
    validation_fraction: float = 0.1
    random_seed: int = 0

    # Search extent. Each architecture gets its noiseless reference
    # configuration plus n_configs_per_architecture - 1 random draws.
    # The four families this experiment compares, named explicitly so that a
    # decoder added to the registry later does not silently join the search.
    architectures: str = "baseline,tiny_transformer,gru,lds"
    n_configs_per_architecture: int = 10
    search_seed: int = 0

    # Optimization. The epoch budget is deliberately short; whether a run was
    # still improving at the budget is reported rather than worked around.
    batch_size: int = 256
    num_workers: int = 0
    epochs: int = 50
    minimum_epochs: int = 15
    patience: int = 12
    gradient_clip: float = 1.0

    # Fallback optimizer settings. Every searched configuration overrides these
    # per run; they only matter when a configuration omits them.
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2

    # Validation MSE ranks TVSD decoders by shrinkage rather than by image
    # selectivity, so checkpoints and early stopping follow stim_r.
    selection_metric: str = "stim_r"

    # Evaluation and bookkeeping. Finished runs are reloaded from disk unless
    # they are explicitly recomputed.
    noise_ceiling_resamples: int = 40
    overwrite_finished_runs: bool = False
    device: str = "auto"
    smoke_test: bool = False


"""
parse_args
Parse command-line overrides into a configuration object.

OUTPUT:
    - cfg: Cfg -> data, window, search, and optimization settings
"""
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    default_layer_names = Cfg.__dataclass_fields__["layer_names"].default_factory()
    for field_name, field_definition in Cfg.__dataclass_fields__.items():
        if field_name == "layer_names":
            # An archive may hold more depths than one experiment uses, so the
            # selection is a list rather than a single typed value.
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
        layer_name.strip()
        for layer_name in arguments["layer_names"].split(",")
        if layer_name.strip()
    ]
    if not arguments["layer_names"]:
        raise ValueError("layer_names must contain at least one ANN layer.")
    # end if no ANN layers were selected
    return Cfg(**arguments)
# EOF


"""
sample_configurations
Build the configurations searched for one architecture.

The first entry is the architecture's noiseless reference, so every noise
result has a matched anchor; the rest are independent random draws from the
shared and architecture-specific grids, deduplicated.

INPUT:
    - architecture: str -> key in TIMEBIN_MODEL_CLASSES
    - n_configs: int -> configurations requested, reference included
    - rng: np.random.Generator -> sampler for the random draws

OUTPUT:
    - configurations: list[dict] -> hyperparameter dictionaries
"""
def sample_configurations(architecture, n_configs, rng):
    if architecture not in ARCHITECTURE_SEARCH_SPACES:
        raise KeyError(f"No search space defined for {architecture!r}.")
    # end if the architecture is unknown
    if n_configs <= 0:
        raise ValueError("n_configs_per_architecture must be positive.")
    # end if nothing would be searched

    architecture_space = ARCHITECTURE_SEARCH_SPACES[architecture]
    reference = {
        **REFERENCE_SHARED_CONFIG,
        **ARCHITECTURE_REFERENCE_CONFIGS[architecture],
    }
    configurations = [reference]
    seen = {tuple(sorted(reference.items()))}

    # Rejection sampling keeps the draws distinct without enumerating the grid.
    attempts = 0
    max_attempts = 50 * n_configs
    while len(configurations) < n_configs and attempts < max_attempts:
        attempts += 1
        candidate = {
            name: values[int(rng.integers(len(values)))]
            for name, values in {
                **SHARED_SEARCH_SPACE,
                **architecture_space,
            }.items()
        }
        key = tuple(sorted(candidate.items()))
        if key in seen:
            continue
        # end if this draw repeats an earlier configuration
        seen.add(key)
        configurations.append(candidate)
    # end while the requested number of configurations is not reached
    return configurations
# EOF


"""
split_hyperparameters
Separate optimizer settings from model constructor arguments.

INPUT:
    - configuration: dict -> one sampled hyperparameter set

OUTPUT:
    - optimizer_kwargs: dict -> learning rate and weight decay
    - model_kwargs: dict -> architecture and noise arguments
"""
def split_hyperparameters(configuration):
    optimizer_kwargs = {
        name: configuration[name] for name in OPTIMIZER_HYPERPARAMETERS
    }
    model_kwargs = {
        name: value
        for name, value in configuration.items()
        if name not in OPTIMIZER_HYPERPARAMETERS
    }
    return optimizer_kwargs, model_kwargs
# EOF


"""
per_epoch_slope
Least-squares slope per epoch of the last entries of one history field.

A negative training-MSE slope means the loss was still falling when the budget
ran out; a positive validation stim_r slope means the fit was still improving.

INPUT:
    - history: list[dict] -> per-epoch records
    - field_name: str -> record key to fit
    - window: int -> number of trailing epochs used

OUTPUT:
    - slope: float -> change per epoch, NaN when fewer than two epochs are left
"""
def per_epoch_slope(history, field_name, window=TREND_WINDOW):
    values = np.array([entry[field_name] for entry in history[-window:]], dtype=float)
    if len(values) < 2:
        return float("nan")
    # end if the trend is undefined
    epochs = np.arange(len(values), dtype=float)
    return float(np.polyfit(epochs, values, 1)[0])
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
    epochs_run = len(history)
    train_slope = per_epoch_slope(history, "train_mse")
    validation_slope = per_epoch_slope(history, "validation_stim_r")
    final_train_mse = history[-1]["train_mse"]
    return {
        "epochs_run": epochs_run,
        "stopped_early": epochs_run < cfg.epochs,
        # Relative slope makes runs with different loss scales comparable.
        "train_mse_slope_per_epoch": round(train_slope, 8),
        "train_mse_percent_per_epoch": round(
            100.0 * train_slope / final_train_mse, 4
        ),
        "validation_stim_r_slope_per_epoch": round(validation_slope, 6),
        # The checkpoint landing in the final epochs of an exhausted budget is
        # the practical sign that more epochs would still have helped.
        "still_improving_at_budget": bool(
            epochs_run == cfg.epochs
            and best_epoch >= epochs_run - 2
            and validation_slope > 0.0
        ),
    }
# EOF


"""
run_one_configuration
Train and score one architecture-hyperparameter combination.

INPUT:
    - architecture: str -> key in TIMEBIN_MODEL_CLASSES
    - configuration: dict -> sampled hyperparameters
    - shapes: dict -> n_layers, feature_dim, n_timepoints, and n_neurons
    - loaders: dict -> train, validation, and test loaders
    - scoring: dict -> test image ids, fit-split mean, ceiling, response slice
    - cfg: Cfg -> optimization settings
    - device: torch.device -> compute device

OUTPUT:
    - row: dict -> hyperparameters, optimization trend, and test metrics
    - site_correlations: np.ndarray -> [time, channels] test stim_r
    - history: list[dict] -> per-epoch optimization record
"""
def run_one_configuration(
    architecture, configuration, shapes, loaders, scoring, cfg, device
):
    optimizer_kwargs, model_kwargs = split_hyperparameters(configuration)

    # Every configuration starts from the same initialization stream, so runs
    # differ by their hyperparameters rather than by their random weights.
    torch.manual_seed(cfg.random_seed)
    model = build_timebin_model(
        architecture, **shapes, **model_kwargs
    ).to(device)
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    # The trainer reads its optimizer settings off cfg, so the sampled values
    # are applied to a copy rather than to the shared configuration object.
    run_cfg = replace(cfg, **optimizer_kwargs)
    run_start = time.perf_counter()
    (
        history,
        best_epoch,
        best_validation_mse,
        best_validation_stim_r,
    ) = train_cached_feature_decoder(
        model, loaders, run_cfg, device, verbose=False
    )
    train_seconds = time.perf_counter() - run_start

    predictions, targets = evaluate_cached_feature_decoder(
        model, loaders["test"], device
    )
    metrics, site_correlations = score_predictions(
        predictions,
        targets,
        scoring["test_image_ids"],
        REDUCER,
        scoring["fit_target_mean"],
        scoring["ceiling"],
        scoring["response_slice"],
    )
    row = {
        "architecture": architecture,
        "trainable_parameters": trainable_parameters,
        "train_seconds": round(train_seconds, 1),
        "best_epoch": best_epoch,
        "validation_mse": round(best_validation_mse, 5),
        "validation_stim_r": round(best_validation_stim_r, 4),
        **describe_optimization(history, best_epoch, cfg),
        **metrics,
        **{name: configuration[name] for name in sorted(configuration)},
    }
    if architecture == "lds":
        # The learned dynamics are only interpretable through their spectrum.
        row["spectral_radius"] = round(model.spectral_radius(), 4)
    # end if the run fitted a linear dynamical system

    # Sixty-odd models are built in one search, so each one's device memory is
    # released before the next is allocated.
    del model
    release_device_memory(device)
    return row, site_correlations, history
# EOF


"""
release_device_memory
Return the finished run's cached accelerator memory to the driver.

INPUT:
    - device: torch.device -> compute device

OUTPUT:
    - None
"""
def release_device_memory(device):
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    # end if the device caches its allocations
# EOF


"""
save_finished_run
Store one finished run so a later search reuses it instead of repeating it.

INPUT:
    - run_dir: Path -> directory holding one file per finished run
    - row: dict -> the run's metrics, keyed by config_id
    - history: list[dict] -> per-epoch optimization record
    - site_correlations: np.ndarray -> [time, channels] test stim_r

OUTPUT:
    - None: writes the run's record and its per-site correlations
"""
def save_finished_run(run_dir, row, history, site_correlations):
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / f"{row['config_id']}_site_stim_r.npy", site_correlations)
    with open(run_dir / f"{row['config_id']}.json", "w") as run_file:
        json.dump({"row": row, "history": history}, run_file, indent=2)
    # end with saved run record
# EOF


"""
load_finished_run
Read one stored run back, or report that it has not been run yet.

INPUT:
    - run_dir: Path -> directory holding one file per finished run
    - config_id: str -> identifier of the requested run

OUTPUT:
    - finished: tuple | None -> (row, history, site_correlations), or None
"""
def load_finished_run(run_dir, config_id):
    record_path = run_dir / f"{config_id}.json"
    site_path = run_dir / f"{config_id}_site_stim_r.npy"
    if not (record_path.is_file() and site_path.is_file()):
        return None
    # end if this run has not finished before
    with open(record_path, "r") as run_file:
        record = json.load(run_file)
    # end with stored run record
    return record["row"], record["history"], np.load(site_path)
# EOF


"""
best_row_per_architecture
Select each architecture's best run by validation stim_r.

INPUT:
    - rows: list[dict] -> completed run records

OUTPUT:
    - best_rows: dict -> architecture to its best run record
"""
def best_row_per_architecture(rows):
    best_rows = {}
    for row in rows:
        architecture = row["architecture"]
        current_best = best_rows.get(architecture)
        if (
            current_best is None
            or row["validation_stim_r"] > current_best["validation_stim_r"]
        ):
            best_rows[architecture] = row
        # end if this run is the architecture's best so far
    # end for completed run
    return best_rows
# EOF


"""
noise_profile
Best validation stim_r reached at every value of one noise knob.

INPUT:
    - rows: list[dict] -> runs of a single architecture
    - noise_name: str -> "input_noise_std" or "temporal_noise_std"

OUTPUT:
    - noise_values: list[float] -> sorted noise scales that were searched
    - best_scores: list[float] -> best validation stim_r at each scale
"""
def noise_profile(rows, noise_name):
    scores_by_value = {}
    for row in rows:
        value = row[noise_name]
        scores_by_value[value] = max(
            scores_by_value.get(value, -np.inf), row["validation_stim_r"]
        )
    # end for run of this architecture
    noise_values = sorted(scores_by_value)
    return noise_values, [scores_by_value[value] for value in noise_values]
# EOF


"""
plot_search_summary
Draw the four panels that carry the search result.

INPUT:
    - cfg: Cfg -> window and model identity used in the titles
    - rows: list[dict] -> every completed decoder run
    - ridge_row: dict -> the ridge reference record
    - histories: dict -> per-epoch records of the best run per architecture
    - site_correlations: dict -> [time, channels] test stim_r per method
    - ceiling: np.ndarray -> [time, channels] noise ceiling
    - bin_edges_ms: np.ndarray -> edges of the output time bins
    - output_path: Path -> destination figure file

OUTPUT:
    - None: writes the figure to disk
"""
def plot_search_summary(
    cfg,
    rows,
    ridge_row,
    histories,
    site_correlations,
    ceiling,
    bin_edges_ms,
    output_path,
):
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    for axis in axes.flat:
        axis.grid(True, color=GRID_COLOR, linewidth=0.8)
        axis.set_axisbelow(True)
        for spine_name in ("top", "right"):
            axis.spines[spine_name].set_visible(False)
        # end for hidden spine
    # end for panel

    best_rows = best_row_per_architecture(rows)
    ranked = sorted(
        [ridge_row, *best_rows.values()],
        key=lambda row: row["mean_stim_r_response"],
    )

    # --- panel 1: the headline, best test stim_r of every method ---
    positions = np.arange(len(ranked))
    axes[0, 0].barh(
        positions,
        [row["mean_stim_r_response"] for row in ranked],
        height=0.55,
        color=[METHOD_COLORS[row["architecture"]] for row in ranked],
    )
    axes[0, 0].axvline(
        float(np.nanmean(ceiling)),
        linewidth=2,
        color=CEILING_COLOR,
        label="noise ceiling",
    )
    for position, row in zip(positions, ranked):
        # A direct label on each bar removes the need to read against the axis.
        axes[0, 0].text(
            row["mean_stim_r_response"] + 0.004,
            position,
            f"{row['mean_stim_r_response']:.3f}",
            va="center",
            fontsize=9,
            color="#3d3d3a",
        )
    # end for ranked method
    axes[0, 0].set_yticks(positions)
    axes[0, 0].set_yticklabels(
        [METHOD_LABELS[row["architecture"]] for row in ranked]
    )
    axes[0, 0].set_xlabel("test stim_r, mean over sites and 20 ms bins")
    axes[0, 0].set_xlim(0.0, float(np.nanmean(ceiling)) + 0.06)
    axes[0, 0].set_title("Best configuration of each method")
    axes[0, 0].legend(frameon=False, fontsize=9, loc="lower right")

    # --- panel 2: what the two noise knobs bought, per architecture ---
    for architecture in sorted(best_rows):
        architecture_rows = [
            row for row in rows if row["architecture"] == architecture
        ]
        for noise_name, line_style in (
            ("input_noise_std", "-"),
            ("temporal_noise_std", "--"),
        ):
            noise_values, best_scores = noise_profile(
                architecture_rows, noise_name
            )
            axes[0, 1].plot(
                noise_values,
                best_scores,
                line_style,
                linewidth=2,
                marker="o",
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=0.8,
                color=METHOD_COLORS[architecture],
            )
        # end for noise knob
    # end for architecture
    axes[0, 1].axhline(
        ridge_row["validation_stim_r"],
        linewidth=2,
        color=METHOD_COLORS["ridge"],
    )
    # Colour carries the architecture, line style the knob, so the legend is
    # split into the two encodings rather than into eight combinations.
    style_handles = [
        plt.Line2D([], [], color=METHOD_COLORS[name], linewidth=2)
        for name in sorted(best_rows)
    ] + [
        plt.Line2D([], [], color="#3d3d3a", linewidth=2, linestyle="-"),
        plt.Line2D([], [], color="#3d3d3a", linewidth=2, linestyle="--"),
        plt.Line2D([], [], color=METHOD_COLORS["ridge"], linewidth=2),
    ]
    axes[0, 1].legend(
        style_handles,
        [METHOD_LABELS[name] for name in sorted(best_rows)]
        + ["input noise", "temporal noise", "ridge"],
        frameon=False,
        fontsize=8,
        loc="best",
        ncol=2,
    )
    axes[0, 1].set_xlabel("noise scale searched")
    axes[0, 1].set_ylabel("best validation stim_r")
    axes[0, 1].set_title("Best result reached at each noise scale")

    # --- panel 3: optimization, the answer to "is the loss still falling" ---
    for architecture, history in sorted(histories.items()):
        axes[1, 0].plot(
            [entry["epoch"] for entry in history],
            [entry["validation_stim_r"] for entry in history],
            linewidth=2,
            color=METHOD_COLORS[architecture],
            label=METHOD_LABELS[architecture],
        )
    # end for best run per architecture
    axes[1, 0].axhline(
        ridge_row["validation_stim_r"],
        linewidth=2,
        color=METHOD_COLORS["ridge"],
        label=METHOD_LABELS["ridge"],
    )
    axes[1, 0].set_xlabel(f"epoch (budget {cfg.epochs})")
    axes[1, 0].set_ylabel("validation stim_r (single trials)")
    axes[1, 0].set_title("Optimization of the best run per architecture")
    axes[1, 0].legend(frameon=False, fontsize=9, loc="lower right", ncol=2)

    # --- panel 4: where in the response window each method wins ---
    bin_centers = (bin_edges_ms[:-1] + bin_edges_ms[1:]) / 2.0
    axes[1, 1].plot(
        bin_centers,
        np.nanmean(ceiling, axis=1),
        linewidth=2,
        marker="o",
        markersize=8,
        color=CEILING_COLOR,
        label="noise ceiling",
    )
    for method_name, correlations in sorted(site_correlations.items()):
        axes[1, 1].plot(
            bin_centers,
            np.nanmean(correlations, axis=1),
            linewidth=2,
            marker="o",
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=0.8,
            color=METHOD_COLORS[method_name],
            label=METHOD_LABELS[method_name],
        )
    # end for scored method
    axes[1, 1].set_xticks(bin_centers)
    axes[1, 1].set_xticklabels(
        [
            f"{start:g}-{end:g}"
            for start, end in zip(bin_edges_ms[:-1], bin_edges_ms[1:])
        ]
    )
    axes[1, 1].set_xlabel("time bin (ms after stimulus onset)")
    axes[1, 1].set_ylabel("mean test stim_r over sites")
    axes[1, 1].set_title("Prediction across the response window")
    axes[1, 1].legend(frameon=False, fontsize=8, loc="lower left", ncol=2)

    figure.suptitle(
        f"TVSD monkey F {cfg.area} from {cfg.model_name}: architecture and "
        f"hyperparameter search, {cfg.timebin_ms:g} ms bins "
        f"{cfg.window_start_ms:g}-{cfg.window_end_ms:g} ms"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
# EOF


"""
save_rows
Write a list of result dictionaries to CSV, filling absent keys.

INPUT:
    - path: Path -> destination CSV file
    - rows: list[dict] -> result records, possibly with different keys

OUTPUT:
    - None: writes the file
"""
def save_rows(path, rows):
    if not rows:
        return
    # end if there is nothing to write
    field_names = []
    for row in rows:
        for name in row:
            if name not in field_names:
                field_names.append(name)
            # end if this column is new
        # end for column of this row
    # end for result row
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)
    # end with result file
# EOF


"""
fit_ridge_reference_row
Fit the RidgeCV reference and score it through the shared scoring path.

INPUT:
    - cfg: Cfg -> shared protocol settings
    - train_features: np.ndarray -> ordered train-stimulus features
    - test_features: np.ndarray -> ordered repeated-test-stimulus features
    - allmat: np.ndarray -> ALLMAT metadata rows
    - indices: dict -> presentation indices per subset
    - subset_targets: dict -> standardized targets per subset
    - fit_target_mean: np.ndarray -> [time, channels] fit-split mean predictor
    - scoring: dict -> test image ids, ceiling, and response slice

OUTPUT:
    - ridge_row: dict -> the reference record, in the same shape as a run row
    - site_correlations: np.ndarray -> [time, channels] test stim_r
"""
def fit_ridge_reference_row(
    cfg,
    train_features,
    test_features,
    allmat,
    indices,
    subset_targets,
    fit_target_mean,
    scoring,
):
    print("fitting RidgeCV reference")
    ridge_fit = fit_ridge_map(
        cfg,
        train_features,
        test_features,
        allmat,
        indices,
        subset_targets["train"],
        subset_targets["validation"],
    )
    metrics, site_correlations = score_predictions(
        ridge_fit["trial_predictions"],
        subset_targets["test"],
        scoring["test_image_ids"],
        REDUCER,
        fit_target_mean,
        scoring["ceiling"],
        scoring["response_slice"],
    )
    ridge_row = {
        "architecture": "ridge",
        "config_id": "ridge",
        "trainable_parameters": ridge_fit["n_coefficients"],
        "validation_mse": round(ridge_fit["validation_mse"], 5),
        "validation_stim_r": round(
            float(
                np.nanmean(
                    stimulus_correlation(
                        ridge_fit["validation_predictions"],
                        subset_targets["validation"],
                    )
                )
            ),
            4,
        ),
        "ridge_alpha": ridge_fit["alpha"],
        **metrics,
    }
    return ridge_row, site_correlations
# EOF


def main():
    cfg = parse_args()
    if cfg.smoke_test:
        cfg.epochs = min(cfg.epochs, 3)
        cfg.minimum_epochs = 1
        cfg.patience = 1
        cfg.n_configs_per_architecture = 2
        cfg.noise_ceiling_resamples = 4
    # end if a smoke run was requested

    paths = load_project_paths(cfg)
    device = resolve_device(cfg.device)
    output_dir = Path(
        cfg.output_dir
        or PROJECT_ROOT
        / "results"
        / f"tvsd_timebin_architecture_search_{cfg.model_name}"
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    architectures = [
        name.strip() for name in cfg.architectures.split(",") if name.strip()
    ]
    unknown = sorted(set(architectures) - set(TIMEBIN_MODEL_CLASSES))
    if unknown:
        raise KeyError(f"Unknown architectures: {unknown}.")
    # end if an architecture name is invalid

    data = prepare_search_data(cfg, paths, device)
    loaders, shapes, scoring = data["loaders"], data["shapes"], data["scoring"]
    subset_targets, indices = data["subset_targets"], data["indices"]
    fit_target_mean = data["fit_target_mean"]
    bin_indices, covered_ms = data["bin_indices"], data["covered_ms"]
    bin_edges_ms = data["bin_edges_ms"]

    # --- ridge reference on the identical split and scoring path ---
    run_dir = output_dir / "runs"
    finished_ridge = (
        None
        if cfg.overwrite_finished_runs
        else load_finished_run(run_dir, "ridge")
    )
    if finished_ridge is not None:
        ridge_row, _, ridge_site_r = finished_ridge
        print("reusing the stored RidgeCV reference")
    else:
        ridge_row, ridge_site_r = fit_ridge_reference_row(
            cfg,
            data["train_features"],
            data["test_features"],
            data["allmat"],
            indices,
            subset_targets,
            fit_target_mean,
            scoring,
        )
        save_finished_run(run_dir, ridge_row, [], ridge_site_r)
    # end if the ridge reference was reused or fitted
    print(
        f"  ridge validation stim_r {ridge_row['validation_stim_r']:.4f} | "
        f"test stim_r {ridge_row['mean_stim_r_response']:.4f} | ceiling "
        f"{ridge_row['target_reliability']:.4f}"
    )

    # The full stimulus-ordered feature arrays are only needed by ridge; the
    # search itself reads the per-presentation subsets already materialized.
    for array_name in ("train_features", "test_features"):
        del data[array_name]
    # end for stimulus-ordered feature array
    gc.collect()

    # --- the search itself ---
    search_rng = np.random.default_rng(cfg.search_seed)
    rows, histories = [], {}
    best_site_correlations = {"ridge": ridge_site_r}
    for architecture in architectures:
        configurations = sample_configurations(
            architecture, cfg.n_configs_per_architecture, search_rng
        )
        print(
            f"\n{architecture}: {len(configurations)} configurations "
            f"({cfg.epochs} epochs max each)"
        )
        for config_index, configuration in enumerate(configurations):
            config_id = f"{architecture}_{config_index:02d}"
            finished = (
                None
                if cfg.overwrite_finished_runs
                else load_finished_run(run_dir, config_id)
            )
            if finished is not None:
                row, history, site_r = finished
            else:
                row, site_r, history = run_one_configuration(
                    architecture,
                    configuration,
                    shapes,
                    loaders,
                    scoring,
                    cfg,
                    device,
                )
                row = {"config_id": config_id, **row}
                save_finished_run(run_dir, row, history, site_r)
            # end if this configuration was reused or trained
            rows.append(row)
            if best_row_per_architecture(rows).get(architecture) is row:
                # Only the architecture's leading run is carried into the
                # figure; every run itself lives in its own file.
                histories[architecture] = history
                best_site_correlations[architecture] = site_r
            # end if this run leads its architecture
            print(
                f"  {row['config_id']:<22} val r {row['validation_stim_r']:.4f}"
                f" | test r {row['mean_stim_r_response']:.4f} | ceil frac "
                f"{row['fraction_of_ceiling']:.3f} | epoch "
                f"{row['best_epoch']:02d}/{row['epochs_run']:02d} | "
                f"{row['train_seconds']:.0f}s | "
                f"noise in {row['input_noise_std']:.2f} t "
                f"{row['temporal_noise_std']:.2f}"
                + ("" if finished is None else " | reused")
            )
            # Partial results survive an interrupted search.
            save_rows(output_dir / "search_results.csv", [ridge_row, *rows])
        # end for configuration
    # end for architecture

    best_rows = best_row_per_architecture(rows)
    with open(output_dir / "config.json", "w") as config_file:
        json.dump(
            {
                **asdict(cfg),
                "covered_window_ms": list(covered_ms),
                "window_bin_indices": bin_indices.tolist(),
                "bin_edges_ms": bin_edges_ms.tolist(),
                "shared_search_space": SHARED_SEARCH_SPACE,
                "architecture_search_spaces": ARCHITECTURE_SEARCH_SPACES,
            },
            config_file,
            indent=2,
        )
    # end with saved configuration
    with open(output_dir / "search_results.json", "w") as results_file:
        json.dump(
            {
                "ridge": ridge_row,
                "runs": rows,
                "best_per_architecture": best_rows,
                "histories": histories,
            },
            results_file,
            indent=2,
        )
    # end with saved search results
    np.savez_compressed(
        output_dir / "site_stim_r.npz",
        ceiling=scoring["ceiling"],
        bin_edges_ms=bin_edges_ms,
        **best_site_correlations,
    )
    plot_search_summary(
        cfg,
        rows,
        ridge_row,
        histories,
        best_site_correlations,
        scoring["ceiling"],
        bin_edges_ms,
        output_dir / "timebin_architecture_search.png",
    )

    # The printed tables double as the accessible alternative to the figure.
    header = (
        f"{'method':<28}{'val r':>8}{'test r':>9}{'median r':>10}"
        f"{'frac ceil':>11}{'test MSE':>10}{'params':>10}{'epoch':>8}"
    )
    print("\nBest configuration of each method, ranked by test stim_r")
    print(header)
    print("-" * len(header))
    ranked = sorted(
        [ridge_row, *best_rows.values()],
        key=lambda row: -row["mean_stim_r_response"],
    )
    for row in ranked:
        best_epoch = row.get("best_epoch") or 0
        print(
            f"{METHOD_LABELS[row['architecture']]:<28}"
            f"{row['validation_stim_r']:>8.4f}"
            f"{row['mean_stim_r_response']:>9.4f}"
            f"{row['median_stim_r_response']:>10.4f}"
            f"{row['fraction_of_ceiling']:>11.3f}"
            f"{row['test_mse']:>10.5f}"
            f"{row['trainable_parameters']:>10,}"
            f"{best_epoch:>8d}"
        )
    # end for ranked method

    print("\nWas the loss still decreasing at the epoch budget?")
    for architecture in architectures:
        row = best_rows[architecture]
        verdict = (
            "still improving"
            if row["still_improving_at_budget"]
            else ("converged" if row["stopped_early"] else "flat at budget")
        )
        print(
            f"  {METHOD_LABELS[architecture]:<28}{verdict:<17}"
            f"epochs {row['epochs_run']:02d}, best {row['best_epoch']:02d} | "
            f"train MSE {row['train_mse_percent_per_epoch']:+.3f}%/epoch | "
            f"validation stim_r "
            f"{row['validation_stim_r_slope_per_epoch']:+.5f}/epoch"
        )
    # end for architecture

    print("\nBest hyperparameters per architecture")
    for architecture in architectures:
        row = best_rows[architecture]
        searched_names = sorted(
            set(SHARED_SEARCH_SPACE) | set(ARCHITECTURE_SEARCH_SPACES[architecture])
        )
        settings = ", ".join(
            f"{name}={row[name]}" for name in searched_names
        )
        print(f"  {architecture}: {settings}")
    # end for architecture
    print(f"\nsaved to {output_dir}")
# EOF


if __name__ == "__main__":
    main()
# EOC
