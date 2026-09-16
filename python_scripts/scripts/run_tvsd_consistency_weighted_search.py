"""
Re-run the TVSD time-bin model selection under a consistency-weighted MSE.

The architecture search in ``run_tvsd_timebin_architecture_search.py`` trained
every decoder under a plain MSE over the 1600 (20 ms bin, IT site) cells of the
target, which spends as much gradient on a dead site as on one with a
split-half reliability of 0.95. This script keeps that protocol -- same
I-JEPA depths, same 80-180 ms window, same split, same scoring -- and changes
only the loss: each cell's error is weighted by the consistency of the
recording at that cell, and a softmax temperature sets how spiky that weighting
is. The infinite-temperature entry of the grid is the plain MSE, so every
weighted run has a matched anchor with the identical configuration and seed.

Four decoders are compared, each at the configuration that won the plain-MSE
search, and each against the same RidgeCV reference:
    - the noisy layer-attention baseline (NoisyBaselineModel);
    - a GRU that attends over the ANN depths with no time code anywhere, so the
      whole latency profile has to come out of the recurrence;
    - the tiny transformer;
    - the linear dynamical system.

Ridge is fitted per output cell, so no reweighting of the cells can change its
solution: it is the fixed reference the weighted decoders move against.

TVSD repeats only its 100 test images, so both the weights and the noise
ceiling are estimated there. The weights are a per-site property measured on
held-out images and carry no image-level information into training, but they
are not blind to the test recording -- every number here inherits that caveat.

The MSE noise ceiling of the dataset is computed two ways, from the repetition
scatter and from the reliability, and reported alongside the decoders: on an
average of 30 noisy repetitions even a perfect model pays a floor, and a raw
test MSE is uninterpretable without it.

Every finished run is written to its own file under ``runs/`` before the next
one starts, so an interrupted sweep resumes where it stopped.
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
from scipy.stats import wilcoxon  # noqa: E402


sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "python_scripts" / "src")
)

from IT_recap.neural_prediction_training import stimulus_correlation  # noqa: E402
from IT_recap.tvsd_consistency import (  # noqa: E402
    consistency_weights,
    describe_weights,
    effective_number_of_cells,
    make_weighted_mse_objective,
    mse_noise_ceiling,
    parse_temperature_grid,
    per_cell_test_mse,
    score_against_floor,
    temperature_label,
    weighted_mean,
)
from IT_recap.tvsd_experiments import (  # noqa: E402
    N_TEST_IMAGES,
    PROJECT_ROOT,
    evaluate_cached_feature_decoder,
    fit_ridge_map,
    load_project_paths,
    prepare_search_data,
    resolve_device,
    score_predictions,
    train_cached_feature_decoder,
)
from model_classes.timebin_models import build_timebin_model  # noqa: E402


# One fixed colour per compared method, kept identical to the plain-MSE search
# so the two figures can be read side by side. The attending GRU inherits the
# green that experiment gave the GRU family.
METHOD_COLORS = {
    "ridge": "#2a78d6",
    "baseline": "#eb6834",
    "gru_attention": "#1f9c6b",
    "tiny_transformer": "#6a4bbc",
    "lds": "#b23c8f",
}
METHOD_LABELS = {
    "ridge": "RidgeCV (linear)",
    "baseline": "NoisyBaselineModel",
    "gru_attention": "GRU + attention, no time code",
    "tiny_transformer": "Transformer only",
    "lds": "Linear dynamical system",
}
CEILING_COLOR = "#8a8a86"
FLOOR_COLOR = "#8a8a86"
GRID_COLOR = "#e3e3e0"

# Only the repetition mean is scored: it is the aggregation with a defined
# Spearman-Brown noise ceiling and a defined MSE floor.
REDUCER = "mean"

# The winning configuration of each architecture in the plain-MSE search
# (results/tvsd_timebin_architecture_search_ijepa_vith14_1k). Holding these
# fixed is what makes the temperature the only thing that varies here. The
# attending GRU never entered that search, so it takes the shared knobs of the
# GRU winner with widths matched to the other recurrent models.
ARCHITECTURE_CONFIGS = {
    "baseline": {
        "learning_rate": 1e-3,
        "weight_decay": 1e-2,
        "dropout": 0.5,
        "input_noise_std": 0.1,
        "temporal_noise_std": 0.1,
        "time_embedding_dim": 128,
        "value_dim": 256,
        "mlp_hidden_dim": 512,
    },
    "gru_attention": {
        "learning_rate": 3e-4,
        "weight_decay": 1e-2,
        "dropout": 0.1,
        "input_noise_std": 0.25,
        "temporal_noise_std": 0.5,
        "hidden_dim": 256,
        "attention_dim": 128,
    },
    "tiny_transformer": {
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "dropout": 0.1,
        "input_noise_std": 0.5,
        "temporal_noise_std": 0.5,
        "hidden_dim": 256,
        "n_attention_heads": 8,
        "n_transformer_layers": 2,
    },
    "lds": {
        "learning_rate": 1e-4,
        "weight_decay": 1e-1,
        "dropout": 0.1,
        "input_noise_std": 0.5,
        "temporal_noise_std": 0.1,
        "state_dim": 256,
    },
}

# Everything else in a configuration is a model constructor argument.
OPTIMIZER_HYPERPARAMETERS = ("learning_rate", "weight_decay")


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

    # The fitted response window and its output bin width, identical to the
    # plain-MSE architecture search.
    window_start_ms: float = 80.0
    window_end_ms: float = 180.0
    timebin_ms: float = 20.0

    # Three I-JEPA depths, the same selection as every other TVSD experiment.
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

    # The compared decoders and the spikiness of the loss they are trained
    # under. "uniform" is the plain-MSE anchor; the smaller the temperature,
    # the fewer cells the loss effectively fits.
    architectures: str = "baseline,gru_attention,tiny_transformer,lds"
    temperature_grid: str = "uniform,0.5,0.25,0.1,0.05,0.02"
    # Differences between architectures were ~0.01 in the plain-MSE search,
    # which a single run cannot resolve; every cell of the grid is repeated
    # under several initialization seeds.
    model_seeds: str = "0,1,2"

    # Optimization, unchanged from the plain-MSE search.
    batch_size: int = 256
    num_workers: int = 0
    epochs: int = 50
    minimum_epochs: int = 15
    patience: int = 12
    gradient_clip: float = 1.0
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2

    # Validation MSE ranks TVSD decoders by shrinkage rather than by image
    # selectivity, so checkpoints and early stopping follow stim_r. It is the
    # unweighted stim_r for every run, so the loss is the only difference.
    selection_metric: str = "stim_r"

    # Evaluation and bookkeeping.
    noise_ceiling_resamples: int = 40
    overwrite_finished_runs: bool = False
    device: str = "auto"
    smoke_test: bool = False


"""
parse_args
Parse command-line overrides into a configuration object.

OUTPUT:
    - cfg: Cfg -> data, window, loss, and optimization settings
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
            parser.add_argument(argument_name, type=type(default), default=default)
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
split_hyperparameters
Separate optimizer settings from model constructor arguments.

INPUT:
    - configuration: dict -> one architecture's hyperparameters

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
score_run
Score one set of test predictions under both the plain and the weighted view.

The unweighted numbers are what the plain-MSE search reported, so the two
experiments stay comparable; the weighted ones are what the training loss was
actually optimizing.

INPUT:
    - predictions: np.ndarray -> [presentations, time, neurons] test predictions
    - targets: np.ndarray -> [presentations, time, neurons] test targets
    - scoring: dict -> test image ids, fit-split mean, ceiling, response slice
    - weights: np.ndarray -> [time, neurons] consistency weights of this run
    - floors: dict -> MSE noise ceiling maps from mse_noise_ceiling

OUTPUT:
    - row: dict -> metrics under both views
    - cell_correlations: np.ndarray -> [time, neurons] test stim_r
"""
def score_run(predictions, targets, scoring, weights, floors):
    metrics, cell_correlations = score_predictions(
        predictions,
        targets,
        scoring["test_image_ids"],
        REDUCER,
        scoring["fit_target_mean"],
        scoring["ceiling"],
        scoring["response_slice"],
    )
    mse_map = per_cell_test_mse(
        predictions, targets, scoring["test_image_ids"], N_TEST_IMAGES
    )
    row = {
        **metrics,
        **score_against_floor(mse_map, floors, weights),
        # The image selectivity the loss was weighting, i.e. stim_r read with
        # the same weights the gradient used.
        "weighted_stim_r": round(
            weighted_mean(cell_correlations, weights), 4
        ),
        "weighted_fraction_of_ceiling": round(
            weighted_mean(cell_correlations, weights)
            / weighted_mean(scoring["ceiling"], weights),
            3,
        ),
    }
    return row, cell_correlations
# EOF


"""
run_one_setting
Train and score one architecture at one temperature under one seed.

INPUT:
    - architecture: str -> key in ARCHITECTURE_CONFIGS
    - temperature: float -> softmax scale of the loss weights
    - weights: np.ndarray -> [time, neurons] weights derived from it
    - seed: int -> model-initialization seed
    - shapes: dict -> n_layers, feature_dim, n_timepoints, n_neurons
    - loaders: dict -> train, validation, and test loaders
    - scoring: dict -> test image ids, fit-split mean, ceiling, response slice
    - floors: dict -> MSE noise ceiling maps
    - cfg: Cfg -> optimization settings
    - device: torch.device -> compute device

OUTPUT:
    - row: dict -> configuration, optimization, and test metrics
    - cell_correlations: np.ndarray -> [time, neurons] test stim_r
    - history: list[dict] -> per-epoch optimization record
"""
def run_one_setting(
    architecture,
    temperature,
    weights,
    seed,
    shapes,
    loaders,
    scoring,
    floors,
    cfg,
    device,
):
    configuration = ARCHITECTURE_CONFIGS[architecture]
    optimizer_kwargs, model_kwargs = split_hyperparameters(configuration)

    # Only the initialization stream depends on the seed; the loader order is
    # driven by cfg.random_seed, so seeds differ by their random weights alone.
    torch.manual_seed(seed)
    model = build_timebin_model(architecture, **shapes, **model_kwargs).to(device)
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    # Uniform weights make this exactly torch.nn.MSELoss, so the anchor runs
    # go through the identical code path as the weighted ones.
    cost_function = make_weighted_mse_objective(weights, device)
    run_cfg = replace(cfg, **optimizer_kwargs)
    run_start = time.perf_counter()
    (
        history,
        best_epoch,
        best_validation_mse,
        best_validation_stim_r,
    ) = train_cached_feature_decoder(
        model, loaders, run_cfg, device, verbose=False, cost_function=cost_function
    )
    train_seconds = time.perf_counter() - run_start

    predictions, targets = evaluate_cached_feature_decoder(
        model, loaders["test"], device
    )
    metrics, cell_correlations = score_run(
        predictions, targets, scoring, weights, floors
    )
    row = {
        "architecture": architecture,
        "temperature": temperature_label(temperature),
        "seed": seed,
        "trainable_parameters": trainable_parameters,
        "train_seconds": round(train_seconds, 1),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "validation_mse": round(best_validation_mse, 5),
        "validation_loss": round(history[best_epoch - 1]["validation_loss"], 5),
        "validation_stim_r": round(best_validation_stim_r, 4),
        **describe_weights(weights, scoring["ceiling"]),
        **metrics,
        **{name: configuration[name] for name in sorted(configuration)},
    }
    if architecture == "lds":
        # The learned dynamics are only interpretable through their spectrum.
        row["spectral_radius"] = round(model.spectral_radius(), 4)
    # end if the run fitted a linear dynamical system

    del model
    release_device_memory(device)
    return row, cell_correlations, history
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
Store one finished run so a later sweep reuses it instead of repeating it.

INPUT:
    - run_dir: Path -> directory holding one file per finished run
    - run_id: str -> identifier of the run
    - row: dict -> the run's metrics
    - history: list[dict] -> per-epoch optimization record
    - cell_correlations: np.ndarray -> [time, channels] test stim_r

OUTPUT:
    - None: writes the run's record and its per-cell correlations
"""
def save_finished_run(run_dir, run_id, row, history, cell_correlations):
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / f"{run_id}_cell_stim_r.npy", cell_correlations)
    with open(run_dir / f"{run_id}.json", "w") as run_file:
        json.dump({"row": row, "history": history}, run_file, indent=2)
    # end with saved run record
# EOF


"""
load_finished_run
Read one stored run back, or report that it has not been run yet.

INPUT:
    - run_dir: Path -> directory holding one file per finished run
    - run_id: str -> identifier of the requested run

OUTPUT:
    - finished: tuple | None -> (row, history, cell_correlations), or None
"""
def load_finished_run(run_dir, run_id):
    record_path = run_dir / f"{run_id}.json"
    correlation_path = run_dir / f"{run_id}_cell_stim_r.npy"
    if not (record_path.is_file() and correlation_path.is_file()):
        return None
    # end if this run has not finished before
    with open(record_path, "r") as run_file:
        record = json.load(run_file)
    # end with stored run record
    return record["row"], record["history"], np.load(correlation_path)
# EOF


"""
summarize_seeds
Collapse the seeds of one (architecture, temperature) cell into one record.

INPUT:
    - rows: list[dict] -> runs sharing an architecture and a temperature

OUTPUT:
    - summary: dict -> seed means, standard deviations, and the run count
"""
def summarize_seeds(rows):
    summary = {
        "architecture": rows[0]["architecture"],
        "temperature": rows[0]["temperature"],
        "n_effective_cells": rows[0]["n_effective_cells"],
        "effective_cell_fraction": rows[0]["effective_cell_fraction"],
        "n_seeds": len(rows),
    }
    averaged_fields = (
        "validation_stim_r",
        "mean_stim_r_response",
        "weighted_stim_r",
        "fraction_of_ceiling",
        "weighted_fraction_of_ceiling",
        "test_mse",
        "weighted_test_mse",
        "mse_above_floor",
        "reducible_mse_explained",
        "weighted_reducible_mse_explained",
        "best_epoch",
    )
    for name in averaged_fields:
        values = np.array([row[name] for row in rows], dtype=float)
        summary[f"{name}_mean"] = round(float(values.mean()), 5)
        summary[f"{name}_std"] = round(float(values.std(ddof=0)), 5)
    # end for averaged field
    return summary
# EOF


"""
compare_cells
Compare two methods cell by cell with a paired signed-rank test.

INPUT:
    - method_r: np.ndarray -> [time, channels] stim_r of the method
    - reference_r: np.ndarray -> [time, channels] stim_r of the reference
    - weights: np.ndarray -> [time, channels] weights the loss used

OUTPUT:
    - comparison: dict -> paired gain, win rate, and test statistics
"""
def compare_cells(method_r, reference_r, weights):
    valid = np.isfinite(method_r) & np.isfinite(reference_r)
    differences = method_r[valid] - reference_r[valid]
    statistic, p_value = wilcoxon(differences)
    return {
        "n_cells_compared": int(valid.sum()),
        "mean_stim_r_gain": round(float(np.mean(differences)), 4),
        "median_stim_r_gain": round(float(np.median(differences)), 4),
        # The same gain read with the loss's own weights: a weighted loss is
        # allowed to lose on cells it deliberately stopped fitting.
        "weighted_stim_r_gain": round(
            weighted_mean(np.where(valid, method_r - reference_r, np.nan), weights),
            4,
        ),
        "cells_won": int((differences > 0).sum()),
        "win_rate": round(float((differences > 0).mean()), 3),
        "wilcoxon_p_value": float(p_value),
    }
# EOF


"""
reliability_binned_gain
Mean stim_r gain over the anchor, in bins of cell reliability.

This is where a consistency-weighted loss is supposed to show up: if it works
by trading unreliable cells for reliable ones, the gain rises with reliability
and the loss on the low-reliability cells is the price.

INPUT:
    - gain: np.ndarray -> [time, channels] stim_r difference from the anchor
    - reliability: np.ndarray -> [time, channels] cell reliability
    - bin_edges: np.ndarray -> reliability bin edges

OUTPUT:
    - centers: np.ndarray -> bin centers
    - mean_gain: np.ndarray -> mean gain per bin, NaN where a bin is empty
    - counts: np.ndarray -> cells per bin
"""
def reliability_binned_gain(gain, reliability, bin_edges):
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    mean_gain, counts = [], []
    for low, high in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (reliability >= low) & (reliability < high) & np.isfinite(gain)
        counts.append(int(in_bin.sum()))
        mean_gain.append(float(gain[in_bin].mean()) if in_bin.any() else np.nan)
    # end for reliability bin
    return centers, np.array(mean_gain), np.array(counts)
# EOF


"""
fit_ridge_reference_row
Fit the RidgeCV reference and score it through the shared scoring path.

Ridge solves one independent regression per output cell, so the consistency
weights cannot change its fit. It is scored under every temperature's weights
all the same, because the *report* of a weighted metric still depends on them.

INPUT:
    - cfg: Cfg -> shared protocol settings
    - data: dict -> prepared split, features, and targets
    - scoring: dict -> test image ids, fit-split mean, ceiling, response slice
    - weights: dict -> weight map per temperature label
    - floors: dict -> MSE noise ceiling maps

OUTPUT:
    - ridge_row: dict -> the reference record, in the shape of a run row
    - cell_correlations: np.ndarray -> [time, channels] test stim_r
"""
def fit_ridge_reference_row(cfg, data, scoring, weights, floors):
    print("fitting RidgeCV reference")
    subset_targets, indices = data["subset_targets"], data["indices"]
    ridge_fit = fit_ridge_map(
        cfg,
        data["train_features"],
        data["test_features"],
        data["allmat"],
        indices,
        subset_targets["train"],
        subset_targets["validation"],
    )
    uniform_weights = np.ones_like(scoring["ceiling"])
    metrics, cell_correlations = score_run(
        ridge_fit["trial_predictions"],
        subset_targets["test"],
        scoring,
        uniform_weights,
        floors,
    )
    ridge_row = {
        "architecture": "ridge",
        "temperature": "n/a",
        "seed": -1,
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
    # The weighted view of the same fixed fit, one entry per temperature.
    ridge_row["weighted_stim_r_by_temperature"] = {
        label: round(weighted_mean(cell_correlations, weight_map), 4)
        for label, weight_map in weights.items()
    }
    return ridge_row, cell_correlations
# EOF


"""
plot_sweep_summary
Draw the four panels that carry the result of the temperature sweep.

INPUT:
    - cfg: Cfg -> window and model identity used in the titles
    - summaries: list[dict] -> one record per (architecture, temperature)
    - ridge_row: dict -> the ridge reference record
    - weights: dict -> weight map per temperature label
    - reliability: np.ndarray -> [time, channels] cell reliability
    - floors: dict -> MSE noise ceiling maps
    - anchor_gains: dict -> architecture to [time, channels] gain over uniform
    - output_path: Path -> destination figure file

OUTPUT:
    - None: writes the figure to disk
"""
def plot_sweep_summary(
    cfg,
    summaries,
    ridge_row,
    weights,
    reliability,
    floors,
    anchor_gains,
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

    # The sweep axis is the effective number of cells the loss fits, not the
    # temperature itself: the temperature is only meaningful against the spread
    # of the reliability map, the effective count is directly readable.
    temperature_labels = list(weights)
    positions = np.arange(len(temperature_labels))
    # Each tick carries the temperature and the effective cell count it means.
    tick_labels = [
        f"{label}\nN_eff {effective_number_of_cells(weights[label]):.0f}"
        for label in temperature_labels
    ]
    architectures = sorted({summary["architecture"] for summary in summaries})

    # --- panel 1: unweighted test stim_r against the plain-MSE anchor ---
    for architecture in architectures:
        ordered = [
            next(
                summary
                for summary in summaries
                if summary["architecture"] == architecture
                and summary["temperature"] == label
            )
            for label in temperature_labels
        ]
        axes[0, 0].errorbar(
            positions,
            [summary["mean_stim_r_response_mean"] for summary in ordered],
            yerr=[summary["mean_stim_r_response_std"] for summary in ordered],
            linewidth=2,
            marker="o",
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=0.8,
            capsize=4,
            color=METHOD_COLORS[architecture],
            label=METHOD_LABELS[architecture],
        )
        axes[0, 1].errorbar(
            positions,
            [summary["weighted_stim_r_mean"] for summary in ordered],
            yerr=[summary["weighted_stim_r_std"] for summary in ordered],
            linewidth=2,
            marker="o",
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=0.8,
            capsize=4,
            color=METHOD_COLORS[architecture],
            label=METHOD_LABELS[architecture],
        )
    # end for architecture
    axes[0, 0].axhline(
        ridge_row["mean_stim_r_response"],
        linewidth=2,
        color=METHOD_COLORS["ridge"],
        label=METHOD_LABELS["ridge"],
    )
    axes[0, 0].set_xticks(positions)
    axes[0, 0].set_xticklabels(tick_labels, fontsize=8)
    axes[0, 0].set_xlabel("loss weighting (softmax temperature)")
    axes[0, 0].set_ylabel("test stim_r, unweighted mean over cells")
    axes[0, 0].set_title(
        "Unweighted test stim_r: what the plain-MSE search reported"
    )
    axes[0, 0].legend(frameon=False, fontsize=8, loc="best")

    # --- panel 2: the weighted view, which is what the loss optimized ---
    for label, position in zip(temperature_labels, positions):
        axes[0, 1].plot(
            [position],
            [ridge_row["weighted_stim_r_by_temperature"][label]],
            marker="_",
            markersize=18,
            markeredgewidth=2.5,
            color=METHOD_COLORS["ridge"],
        )
    # end for temperature
    axes[0, 1].set_xticks(positions)
    axes[0, 1].set_xticklabels(tick_labels, fontsize=8)
    axes[0, 1].set_xlabel("loss weighting (softmax temperature)")
    axes[0, 1].set_ylabel("test stim_r, weighted by cell consistency")
    axes[0, 1].set_title("Weighted test stim_r (ridge dashes: same fixed fit)")

    # --- panel 3: what the weighting actually does to the loss ---
    sorted_reliability = np.sort(reliability.ravel())[::-1]
    cell_positions = np.arange(sorted_reliability.size)
    reliability_axis = axes[1, 0]
    reliability_axis.plot(
        cell_positions,
        sorted_reliability,
        linewidth=2,
        color=CEILING_COLOR,
        label="cell reliability (right axis)",
    )
    reliability_axis.set_ylabel("split-half reliability", color=CEILING_COLOR)
    reliability_axis.set_xlabel("(bin, site) cells, sorted by reliability")
    weight_axis = reliability_axis.twinx()
    weight_axis.grid(False)
    order = np.argsort(reliability.ravel())[::-1]
    # A blue-to-purple ramp over the grid, darkest where the loss is spikiest.
    ramp = plt.get_cmap("viridis")(np.linspace(0.15, 0.9, len(temperature_labels)))
    for label, color in zip(temperature_labels, ramp):
        weight_axis.plot(
            cell_positions,
            weights[label].ravel()[order] / weights[label].mean(),
            linewidth=1.8,
            color=color,
            label=label,
        )
    # end for temperature
    weight_axis.set_ylabel("loss weight, relative to the mean")
    weight_axis.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
    reliability_axis.set_title(
        "The weighting: reliability profile and the weight it induces"
    )

    # --- panel 4: where the weighted loss wins and where it pays for it ---
    bin_edges = np.linspace(0.0, 1.0, 11)
    for architecture, gain in sorted(anchor_gains.items()):
        centers, mean_gain, counts = reliability_binned_gain(
            gain, reliability, bin_edges
        )
        axes[1, 1].plot(
            centers[counts > 0],
            mean_gain[counts > 0],
            linewidth=2,
            marker="o",
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=0.8,
            color=METHOD_COLORS[architecture],
            label=METHOD_LABELS[architecture],
        )
    # end for architecture
    axes[1, 1].axhline(0.0, linewidth=2, color="#3d3d3a")
    axes[1, 1].set_xlabel("cell reliability")
    axes[1, 1].set_ylabel("stim_r gain over the plain-MSE anchor")
    axes[1, 1].set_title(
        "Best weighted temperature minus the plain-MSE anchor, by cell reliability"
    )
    axes[1, 1].legend(frameon=False, fontsize=8, loc="best")

    figure.suptitle(
        f"TVSD monkey F {cfg.area} from {cfg.model_name}: consistency-weighted "
        f"MSE, {cfg.timebin_ms:g} ms bins {cfg.window_start_ms:g}-"
        f"{cfg.window_end_ms:g} ms | MSE floor "
        f"{floors['mean_repetition_floor']:.4f} of "
        f"{floors['mean_null_mse']:.4f}"
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


def main():
    cfg = parse_args()
    if cfg.smoke_test:
        cfg.epochs = min(cfg.epochs, 3)
        cfg.minimum_epochs = 1
        cfg.patience = 1
        cfg.temperature_grid = "uniform,0.1"
        cfg.model_seeds = "0"
        cfg.noise_ceiling_resamples = 4
    # end if a smoke run was requested

    paths = load_project_paths(cfg)
    device = resolve_device(cfg.device)
    output_dir = Path(
        cfg.output_dir
        or PROJECT_ROOT
        / "results"
        / f"tvsd_consistency_weighted_{cfg.model_name}"
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / "runs"

    architectures = [
        name.strip() for name in cfg.architectures.split(",") if name.strip()
    ]
    unknown = sorted(set(architectures) - set(ARCHITECTURE_CONFIGS))
    if unknown:
        raise KeyError(f"No configuration defined for {unknown}.")
    # end if an architecture name is invalid
    temperatures = parse_temperature_grid(cfg.temperature_grid)
    seeds = [int(seed) for seed in cfg.model_seeds.split(",") if seed.strip()]

    data = prepare_search_data(cfg, paths, device)
    loaders, shapes, scoring = data["loaders"], data["shapes"], data["scoring"]

    # --- the weights and the MSE noise ceiling of the dataset ---
    reliability = scoring["ceiling"]
    floors = mse_noise_ceiling(
        data["subset_targets"]["test"],
        scoring["test_image_ids"],
        N_TEST_IMAGES,
        reliability,
        scoring["fit_target_mean"],
    )
    print(
        f"\nMSE noise ceiling on {N_TEST_IMAGES} test images averaging "
        f"{floors['mean_repetitions']:.0f} repetitions (standardized units)"
    )
    print(
        f"  floor from the repetition scatter {floors['mean_repetition_floor']:.5f}"
        f" | from the reliability {floors['mean_reliability_floor']:.5f}"
    )
    print(
        f"  a model predicting the fit-split mean pays "
        f"{floors['mean_null_mse']:.5f}, so the reducible range is "
        f"{floors['mean_null_mse'] - floors['mean_repetition_floor']:.5f}"
    )
    print(f"  mean stim_r ceiling {float(np.nanmean(reliability)):.4f}")

    weights = {
        temperature_label(temperature): consistency_weights(
            reliability, temperature
        )
        for temperature in temperatures
    }
    print("\nLoss weighting at each temperature")
    for label, weight_map in weights.items():
        description = describe_weights(weight_map, reliability)
        weighted_floor = weighted_mean(floors["repetition_floor"], weight_map)
        print(
            f"  {label:<8} effective cells "
            f"{description['n_effective_cells']:>7.1f} "
            f"({100 * description['effective_cell_fraction']:.1f}%) | "
            f"max/mean weight {description['weight_max_over_mean']:>6.2f} | "
            f"weighted reliability {description['weighted_reliability']:.4f} | "
            f"weighted MSE floor {weighted_floor:.5f}"
        )
    # end for temperature

    # --- ridge reference on the identical split and scoring path ---
    finished_ridge = (
        None if cfg.overwrite_finished_runs else load_finished_run(run_dir, "ridge")
    )
    if finished_ridge is not None:
        ridge_row, _, ridge_cell_r = finished_ridge
        print("\nreusing the stored RidgeCV reference")
    else:
        print()
        ridge_row, ridge_cell_r = fit_ridge_reference_row(
            cfg, data, scoring, weights, floors
        )
        save_finished_run(run_dir, "ridge", ridge_row, [], ridge_cell_r)
    # end if the ridge reference was reused or fitted
    print(
        f"  ridge test stim_r {ridge_row['mean_stim_r_response']:.4f} | test MSE "
        f"{ridge_row['test_mse']:.5f} | {ridge_row['reducible_mse_explained']:.3f}"
        " of the reducible MSE | alpha "
        f"{ridge_row.get('ridge_alpha', float('nan')):.0f}"
    )

    # Only ridge needs the stimulus-ordered feature arrays.
    for array_name in ("train_features", "test_features"):
        del data[array_name]
    # end for stimulus-ordered feature array
    gc.collect()

    # --- the sweep itself ---
    rows, cell_correlations = [], {}
    for architecture in architectures:
        print(
            f"\n{architecture}: {len(temperatures)} temperatures x "
            f"{len(seeds)} seeds"
        )
        for temperature in temperatures:
            label = temperature_label(temperature)
            for seed in seeds:
                run_id = f"{architecture}_{label}_seed{seed}"
                finished = (
                    None
                    if cfg.overwrite_finished_runs
                    else load_finished_run(run_dir, run_id)
                )
                if finished is not None:
                    row, history, cell_r = finished
                else:
                    row, cell_r, history = run_one_setting(
                        architecture,
                        temperature,
                        weights[label],
                        seed,
                        shapes,
                        loaders,
                        scoring,
                        floors,
                        cfg,
                        device,
                    )
                    save_finished_run(run_dir, run_id, row, history, cell_r)
                # end if this run was reused or trained
                rows.append(row)
                cell_correlations[(architecture, label, seed)] = cell_r
                print(
                    f"  {label:<8} seed {seed} | test r "
                    f"{row['mean_stim_r_response']:.4f} | weighted r "
                    f"{row['weighted_stim_r']:.4f} | MSE {row['test_mse']:.5f} "
                    f"| above floor {row['mse_above_floor']:.5f} | epoch "
                    f"{row['best_epoch']:02d}/{row['epochs_run']:02d} | "
                    f"{row['train_seconds']:.0f}s"
                    + ("" if finished is None else " | reused")
                )
                save_rows(output_dir / "sweep_results.csv", [ridge_row, *rows])
            # end for seed
        # end for temperature
    # end for architecture

    # --- aggregate over seeds and pick the best temperature per architecture ---
    summaries = []
    for architecture in architectures:
        for temperature in temperatures:
            label = temperature_label(temperature)
            cell_rows = [
                row
                for row in rows
                if row["architecture"] == architecture
                and row["temperature"] == label
            ]
            summaries.append(summarize_seeds(cell_rows))
        # end for temperature
    # end for architecture

    # Seed-averaged per-cell stim_r is the map every comparison below is run on.
    mean_cell_r = {}
    for architecture in architectures:
        for temperature in temperatures:
            label = temperature_label(temperature)
            mean_cell_r[(architecture, label)] = np.nanmean(
                [
                    cell_correlations[(architecture, label, seed)]
                    for seed in seeds
                ],
                axis=0,
            )
        # end for temperature
    # end for architecture

    best_temperature, comparisons, anchor_gains = {}, {}, {}
    anchor_label = temperature_label(temperatures[0])
    for architecture in architectures:
        architecture_summaries = [
            summary for summary in summaries if summary["architecture"] == architecture
        ]
        # Selection is on the unweighted test stim_r so that the temperatures
        # are ranked on one fixed scale rather than each on its own weighting.
        best = max(
            architecture_summaries,
            key=lambda summary: summary["mean_stim_r_response_mean"],
        )
        best_temperature[architecture] = best["temperature"]
        best_map = mean_cell_r[(architecture, best["temperature"])]
        anchor_map = mean_cell_r[(architecture, anchor_label)]

        # The best *weighted* setting is tracked separately from the best
        # setting overall: when the plain-MSE anchor wins, the question is
        # still what the weighting did, and a gain of exactly zero would only
        # be the anchor compared with itself.
        weighted_summaries = [
            summary
            for summary in architecture_summaries
            if summary["temperature"] != anchor_label
        ]
        comparison = {
            "best_temperature": best["temperature"],
            "anchor_is_best": best["temperature"] == anchor_label,
            "vs_ridge": compare_cells(
                best_map, ridge_cell_r, weights[best["temperature"]]
            ),
        }
        if weighted_summaries:
            best_weighted = max(
                weighted_summaries,
                key=lambda summary: summary["mean_stim_r_response_mean"],
            )
            weighted_label = best_weighted["temperature"]
            weighted_map = mean_cell_r[(architecture, weighted_label)]
            anchor_gains[architecture] = weighted_map - anchor_map
            comparison["best_weighted_temperature"] = weighted_label
            comparison["best_weighted_vs_anchor"] = compare_cells(
                weighted_map, anchor_map, weights[weighted_label]
            )
        # end if the grid holds a weighted setting at all
        comparisons[architecture] = comparison
    # end for architecture

    with open(output_dir / "config.json", "w") as config_file:
        json.dump(
            {
                **asdict(cfg),
                "covered_window_ms": list(data["covered_ms"]),
                "bin_edges_ms": data["bin_edges_ms"].tolist(),
                "architecture_configs": ARCHITECTURE_CONFIGS,
                "temperatures": [
                    temperature_label(temperature) for temperature in temperatures
                ],
                "seeds": seeds,
            },
            config_file,
            indent=2,
        )
    # end with saved configuration
    with open(output_dir / "sweep_results.json", "w") as results_file:
        json.dump(
            {
                "ridge": ridge_row,
                "runs": rows,
                "summaries": summaries,
                "comparisons": comparisons,
                "mse_noise_ceiling": {
                    "mean_repetitions": floors["mean_repetitions"],
                    "mean_repetition_floor": floors["mean_repetition_floor"],
                    "mean_reliability_floor": floors["mean_reliability_floor"],
                    "mean_null_mse": floors["mean_null_mse"],
                    "mean_stim_r_ceiling": float(np.nanmean(reliability)),
                    "weighted_repetition_floor": {
                        label: round(
                            weighted_mean(floors["repetition_floor"], weight_map), 5
                        )
                        for label, weight_map in weights.items()
                    },
                },
            },
            results_file,
            indent=2,
        )
    # end with saved sweep results
    np.savez_compressed(
        output_dir / "cell_stim_r.npz",
        reliability=reliability,
        repetition_floor=floors["repetition_floor"],
        reliability_floor=floors["reliability_floor"],
        null_mse=floors["null_mse"],
        bin_edges_ms=data["bin_edges_ms"],
        ridge=ridge_cell_r,
        **{
            f"{architecture}_{label}": mean_cell_r[(architecture, label)]
            for architecture, label in mean_cell_r
        },
        **{f"weights_{label}": weight_map for label, weight_map in weights.items()},
    )
    plot_sweep_summary(
        cfg,
        summaries,
        ridge_row,
        weights,
        reliability,
        floors,
        anchor_gains,
        output_dir / "consistency_weighted_search.png",
    )

    # The printed tables double as the accessible alternative to the figure.
    header = (
        f"{'architecture':<30}{'temp':>9}{'N_eff':>8}{'test r':>9}{'+-':>7}"
        f"{'weighted r':>12}{'MSE':>9}{'above floor':>13}{'frac ceil':>11}"
    )
    print("\nEvery (architecture, temperature) cell, seed mean")
    print(header)
    print("-" * len(header))
    for summary in summaries:
        print(
            f"{METHOD_LABELS[summary['architecture']]:<30}"
            f"{summary['temperature']:>9}"
            f"{summary['n_effective_cells']:>8.0f}"
            f"{summary['mean_stim_r_response_mean']:>9.4f}"
            f"{summary['mean_stim_r_response_std']:>7.4f}"
            f"{summary['weighted_stim_r_mean']:>12.4f}"
            f"{summary['test_mse_mean']:>9.5f}"
            f"{summary['mse_above_floor_mean']:>13.5f}"
            f"{summary['fraction_of_ceiling_mean']:>11.3f}"
        )
    # end for summary
    print(
        f"{METHOD_LABELS['ridge']:<30}{'n/a':>9}{reliability.size:>8d}"
        f"{ridge_row['mean_stim_r_response']:>9.4f}{0.0:>7.4f}"
        f"{ridge_row['weighted_stim_r']:>12.4f}{ridge_row['test_mse']:>9.5f}"
        f"{ridge_row['mse_above_floor']:>13.5f}"
        f"{ridge_row['fraction_of_ceiling']:>11.3f}"
    )

    print("\nBest temperature per architecture, paired over the target cells")
    for architecture in architectures:
        comparison = comparisons[architecture]
        against_ridge = comparison["vs_ridge"]
        print(
            f"  {METHOD_LABELS[architecture]:<30}"
            f"best {comparison['best_temperature']:<8} | vs ridge "
            f"{against_ridge['mean_stim_r_gain']:+.4f} "
            f"(win rate {against_ridge['win_rate']:.2f}, p "
            f"{against_ridge['wilcoxon_p_value']:.1e})"
        )
        against_anchor = comparison.get("best_weighted_vs_anchor")
        if against_anchor is not None:
            print(
                f"  {'':<30}best weighted setting "
                f"{comparison['best_weighted_temperature']} vs anchor "
                f"{against_anchor['mean_stim_r_gain']:+.4f} unweighted, "
                f"{against_anchor['weighted_stim_r_gain']:+.4f} weighted "
                f"(win rate {against_anchor['win_rate']:.2f}, p "
                f"{against_anchor['wilcoxon_p_value']:.1e})"
            )
        # end if the grid holds a weighted setting
        if comparison["anchor_is_best"]:
            print(
                f"  {'':<30}the plain-MSE anchor is the best setting: the "
                "weighting does not help this decoder"
            )
        # end if the anchor won outright
    # end for architecture
    print(f"\nsaved to {output_dir}")
# EOF


if __name__ == "__main__":
    main()
# EOC
