"""
Give the time-bin decoders the first 50 ms of the response they must predict.

Every TVSD decoder so far maps an image to the 80-180 ms IT response and never
sees the recording. This experiment adds one input: the same presentation's own
0-50 ms activity, five cached 10 ms bins. IT latency is around 70 ms, so those
bins carry no stimulus drive; what they carry is the trial's own state. The
question is whether the response window is better explained by the image plus
that state than by the image alone.

Six methods are compared on identical data. A GRU and a linear dynamical system
both take the early trace as their initial condition, ridge takes it as extra
regressors, and each of the three has an image-only ablation that differs in
nothing else. The features are AlexNet conv5, average-pooled onto a coarse
grid, standardized on the fit split, and shared by all six; the two decoders
train under 0.4 dropout and Gaussian noise on those features, the regularizers
the earlier searches settled on.

Every finished method is written to its own file under ``runs/`` before the next
one starts, so an interrupted comparison resumes instead of repeating itself.
"""

import argparse
import csv
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
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
    disjoint_half_stimulus_correlation,
    stimulus_correlation,
)
from IT_recap.tvsd_experiments import (  # noqa: E402
    N_TEST_IMAGES,
    PROJECT_ROOT,
    average_targets_over_bin_groups,
    evaluate_cached_feature_decoder,
    fit_ridge_from_presentations,
    gather_presentation_features,
    load_cached_targets,
    load_pooled_spatial_features,
    load_project_paths,
    make_multi_input_loader,
    prepare_timebin_data,
    resolve_device,
    ridge_equivalent_l2_lambda,
    score_and_decompose,
    select_window_bin_indices,
    standardize_features_on_fit_split,
    standardize_targets,
    train_cached_feature_decoder,
)
from model_classes.early_response_models import (  # noqa: E402
    build_early_response_model,
)


# The six compared methods: three mappings, each with and without the early
# trace. The tuple is (architecture, whether the early response is observed);
# "ridge" is not a torch model and takes the linear-reference path instead.
METHOD_SPECS = {
    "gru_early": ("gru", True),
    "gru_image_only": ("gru", False),
    "lds_early": ("lds", True),
    "lds_image_only": ("lds", False),
    "ridge_early": ("ridge", True),
    "ridge_image_only": ("ridge", False),
}

# One fixed colour per mapping, kept from the architecture-search figures; the
# image-only ablation of each is drawn in the same hue, lightened.
METHOD_COLORS = {
    "gru_early": "#1f9c6b",
    "gru_image_only": "#8fd3ba",
    "lds_early": "#b23c8f",
    "lds_image_only": "#dfa4cb",
    "ridge_early": "#2a78d6",
    "ridge_image_only": "#a3c5ec",
}
METHOD_LABELS = {
    "gru_early": "GRU + early response",
    "gru_image_only": "GRU, image only",
    "lds_early": "LDS + early response",
    "lds_image_only": "LDS, image only",
    "ridge_early": "Ridge + early response",
    "ridge_image_only": "Ridge, image only",
}
CEILING_COLOR = "#8a8a86"
GRID_COLOR = "#e3e3e0"

# Knobs both decoders share in the hyperparameter search. The reference run of
# each method is always searched first, so every draw has a matched anchor.
SHARED_SEARCH_SPACE = {
    "learning_rate": [1e-3, 3e-4, 1e-4, 3e-5],
    "weight_decay": [1e-4, 1e-2, 1e-1, 3e-1],
    "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
    # Input noise is in units of the normalized feature scale; above 0.5 the
    # decoder collapses, so the grid stops there.
    "input_noise_std": [0.0, 0.1, 0.25, 0.5],
}

# Ridge maps all 1024 pooled features straight onto the 1600 outputs, while
# both decoders squeeze them through a single projection first. The widths
# reach well past the searched defaults so that bottleneck can be ruled out.
ARCHITECTURE_SEARCH_SPACES = {
    "gru": {
        "hidden_dim": [128, 256, 512, 1024],
        "time_embedding_dim": [16, 32, 64],
    },
    "lds": {"state_dim": [64, 128, 256, 512, 1024]},
}

# Only the repetition mean is scored: it is the aggregation with a defined
# Spearman-Brown noise ceiling.
REDUCER = "mean"

# Everything else in a configuration is a model constructor argument. The
# penalty settings join the optimizer ones: they act on the loss, not on the
# architecture, so the trainer reads them off cfg rather than the constructor.
OPTIMIZER_HYPERPARAMETERS = (
    "learning_rate", "weight_decay", "l2_lambda", "l2_scope",
)

# Epochs used to judge whether a run had stopped improving.
TREND_WINDOW = 10


@dataclass
class Cfg:
    # Caches produced by the marimo notebook and by
    # extract_tvsd_spatial_features.py; nothing is recomputed here.
    env: str | None = None
    mua_file_name: str = "f_THINGS_MUA_trials.mat"
    spatial_stem: str = "tvsd_monkeyF_alexnet_features_11_spatial"
    model_name: str = "alexnet_conv5"
    output_dir: str | None = None

    # Identity of the cached target file, not the fitted window.
    area: str = "IT"
    target_fs: int = 100
    time_start_ms: float = 0.0
    time_end_ms: float = 200.0

    # The predicted window and its output bin width, as in the architecture
    # search, so the numbers here sit next to that experiment's.
    window_start_ms: float = 80.0
    window_end_ms: float = 180.0
    timebin_ms: float = 20.0

    # The observed early window, at the cache's own 10 ms resolution: five bins
    # covering 0-50 ms, before the IT response begins.
    early_start_ms: float = 0.0
    early_end_ms: float = 50.0

    # AlexNet conv5 is 256 x 13 x 13. Average pooling onto a 2 x 2 grid keeps
    # coarse retinotopy and yields 1024 features, a width both the recurrent
    # decoders and RidgeCV handle comfortably.
    spatial_pool_size: int = 2

    # Split, identical to every other TVSD experiment in this repository.
    validation_fraction: float = 0.1
    random_seed: int = 0
    # Weight initialization is seeded separately from the split, so a
    # configuration can be repeated on identical data to tell a real gain from
    # a lucky initialization. Unset means "the same seed as the split".
    init_seed: int = -1

    # Methods and their widths. The decoder widths are the ones each family won
    # the 20 ms architecture search with.
    methods: str = ",".join(METHOD_SPECS)
    gru_hidden_dim: int = 256
    gru_time_embedding_dim: int = 32
    lds_state_dim: int = 128
    early_dim: int = 64

    # Regularization, shared by both decoders so the comparison is of mappings.
    dropout: float = 0.4
    input_noise_std: float = 0.25
    temporal_noise_std: float = 0.0

    # Optimization.
    batch_size: int = 256
    num_workers: int = 0
    epochs: int = 60
    minimum_epochs: int = 15
    patience: int = 12
    gradient_clip: float = 1.0
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2

    # Validation MSE ranks TVSD decoders by shrinkage rather than by image
    # selectivity, so checkpoints and early stopping follow stim_r.
    selection_metric: str = "stim_r"

    # An explicit L2 term on the decoder's ridge-analogous maps, which is the
    # objective ridge itself minimizes. AdamW's weight decay is decoupled and
    # never passes through Adam's scaling, so the two are not the same knob.
    l2_lambda: float = 0.0
    l2_scope: str = "readout"

    # Hyperparameter search. With --search the named methods are re-fitted over
    # random draws instead of once at the reference configuration; the winner
    # of each is then reported against the stored ridge references.
    search: bool = False
    search_methods: str = "gru_image_only,lds_image_only"
    n_configs_per_method: int = 12
    search_seed: int = 0

    # With --l2_sweep each method starts from its best searched configuration
    # and is re-fitted across this grid of penalty weights and scopes, with the
    # decoupled weight decay both kept and switched off.
    l2_sweep: bool = False
    l2_lambdas: str = "1e-4,1e-3,1e-2"
    l2_scopes: str = "readout,all"

    # Evaluation and bookkeeping.
    noise_ceiling_resamples: int = 40
    overwrite_finished_runs: bool = False
    device: str = "auto"
    smoke_test: bool = False


"""
parse_args
Parse command-line overrides into a configuration object.

OUTPUT:
    - cfg: Cfg -> data, window, method, and optimization settings
"""
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for field_name, field_definition in Cfg.__dataclass_fields__.items():
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
    return Cfg(**vars(parser.parse_args()))
# EOF


"""
extract_early_response
Read the observed early bins of every presentation off the target cache.

The early trace is left at the cache's own 10 ms resolution -- a group size of
one -- because its point is the fine detail of the pre-response state, and it
is standardized with the target's own train-only channel statistics, so decoder
input and decoder output live on one scale.

INPUT:
    - cfg: Cfg -> early window and cache identity
    - targets: np.ndarray -> [presentations, time, channels] target cache
    - indices: dict -> presentation indices per subset
    - channel_standardization: tuple -> train channel mean and scale

OUTPUT:
    - subset_early: dict -> split name to [presentations, early bins, channels]
    - covered_ms: tuple[float, float] -> the early window actually covered
"""
def extract_early_response(cfg, targets, indices, channel_standardization):
    bin_indices, covered_ms = select_window_bin_indices(
        targets.shape[1],
        cfg.target_fs,
        cfg.time_start_ms,
        cfg.early_start_ms,
        cfg.early_end_ms,
    )
    channel_mean, channel_scale = channel_standardization
    early_response = average_targets_over_bin_groups(targets, bin_indices, 1)
    subset_early = {
        subset_name: standardize_targets(
            early_response[subset_indices], channel_mean, channel_scale
        )
        for subset_name, subset_indices in indices.items()
    }
    return subset_early, covered_ms
# EOF


"""
reference_configuration
The hyperparameters a method runs at when nothing is being searched.

Keeping the reference in one place means the search's anchor and the plain
comparison are the same configuration by construction.

INPUT:
    - architecture: str -> "gru" or "lds"
    - cfg: Cfg -> widths, regularization, and optimizer settings

OUTPUT:
    - configuration: dict -> every knob the search can vary
"""
def reference_configuration(architecture, cfg):
    architecture_kwargs = {
        "gru": {
            "hidden_dim": cfg.gru_hidden_dim,
            "time_embedding_dim": cfg.gru_time_embedding_dim,
        },
        "lds": {"state_dim": cfg.lds_state_dim},
    }[architecture]
    return {
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "l2_lambda": cfg.l2_lambda,
        "l2_scope": cfg.l2_scope,
        "dropout": cfg.dropout,
        "input_noise_std": cfg.input_noise_std,
        **architecture_kwargs,
    }
# EOF


"""
sample_configurations
Build the configurations searched for one architecture.

The first entry is the reference of that architecture, so every draw is read
against the run the plain comparison already reported; the rest are independent
random draws from the shared and architecture-specific grids, deduplicated.

INPUT:
    - architecture: str -> "gru" or "lds"
    - n_configs: int -> configurations requested, reference included
    - rng: np.random.Generator -> sampler for the random draws
    - cfg: Cfg -> supplies the reference configuration

OUTPUT:
    - configurations: list[dict] -> hyperparameter dictionaries
"""
def sample_configurations(architecture, n_configs, rng, cfg):
    if n_configs <= 0:
        raise ValueError("n_configs_per_method must be positive.")
    # end if nothing would be searched
    reference = reference_configuration(architecture, cfg)
    configurations = [reference]
    seen = {tuple(sorted(reference.items()))}

    # Rejection sampling keeps the draws distinct without enumerating the grid.
    # The penalty is swept separately, so a draw inherits the reference value.
    grid = {**SHARED_SEARCH_SPACE, **ARCHITECTURE_SEARCH_SPACES[architecture]}
    inherited = {
        "l2_lambda": reference["l2_lambda"], "l2_scope": reference["l2_scope"]
    }
    attempts, max_attempts = 0, 50 * n_configs
    while len(configurations) < n_configs and attempts < max_attempts:
        attempts += 1
        candidate = {
            **inherited,
            **{
                name: values[int(rng.integers(len(values)))]
                for name, values in grid.items()
            },
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
build_method_model
Instantiate one decoder, with or without the early-response input.

INPUT:
    - architecture: str -> "gru" or "lds"
    - use_early_response: bool -> whether the decoder observes the early trace
    - shapes: dict -> n_layers, feature_dim, n_timepoints, and n_neurons
    - n_early_bins: int -> observed response bins
    - configuration: dict -> widths, dropout, and the feature-noise scale
    - cfg: Cfg -> settings the search never varies

OUTPUT:
    - model: nn.Module -> the requested decoder
"""
def build_method_model(
    architecture, use_early_response, shapes, n_early_bins, configuration, cfg
):
    # Optimizer settings reach the trainer through cfg, not the constructor.
    model_kwargs = {
        name: value
        for name, value in configuration.items()
        if name not in OPTIMIZER_HYPERPARAMETERS
    }
    return build_early_response_model(
        architecture,
        **shapes,
        **model_kwargs,
        n_early_bins=n_early_bins,
        early_dim=cfg.early_dim,
        use_early_response=use_early_response,
        temporal_noise_std=cfg.temporal_noise_std,
    )
# EOF


"""
per_epoch_slope
Least-squares slope per epoch of the last entries of one history field.

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
    return float(np.polyfit(np.arange(len(values), dtype=float), values, 1)[0])
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
    return {
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "stopped_early": epochs_run < cfg.epochs,
        "train_mse_percent_per_epoch": round(
            100.0 * train_slope / history[-1]["train_mse"], 4
        ),
        "validation_stim_r_slope_per_epoch": round(validation_slope, 6),
        # A checkpoint landing in the final epochs of an exhausted budget is
        # the practical sign that more epochs would still have helped.
        "still_improving_at_budget": bool(
            epochs_run == cfg.epochs
            and best_epoch >= epochs_run - 2
            and validation_slope > 0.0
        ),
    }
# EOF


"""
run_decoder_method
Train one recurrent decoder and score it on the repeated-test presentations.

INPUT:
    - method: str -> key in METHOD_SPECS
    - loaders: dict -> train, validation, and test loaders
    - shapes: dict -> data shapes shared by every method
    - scoring: dict -> test image ids, fit-split mean, ceiling, response slice
    - n_early_bins: int -> observed response bins
    - configuration: dict -> the hyperparameters this run is fitted at
    - run_id: str -> name written into the row and the stored file
    - cfg: Cfg -> settings the search never varies
    - device: torch.device -> compute device

OUTPUT:
    - row: dict -> metrics, optimization trend, and settings
    - site_correlations: np.ndarray -> [time, channels] test stim_r
    - history: list[dict] -> per-epoch optimization record
"""
def run_decoder_method(
    method, loaders, shapes, scoring, n_early_bins, configuration, run_id, cfg, device
):
    architecture, use_early_response = METHOD_SPECS[method]

    # Every method starts from the same initialization stream, so runs differ
    # by what they are given rather than by their random weights.
    torch.manual_seed(
        cfg.random_seed if cfg.init_seed < 0 else cfg.init_seed
    )
    model = build_method_model(
        architecture, use_early_response, shapes, n_early_bins, configuration, cfg
    ).to(device)

    # The trainer reads its optimizer settings off cfg, so a draw's values are
    # applied to a copy rather than to the shared configuration object.
    run_cfg = replace(
        cfg,
        **{
            name: configuration[name]
            for name in OPTIMIZER_HYPERPARAMETERS
            if name in configuration
        },
    )
    run_start = time.perf_counter()
    history, best_epoch, _, best_validation_stim_r = train_cached_feature_decoder(
        model, loaders, run_cfg, device, verbose=False
    )
    train_seconds = time.perf_counter() - run_start

    predictions, targets = evaluate_cached_feature_decoder(
        model, loaders["test"], device
    )
    row, site_correlations = score_and_decompose(
        predictions, targets, scoring, run_id, REDUCER
    )
    row.update(
        {
            "method": method,
            "architecture": architecture,
            "uses_early_response": use_early_response,
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "validation_stim_r": round(best_validation_stim_r, 4),
            "train_seconds": round(train_seconds, 1),
            **describe_optimization(history, best_epoch, cfg),
            **{name: configuration[name] for name in sorted(configuration)},
        }
    )
    add_disjoint_half_metric(row, predictions, targets, scoring, cfg)
    if architecture == "lds":
        # The learned dynamics are only interpretable through their spectrum.
        row["spectral_radius"] = round(model.spectral_radius(), 4)
    # end if the run fitted a linear dynamical system

    del model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    # end if the device caches its allocations
    return row, site_correlations, history
# EOF


"""
run_ridge_method
Fit the linear reference on the same inputs the decoders receive.

The design matrix is per presentation, not per stimulus, because with the early
trace the 30 repetitions of a test image no longer share an input. Without it
they do, so the image-only ridge reduces to the usual per-stimulus map.

INPUT:
    - method: str -> key in METHOD_SPECS
    - subset_features: dict -> [presentations, cells, channels] per split
    - subset_early: dict -> [presentations, early bins, channels] per split
    - subset_targets: dict -> standardized targets per split
    - scoring: dict -> test image ids, fit-split mean, ceiling, response slice
    - cfg: Cfg -> resample settings of the disjoint-half metric

OUTPUT:
    - row: dict -> metrics and settings
    - site_correlations: np.ndarray -> [time, channels] test stim_r
"""
def run_ridge_method(
    method, subset_features, subset_early, subset_targets, scoring, cfg
):
    _, use_early_response = METHOD_SPECS[method]
    design_matrices = {}
    for subset_name, features in subset_features.items():
        blocks = [features.reshape(len(features), -1)]
        if use_early_response:
            blocks.append(
                subset_early[subset_name].reshape(len(features), -1)
            )
        # end if the early trace is part of the design matrix
        design_matrices[subset_name] = np.concatenate(blocks, axis=1)
    # end for split

    run_start = time.perf_counter()
    ridge_fit = fit_ridge_from_presentations(design_matrices, subset_targets)
    train_seconds = time.perf_counter() - run_start

    row, site_correlations = score_and_decompose(
        ridge_fit["trial_predictions"],
        subset_targets["test"],
        scoring,
        method,
        REDUCER,
    )
    row.update(
        {
            "method": method,
            "architecture": "ridge",
            "uses_early_response": use_early_response,
            "trainable_parameters": ridge_fit["n_coefficients"],
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
            "train_seconds": round(train_seconds, 1),
        }
    )
    add_disjoint_half_metric(
        row,
        ridge_fit["trial_predictions"],
        subset_targets["test"],
        scoring,
        cfg,
    )
    return row, site_correlations
# EOF


"""
add_disjoint_half_metric
Score one method again with predictions and targets from disjoint repetitions.

The ordinary stim_r lets a decoder that reads a presentation's own early bins
profit from noise those bins share with the response window, because both sides
of the correlation average over the same 30 repetitions. This metric closes
that path, so the difference between the two numbers is the size of it.

INPUT:
    - row: dict -> the method's record, extended in place
    - trial_predictions: np.ndarray -> [presentations, time, neurons]
    - trial_targets: np.ndarray -> [presentations, time, neurons]
    - scoring: dict -> test image ids, response slice, and the matched reference
    - cfg: Cfg -> resample count and seed

OUTPUT:
    - row: dict -> the same record, with the cross-half scores added
"""
def add_disjoint_half_metric(row, trial_predictions, trial_targets, scoring, cfg):
    correlations = disjoint_half_stimulus_correlation(
        trial_predictions,
        trial_targets,
        scoring["test_image_ids"],
        N_TEST_IMAGES,
        n_resamples=cfg.noise_ceiling_resamples,
        seed=cfg.random_seed,
    )
    response_correlations = correlations[scoring["response_slice"]]
    row["disjoint_half_stim_r"] = round(
        float(np.nanmean(response_correlations)), 4
    )
    row["disjoint_half_fraction_of_reference"] = round(
        float(
            np.nanmean(response_correlations)
            / np.nanmean(scoring["disjoint_half_reference"][scoring["response_slice"]])
        ),
        3,
    )
    return row
# EOF


"""
save_finished_run
Store one finished method so a later run reuses it instead of repeating it.

INPUT:
    - run_dir: Path -> directory holding one file per finished method
    - row: dict -> the method's metrics, keyed by its name
    - history: list[dict] -> per-epoch record, empty for ridge
    - site_correlations: np.ndarray -> [time, channels] test stim_r

OUTPUT:
    - None: writes the method's record and its per-site correlations
"""
def save_finished_run(run_dir, row, history, site_correlations):
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / f"{row['model']}_site_stim_r.npy", site_correlations)
    with open(run_dir / f"{row['model']}.json", "w") as run_file:
        json.dump({"row": row, "history": history}, run_file, indent=2)
    # end with saved run record
# EOF


"""
load_finished_run
Read one stored method back, or report that it has not been run yet.

INPUT:
    - run_dir: Path -> directory holding one file per finished method
    - method: str -> name of the requested method

OUTPUT:
    - finished: tuple | None -> (row, history, site_correlations), or None
"""
def load_finished_run(run_dir, method):
    record_path = run_dir / f"{method}.json"
    site_path = run_dir / f"{method}_site_stim_r.npy"
    if not (record_path.is_file() and site_path.is_file()):
        return None
    # end if this method has not finished before
    with open(record_path, "r") as run_file:
        record = json.load(run_file)
    # end with stored run record
    return record["row"], record["history"], np.load(site_path)
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
obtain_ridge_rows
Load the two ridge references, fitting them only if they are not on disk.

The search is judged against ridge, and ridge is deterministic, so re-fitting it
for every search would just spend minutes reproducing two stored numbers.

INPUT:
    - cfg: Cfg -> resample settings
    - run_dir: Path -> directory of the plain comparison's finished runs
    - subset_features, subset_early, subset_targets: dict -> per-split arrays
    - scoring: dict -> test image ids, ceiling, response slice

OUTPUT:
    - ridge_rows: dict -> method name to its record
"""
def obtain_ridge_rows(
    cfg, run_dir, subset_features, subset_early, subset_targets, scoring
):
    ridge_rows = {}
    for method in ("ridge_image_only", "ridge_early"):
        finished = load_finished_run(run_dir, method)
        if finished is not None:
            ridge_rows[method] = finished[0]
            continue
        # end if the reference was already fitted
        row, site_r = run_ridge_method(
            method, subset_features, subset_early, subset_targets, scoring, cfg
        )
        save_finished_run(run_dir, row, [], site_r)
        ridge_rows[method] = row
    # end for ridge reference
    return ridge_rows
# EOF


"""
run_hyperparameter_search
Re-fit the named methods over random draws and rank them against ridge.

Selection is on validation stim_r, which is sound for a decoder whose only
input is the image. A method that also reads the early response inflates that
criterion with trial-noise coupling, so for those the search reports the
cross-half score alongside it and says so.

INPUT:
    - cfg: Cfg -> search extent and the settings a draw does not vary
    - loaders: dict -> train, validation, and test loaders
    - shapes: dict -> data shapes shared by every run
    - scoring: dict -> test image ids, fit-split mean, ceiling, response slice
    - n_early_bins: int -> observed response bins
    - ridge_rows: dict -> the linear references this search is judged against
    - output_dir: Path -> destination of the search records
    - device: torch.device -> compute device

OUTPUT:
    - rows: list[dict] -> every completed search run
"""
def run_hyperparameter_search(
    cfg, loaders, shapes, scoring, n_early_bins, ridge_rows, output_dir, device
):
    methods = [
        name.strip() for name in cfg.search_methods.split(",") if name.strip()
    ]
    unknown = sorted(set(methods) - set(METHOD_SPECS))
    if unknown:
        raise KeyError(f"Unknown search methods: {unknown}.")
    # end if a method name is invalid
    if any(METHOD_SPECS[method][0] == "ridge" for method in methods):
        raise ValueError("Ridge has no hyperparameters to search here.")
    # end if a linear reference was put in the search list

    search_dir = output_dir / "search"
    search_rng = np.random.default_rng(cfg.search_seed)
    rows = []
    for method in methods:
        architecture, use_early_response = METHOD_SPECS[method]
        configurations = sample_configurations(
            architecture, cfg.n_configs_per_method, search_rng, cfg
        )
        print(
            f"\n{method}: {len(configurations)} configurations "
            f"({cfg.epochs} epochs max each)"
        )
        if use_early_response:
            print(
                "  note: validation stim_r is inflated for this method, so its "
                "ranking here is not trustworthy"
            )
        # end if the searched method reads the early response
        for config_index, configuration in enumerate(configurations):
            run_id = f"{method}_{config_index:02d}"
            finished = (
                None
                if cfg.overwrite_finished_runs
                else load_finished_run(search_dir, run_id)
            )
            if finished is not None:
                row, history, site_r = finished
            else:
                row, site_r, history = run_decoder_method(
                    method,
                    loaders,
                    shapes,
                    scoring,
                    n_early_bins,
                    configuration,
                    run_id,
                    cfg,
                    device,
                )
                save_finished_run(search_dir, row, history, site_r)
            # end if this configuration was reused or trained
            rows.append(row)
            print(
                f"  {run_id:<22} val r {row['validation_stim_r']:.4f} | test r "
                f"{row['mean_stim_r_response']:.4f} | cross-half "
                f"{row['disjoint_half_stim_r']:.4f} | epoch "
                f"{row['best_epoch']:02d}/{row['epochs_run']:02d} | "
                f"{row['train_seconds']:.0f}s"
                + ("" if finished is None else " | reused")
            )
            # Partial results survive an interrupted search.
            save_rows(output_dir / "search_results.csv", rows)
        # end for configuration
    # end for searched method

    with open(output_dir / "search_results.json", "w") as results_file:
        json.dump(
            {
                "runs": rows,
                "ridge": ridge_rows,
                "shared_search_space": SHARED_SEARCH_SPACE,
                "architecture_search_spaces": ARCHITECTURE_SEARCH_SPACES,
                "search_seed": cfg.search_seed,
                "epochs": cfg.epochs,
            },
            results_file,
            indent=2,
        )
    # end with saved search results
    report_search(cfg, rows, ridge_rows, methods)
    return rows
# EOF


"""
best_searched_configuration
Read one method's best searched configuration back off disk.

The penalty sweep is only interesting against a decoder that has already been
tuned, so it starts from the configuration --search selected rather than from
the script defaults.

INPUT:
    - method: str -> key in METHOD_SPECS
    - output_dir: Path -> directory holding search_results.json
    - cfg: Cfg -> fallback when no search has been run

OUTPUT:
    - configuration: dict -> the winning configuration, or the reference
    - source: str -> where the configuration came from, for the printed header
"""
def best_searched_configuration(method, output_dir, cfg):
    results_path = output_dir / "search_results.json"
    architecture = METHOD_SPECS[method][0]
    if not results_path.is_file():
        return reference_configuration(architecture, cfg), "script defaults"
    # end if no search has been run for this output directory
    with open(results_path, "r") as results_file:
        rows = json.load(results_file)["runs"]
    # end with stored search results
    method_rows = [row for row in rows if row.get("method") == method]
    if not method_rows:
        return reference_configuration(architecture, cfg), "script defaults"
    # end if the search never covered this method
    best = max(method_rows, key=lambda row: row["validation_stim_r"])
    configuration = reference_configuration(architecture, cfg)
    # Only the knobs the search actually varied are taken from the winner.
    for name in configuration:
        if name in best:
            configuration[name] = best[name]
        # end if the search recorded this knob
    # end for configuration knob
    return configuration, best["model"]
# EOF


"""
l2_sweep_configurations
Enumerate the penalty grid applied to one tuned configuration.

Every combination is run twice, once keeping the decoupled weight decay the
search chose and once with it switched off, because the question is both
whether an explicit penalty helps and whether it replaces the decoupled one.
The unpenalized anchor appears once per weight-decay setting; scope is
meaningless there and is fixed to "none" so the run is not duplicated.

INPUT:
    - base: dict -> the tuned configuration to start from
    - cfg: Cfg -> the swept lambdas and scopes

OUTPUT:
    - configurations: list[tuple[str, dict]] -> label and configuration
"""
def l2_sweep_configurations(base, cfg):
    lambdas = [float(value) for value in cfg.l2_lambdas.split(",") if value.strip()]
    scopes = [value.strip() for value in cfg.l2_scopes.split(",") if value.strip()]
    unknown = sorted(set(scopes) - {"readout", "all"})
    if unknown:
        raise ValueError(f"Unknown l2 scopes: {unknown}.")
    # end if a scope name is invalid

    configurations = []
    for keep_weight_decay in (True, False):
        weight_decay = base["weight_decay"] if keep_weight_decay else 0.0
        decay_tag = "wd" if keep_weight_decay else "nowd"
        configurations.append(
            (
                f"{decay_tag}_l2none",
                {**base, "weight_decay": weight_decay,
                 "l2_lambda": 0.0, "l2_scope": "none"},
            )
        )
        for scope in scopes:
            for l2_lambda in lambdas:
                configurations.append(
                    (
                        f"{decay_tag}_{scope}_{l2_lambda:g}",
                        {**base, "weight_decay": weight_decay,
                         "l2_lambda": l2_lambda, "l2_scope": scope},
                    )
                )
            # end for penalty weight
        # end for penalty scope
    # end for weight-decay setting
    return configurations
# EOF


"""
run_l2_sweep
Re-fit each method's tuned configuration across the explicit-penalty grid.

INPUT:
    - cfg: Cfg -> the swept grid and the settings a run does not vary
    - loaders: dict -> train, validation, and test loaders
    - shapes: dict -> data shapes shared by every run
    - scoring: dict -> test image ids, fit-split mean, ceiling, response slice
    - n_early_bins: int -> observed response bins
    - ridge_rows: dict -> the linear references the sweep is judged against
    - output_dir: Path -> destination of the sweep records
    - device: torch.device -> compute device

OUTPUT:
    - rows: list[dict] -> every completed sweep run
"""
def run_l2_sweep(
    cfg, loaders, shapes, scoring, n_early_bins, ridge_rows, output_dir, device
):
    methods = [
        name.strip() for name in cfg.search_methods.split(",") if name.strip()
    ]
    unknown = sorted(set(methods) - set(METHOD_SPECS))
    if unknown:
        raise KeyError(f"Unknown methods: {unknown}.")
    # end if a method name is invalid

    equivalent_lambda = ridge_equivalent_l2_lambda(
        ridge_rows["ridge_image_only"]["ridge_alpha"],
        len(loaders["train"].dataset),
        shapes["n_timepoints"],
        shapes["n_neurons"],
    )
    print(
        f"\nridge alpha {ridge_rows['ridge_image_only']['ridge_alpha']:.4g} is "
        f"lambda {equivalent_lambda:.3g} under a mean-MSE loss"
    )

    sweep_dir = output_dir / "l2_sweep"
    rows = []
    for method in methods:
        base, source = best_searched_configuration(method, output_dir, cfg)
        configurations = l2_sweep_configurations(base, cfg)
        print(
            f"\n{method}: {len(configurations)} penalty settings on top of "
            f"{source}"
        )
        for label, configuration in configurations:
            run_id = f"{method}_{label}"
            finished = (
                None
                if cfg.overwrite_finished_runs
                else load_finished_run(sweep_dir, run_id)
            )
            if finished is not None:
                row, history, site_r = finished
            else:
                row, site_r, history = run_decoder_method(
                    method,
                    loaders,
                    shapes,
                    scoring,
                    n_early_bins,
                    configuration,
                    run_id,
                    cfg,
                    device,
                )
                save_finished_run(sweep_dir, row, history, site_r)
            # end if this setting was reused or trained
            rows.append(row)
            print(
                f"  {label:<22} val r {row['validation_stim_r']:.4f} | test r "
                f"{row['mean_stim_r_response']:.4f} | cross-half "
                f"{row['disjoint_half_stim_r']:.4f} | epoch "
                f"{row['best_epoch']:02d}/{row['epochs_run']:02d}"
                + ("" if finished is None else " | reused")
            )
            save_rows(output_dir / "l2_sweep_results.csv", rows)
        # end for penalty setting
    # end for method

    with open(output_dir / "l2_sweep_results.json", "w") as results_file:
        json.dump(
            {
                "runs": rows,
                "ridge": ridge_rows,
                "ridge_equivalent_l2_lambda": equivalent_lambda,
                "l2_lambdas": cfg.l2_lambdas,
                "l2_scopes": cfg.l2_scopes,
            },
            results_file,
            indent=2,
        )
    # end with saved sweep results
    report_l2_sweep(rows, ridge_rows, methods)
    return rows
# EOF


"""
report_l2_sweep
Print the sweep table and say whether the penalty moved anything.

INPUT:
    - rows: list[dict] -> completed sweep runs
    - ridge_rows: dict -> the linear references
    - methods: list[str] -> swept methods, in order

OUTPUT:
    - None: prints the tables
"""
def report_l2_sweep(rows, ridge_rows, methods):
    header = (
        f"{'run':<40}{'val r':>8}{'test r':>9}{'cross-half':>12}{'epochs':>10}"
    )
    print("\nEvery penalty setting, ranked by validation stim_r")
    print(header)
    print("-" * len(header))
    for row in sorted(rows, key=lambda row: -row["validation_stim_r"]):
        print(
            f"{row['model']:<40}"
            f"{row['validation_stim_r']:>8.4f}"
            f"{row['mean_stim_r_response']:>9.4f}"
            f"{row['disjoint_half_stim_r']:>12.4f}"
            f"{row['best_epoch']:>6d}/{row['epochs_run']:<3d}"
        )
    # end for sweep run

    print("\nDid the explicit penalty help?")
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        if not method_rows:
            continue
        # end if this method produced no run
        anchor = max(
            (row for row in method_rows if row["l2_lambda"] == 0.0),
            key=lambda row: row["validation_stim_r"],
        )
        best = max(method_rows, key=lambda row: row["validation_stim_r"])
        reference = ridge_rows[
            "ridge_early" if METHOD_SPECS[method][1] else "ridge_image_only"
        ]
        print(f"  {method}")
        print(
            f"    best unpenalized  {anchor['model'].split('_')[-1]:<10} "
            f"test {anchor['mean_stim_r_response']:.4f} | cross-half "
            f"{anchor['disjoint_half_stim_r']:.4f}"
        )
        print(
            f"    best overall      {best['model'].replace(method + '_', ''):<10} "
            f"test {best['mean_stim_r_response']:.4f} | cross-half "
            f"{best['disjoint_half_stim_r']:.4f}"
        )
        difference = (
            best["mean_stim_r_response"] - anchor["mean_stim_r_response"]
        )
        print(
            f"    penalty is worth  {difference:+.4f} test stim_r; ridge sits "
            f"at {reference['mean_stim_r_response']:.4f}"
        )
    # end for swept method
# EOF


"""
report_search
Print the search tables: every run, then each method's best against ridge.

INPUT:
    - cfg: Cfg -> epoch budget, used to read the convergence flags
    - rows: list[dict] -> completed search runs
    - ridge_rows: dict -> the linear references
    - methods: list[str] -> searched methods, in order

OUTPUT:
    - None: prints the tables
"""
def report_search(cfg, rows, ridge_rows, methods):
    header = (
        f"{'run':<22}{'val r':>8}{'test r':>9}{'cross-half':>12}{'lr':>9}"
        f"{'wd':>8}{'drop':>7}{'noise':>7}{'width':>8}{'epochs':>9}"
    )
    print("\nEvery searched run, ranked by validation stim_r")
    print(header)
    print("-" * len(header))
    for row in sorted(rows, key=lambda row: -row["validation_stim_r"]):
        width = row.get("hidden_dim") or row.get("state_dim")
        print(
            f"{row['model']:<22}"
            f"{row['validation_stim_r']:>8.4f}"
            f"{row['mean_stim_r_response']:>9.4f}"
            f"{row['disjoint_half_stim_r']:>12.4f}"
            f"{row['learning_rate']:>9.0e}"
            f"{row['weight_decay']:>8.0e}"
            f"{row['dropout']:>7.1f}"
            f"{row['input_noise_std']:>7.2f}"
            f"{width:>8d}"
            f"{row['best_epoch']:>5d}/{row['epochs_run']:<3d}"
        )
    # end for searched run

    print("\nBest of each method against its ridge reference")
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        if not method_rows:
            continue
        # end if this method produced no run
        best = max(method_rows, key=lambda row: row["validation_stim_r"])
        reference = ridge_rows[
            "ridge_early" if METHOD_SPECS[method][1] else "ridge_image_only"
        ]
        print(f"  {method}: {best['model']}")
        for label, key in (
            ("test stim_r ", "mean_stim_r_response"),
            ("cross-half r", "disjoint_half_stim_r"),
        ):
            difference = best[key] - reference[key]
            verdict = "beats ridge" if difference > 0 else "behind ridge"
            print(
                f"    {label}  {best[key]:.4f}  vs ridge {reference[key]:.4f}"
                f"  ({difference:+.4f}, {verdict})"
            )
        # end for reported metric
        searched = sorted(
            set(SHARED_SEARCH_SPACE)
            | set(ARCHITECTURE_SEARCH_SPACES[METHOD_SPECS[method][0]])
        )
        print(
            "    at "
            + ", ".join(f"{name}={best[name]}" for name in searched)
        )
        print(
            f"    epochs {best['epochs_run']}, best {best['best_epoch']}, "
            f"still improving at budget: {best['still_improving_at_budget']}"
        )
    # end for searched method
# EOF


"""
plot_comparison
Draw the four panels that carry the comparison.

INPUT:
    - cfg: Cfg -> windows and backbone used in the titles
    - rows: list[dict] -> one record per finished method
    - histories: dict -> per-epoch records of the trained decoders
    - site_correlations: dict -> [time, channels] test stim_r per method
    - ceiling: np.ndarray -> [time, channels] noise ceiling
    - bin_edges_ms: np.ndarray -> edges of the output time bins
    - output_path: Path -> destination figure file

OUTPUT:
    - None: writes the figure to disk
"""
def plot_comparison(
    cfg, rows, histories, site_correlations, ceiling, bin_edges_ms, output_path
):
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    for axis in axes.flat:
        axis.grid(True, color=GRID_COLOR, linewidth=0.8)
        axis.set_axisbelow(True)
        for spine_name in ("top", "right"):
            axis.spines[spine_name].set_visible(False)
        # end for hidden spine
    # end for panel

    rows_by_method = {row["model"]: row for row in rows}
    mean_ceiling = float(np.nanmean(ceiling))

    # --- panel 1: the headline, test stim_r of every method ---
    ranked = sorted(rows, key=lambda row: row["mean_stim_r_response"])
    positions = np.arange(len(ranked))
    axes[0, 0].barh(
        positions,
        [row["mean_stim_r_response"] for row in ranked],
        height=0.6,
        color=[METHOD_COLORS[row["model"]] for row in ranked],
    )
    axes[0, 0].axvline(
        mean_ceiling, linewidth=2, color=CEILING_COLOR, label="noise ceiling"
    )
    for position, row in zip(positions, ranked):
        # A direct label removes the need to read each bar against the axis.
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
    axes[0, 0].set_yticklabels([METHOD_LABELS[row["model"]] for row in ranked])
    axes[0, 0].set_xlim(0.0, mean_ceiling + 0.06)
    axes[0, 0].set_xlabel("test stim_r, mean over sites and 20 ms bins")
    axes[0, 0].set_title("Every method, on identical inputs")
    axes[0, 0].legend(frameon=False, fontsize=9, loc="lower right")

    # --- panel 2: the gain the early trace bought, scored two ways ---
    # The pair matters more than either bar: a gain that is present under the
    # ordinary metric and gone under the cross-half one is shared trial noise,
    # not image information.
    architectures = ["gru", "lds", "ridge"]
    metric_names = ("mean_stim_r_response", "disjoint_half_stim_r")
    metric_labels = ("same repetitions (stim_r)", "disjoint repetitions")
    bar_offsets = (-0.19, 0.19)
    for metric_index, metric_name in enumerate(metric_names):
        drawn_positions, drawn_gains, drawn_colors = [], [], []
        for index, architecture in enumerate(architectures):
            early_row = rows_by_method.get(f"{architecture}_early")
            plain_row = rows_by_method.get(f"{architecture}_image_only")
            if early_row is None or plain_row is None:
                continue
            # end if this mapping was not run on both sides
            drawn_positions.append(index + bar_offsets[metric_index])
            drawn_gains.append(early_row[metric_name] - plain_row[metric_name])
            drawn_colors.append(METHOD_COLORS[f"{architecture}_early"])
        # end for architecture
        axes[0, 1].bar(
            drawn_positions,
            drawn_gains,
            width=0.34,
            color=drawn_colors,
            # The two metrics are told apart by the hatch, so each mapping
            # keeps the one colour it has in every other panel.
            hatch="" if metric_index == 0 else "///",
            edgecolor="white",
            linewidth=0.8,
        )
        for position, gain in zip(drawn_positions, drawn_gains):
            axes[0, 1].text(
                position,
                gain,
                f"{gain:+.3f}",
                ha="center",
                va="bottom" if gain >= 0 else "top",
                fontsize=8,
                color="#3d3d3a",
            )
        # end for drawn mapping
    # end for scoring metric
    axes[0, 1].axhline(0.0, linewidth=1.5, color="#3d3d3a")
    # The bars carry the mapping in their colour, so the legend explains the
    # hatch alone rather than repeating one mapping's hue.
    axes[0, 1].legend(
        [
            plt.Rectangle((0, 0), 1, 1, facecolor="#6f6f6b", hatch=hatch,
                          edgecolor="white")
            for hatch in ("", "///")
        ],
        metric_labels,
        frameon=False,
        fontsize=9,
        loc="best",
    )
    axes[0, 1].set_xticks(np.arange(len(architectures)))
    axes[0, 1].set_xticklabels(["GRU", "LDS", "Ridge"])
    axes[0, 1].set_ylabel("gain over the image-only ablation")
    axes[0, 1].set_title(
        f"What the {cfg.early_start_ms:g}-{cfg.early_end_ms:g} ms trace adds"
    )

    # --- panel 3: optimization of the trained decoders ---
    for method, history in sorted(histories.items()):
        if not history:
            continue
        # end if this method is not an optimized decoder
        axes[1, 0].plot(
            [entry["epoch"] for entry in history],
            [entry["validation_stim_r"] for entry in history],
            linewidth=2,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    # end for trained decoder
    for method in ("ridge_early", "ridge_image_only"):
        if method in rows_by_method:
            axes[1, 0].axhline(
                rows_by_method[method]["validation_stim_r"],
                linewidth=2,
                linestyle="--",
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        # end if this reference was fitted
    # end for ridge reference
    axes[1, 0].set_xlabel(f"epoch (budget {cfg.epochs})")
    axes[1, 0].set_ylabel("validation stim_r (single trials)")
    axes[1, 0].set_title("Optimization")
    axes[1, 0].legend(frameon=False, fontsize=8, loc="lower right", ncol=2)

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
    for method, correlations in sorted(site_correlations.items()):
        axes[1, 1].plot(
            bin_centers,
            np.nanmean(correlations, axis=1),
            linewidth=2,
            marker="o",
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=0.8,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
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
        f"TVSD monkey F {cfg.area} from {cfg.model_name} "
        f"({cfg.spatial_pool_size}x{cfg.spatial_pool_size} pooled) with and "
        f"without the observed {cfg.early_start_ms:g}-{cfg.early_end_ms:g} ms "
        f"response, {cfg.timebin_ms:g} ms bins "
        f"{cfg.window_start_ms:g}-{cfg.window_end_ms:g} ms"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
# EOF


def main():
    cfg = parse_args()
    if cfg.smoke_test:
        cfg.epochs = min(cfg.epochs, 3)
        cfg.minimum_epochs = 1
        cfg.patience = 1
        cfg.noise_ceiling_resamples = 4
    # end if a smoke run was requested

    paths = load_project_paths(cfg)
    device = resolve_device(cfg.device)
    output_dir = Path(
        cfg.output_dir
        or PROJECT_ROOT / "results" / f"tvsd_early_response_{cfg.model_name}"
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = [name.strip() for name in cfg.methods.split(",") if name.strip()]
    unknown = sorted(set(methods) - set(METHOD_SPECS))
    if unknown:
        raise KeyError(f"Unknown methods: {unknown}.")
    # end if a method name is invalid

    # --- the shared inputs: pooled conv features and the cached response ---
    targets, allmat = load_cached_targets(cfg, paths)
    pooled_features, feature_shape = load_pooled_spatial_features(
        cfg, paths, cfg.spatial_pool_size
    )
    print(
        f"{cfg.model_name} map {feature_shape} pooled to "
        f"{cfg.spatial_pool_size}x{cfg.spatial_pool_size} -> "
        f"{pooled_features['train'].shape[1:]} per stimulus"
    )

    data = prepare_timebin_data(
        cfg,
        targets,
        pooled_features["train"],
        pooled_features["test"],
        allmat,
        device,
    )
    shapes, scoring, indices = data["shapes"], data["scoring"], data["indices"]
    subset_targets = data["subset_targets"]
    bin_edges_ms = data["bin_edges_ms"]
    # Standardizing on the fit split is what ridge does internally anyway, so
    # applying it once here puts every compared method on identical inputs.
    subset_features = standardize_features_on_fit_split(data["subset_features"])

    # The reference for the cross-half metric: the same quantity computed on
    # the recording itself, i.e. how well one half of an image's repetitions
    # predicts the other. Every method's cross-half score is read against it.
    scoring["disjoint_half_reference"] = disjoint_half_stimulus_correlation(
        subset_targets["test"],
        subset_targets["test"],
        scoring["test_image_ids"],
        N_TEST_IMAGES,
        n_resamples=cfg.noise_ceiling_resamples,
        seed=cfg.random_seed,
    )

    subset_early, early_covered_ms = extract_early_response(
        cfg, targets, indices, data["channel_standardization"]
    )
    n_early_bins = subset_early["train"].shape[1]
    print(
        f"observed early window {early_covered_ms[0]:g}-{early_covered_ms[1]:g}"
        f" ms -> {n_early_bins} bins of {1000.0 / cfg.target_fs:g} ms | "
        f"dropout {cfg.dropout:g}, feature noise {cfg.input_noise_std:g}"
    )

    # One set of loaders serves every decoder: the image-only ablation is handed
    # the early trace as well and ignores it, so nothing else differs.
    loaders = {
        subset_name: make_multi_input_loader(
            [subset_features[subset_name], subset_early[subset_name]],
            subset_targets[subset_name],
            cfg,
            shuffle=subset_name == "train",
        )
        for subset_name in ("train", "validation", "test")
    }

    run_dir = output_dir / "runs"
    if cfg.search or cfg.l2_sweep:
        # The search is judged against ridge on identical inputs, so the two
        # references are loaded (or fitted once) before anything is trained.
        ridge_rows = obtain_ridge_rows(
            cfg, run_dir, subset_features, subset_early, subset_targets, scoring
        )
        driver = run_l2_sweep if cfg.l2_sweep else run_hyperparameter_search
        driver(
            cfg,
            loaders,
            shapes,
            scoring,
            n_early_bins,
            ridge_rows,
            output_dir,
            device,
        )
        print(f"\nsaved to {output_dir}")
        return
    # end if a hyperparameter search was requested

    # --- the comparison itself ---
    rows, histories, site_correlations = [], {}, {}
    for method in methods:
        finished = (
            None
            if cfg.overwrite_finished_runs
            else load_finished_run(run_dir, method)
        )
        if finished is not None:
            row, history, site_r = finished
        elif METHOD_SPECS[method][0] == "ridge":
            row, site_r = run_ridge_method(
                method,
                subset_features,
                subset_early,
                subset_targets,
                scoring,
                cfg,
            )
            history = []
            save_finished_run(run_dir, row, history, site_r)
        else:
            row, site_r, history = run_decoder_method(
                method,
                loaders,
                shapes,
                scoring,
                n_early_bins,
                reference_configuration(METHOD_SPECS[method][0], cfg),
                method,
                cfg,
                device,
            )
            save_finished_run(run_dir, row, history, site_r)
        # end if this method was reused, fitted, or trained
        rows.append(row)
        histories[method] = history
        site_correlations[method] = site_r
        print(
            f"  {method:<18} val r {row['validation_stim_r']:.4f} | test r "
            f"{row['mean_stim_r_response']:.4f} | ceil frac "
            f"{row['fraction_of_ceiling']:.3f} | MSE {row['test_mse']:.5f} | "
            f"{row['train_seconds']:.0f}s"
            + ("" if finished is None else " | reused")
        )
        # Partial results survive an interrupted comparison.
        save_rows(output_dir / "early_response_results.csv", rows)
    # end for method

    with open(output_dir / "config.json", "w") as config_file:
        json.dump(
            {
                **asdict(cfg),
                "covered_window_ms": list(data["covered_ms"]),
                "covered_early_window_ms": list(early_covered_ms),
                "bin_edges_ms": bin_edges_ms.tolist(),
                "n_early_bins": n_early_bins,
                "feature_shape": list(feature_shape),
                "n_sites": shapes["n_neurons"],
                "mean_noise_ceiling": round(
                    float(np.nanmean(scoring["ceiling"])), 4
                ),
                "mean_disjoint_half_reference": round(
                    float(np.nanmean(scoring["disjoint_half_reference"])), 4
                ),
                "n_fit_presentations": len(indices["train"]),
                "n_validation_presentations": len(indices["validation"]),
                "n_test_presentations": len(indices["test"]),
            },
            config_file,
            indent=2,
        )
    # end with saved configuration
    with open(output_dir / "early_response_results.json", "w") as results_file:
        json.dump({"runs": rows, "histories": histories}, results_file, indent=2)
    # end with saved results
    np.savez_compressed(
        output_dir / "site_stim_r.npz",
        ceiling=scoring["ceiling"],
        disjoint_half_reference=scoring["disjoint_half_reference"],
        bin_edges_ms=bin_edges_ms,
        **site_correlations,
    )
    plot_comparison(
        cfg,
        rows,
        histories,
        site_correlations,
        scoring["ceiling"],
        bin_edges_ms,
        output_dir / "early_response_comparison.png",
    )

    # The printed tables double as the accessible alternative to the figure.
    header = (
        f"{'method':<26}{'val r':>8}{'test r':>9}{'frac ceil':>11}"
        f"{'cross-half r':>14}{'frac ref':>10}{'test MSE':>10}{'params':>12}"
    )
    print("\nEvery method, ranked by test stim_r")
    print(header)
    print("-" * len(header))
    for row in sorted(rows, key=lambda row: -row["mean_stim_r_response"]):
        print(
            f"{METHOD_LABELS[row['model']]:<26}"
            f"{row['validation_stim_r']:>8.4f}"
            f"{row['mean_stim_r_response']:>9.4f}"
            f"{row['fraction_of_ceiling']:>11.3f}"
            f"{row['disjoint_half_stim_r']:>14.4f}"
            f"{row['disjoint_half_fraction_of_reference']:>10.3f}"
            f"{row['test_mse']:>10.5f}"
            f"{row['trainable_parameters']:>12,}"
        )
    # end for ranked method

    print("\nWhat the observed early response bought each mapping")
    rows_by_method = {row["model"]: row for row in rows}
    for architecture in ("gru", "lds", "ridge"):
        early_row = rows_by_method.get(f"{architecture}_early")
        plain_row = rows_by_method.get(f"{architecture}_image_only")
        if early_row is None or plain_row is None:
            continue
        # end if this mapping was not run on both sides
        print(
            f"  {architecture:<6}"
            f"stim_r {plain_row['mean_stim_r_response']:.4f} -> "
            f"{early_row['mean_stim_r_response']:.4f} "
            f"({early_row['mean_stim_r_response'] - plain_row['mean_stim_r_response']:+.4f})"
            f"  |  cross-half {plain_row['disjoint_half_stim_r']:.4f} -> "
            f"{early_row['disjoint_half_stim_r']:.4f} "
            f"({early_row['disjoint_half_stim_r'] - plain_row['disjoint_half_stim_r']:+.4f})"
        )
    # end for mapping
    print(
        "  A gain under stim_r that disappears under the cross-half score is "
        "noise the observed\n  bins share with the predicted window, not image "
        "information."
    )
    print(f"\nsaved to {output_dir}")
# EOF


if __name__ == "__main__":
    main()
# EOC
