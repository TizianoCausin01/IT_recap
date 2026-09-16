"""
Shared scaffolding for the TVSD monkey F decoder experiments.

Every experiment script reuses the same protocol: cached DINOv3 features and
baseline-corrected MUA targets, one seeded split of the 22,248 unique-image
presentations, train-only per-channel standardization, optional robust neural
preprocessing, and the untouched 100 repeated test images. Keeping that here
means a new experiment differs from the others only in its model and objective,
never in its data handling.
"""

import copy
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import RidgeCV
from torch.utils.data import DataLoader, TensorDataset

from project_specific_utils.dataloader import (
    apply_neural_preprocessing,
    fit_neural_preprocessing,
)
from IT_recap.neural_prediction_training import (
    aggregate_attention_by_layer,
    neural_activity_timebin_mse_loss,
    training_step,
)
from model_classes.temporal_models import (
    BaselineModel,
    NoiseLayerBaselineModel,
    TemporalNoiseBaselineModel,
)
from IT_recap.tvsd import (
    TVSDTrialDataset,
    compute_tvsd_channel_standardization,
    load_tvsd_metadata,
)
from IT_recap.neural_prediction_training import (
    aggregate_trials_by_image,
    split_half_reliability,
    stimulus_correlation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

# The official TVSD repeated-test pool: 100 images shown 30 times each.
N_TEST_IMAGES = 100


"""
load_project_paths
Resolve the active environment paths and expose useful_stuff on sys.path.

INPUT:
    - cfg: Cfg -> optional environment override

OUTPUT:
    - paths: dict -> active project paths
"""
def load_project_paths(cfg):
    env = cfg.env or os.getenv("MY_ENV", "tiziano_mac_mini")
    with open(PROJECT_ROOT / "config.yaml", "r") as config_file:
        project_config = yaml.safe_load(config_file)
    # end with project configuration
    paths = project_config[env]["paths"]
    sys.path.insert(0, str(Path(paths["useful_stuff_path"]).resolve()))
    return paths
# EOF


"""
resolve_device
Select the best available torch device unless one was requested explicitly.

INPUT:
    - requested: str -> "auto", "cpu", "mps", or "cuda"

OUTPUT:
    - device: torch.device -> device used for decoder training
"""
def resolve_device(requested):
    if requested != "auto":
        return torch.device(requested)
    # end if the device was requested explicitly
    if torch.cuda.is_available():
        return torch.device("cuda")
    # end if a CUDA device is present
    if torch.backends.mps.is_available():
        return torch.device("mps")
    # end if Apple acceleration is present
    return torch.device("cpu")
# EOF


"""
resolve_cache_paths
Name the three files an experiment reads, without requiring them to exist.

The target cache's file name encodes the response window, so a different
time_start_ms or target_fs names a different file rather than slicing the
existing one. Sharing this construction keeps a notebook's "is it there?" check
and load_cached_data's "it is missing" error talking about the same path.

INPUT:
    - cfg: Cfg -> cache names and target-window settings
    - paths: dict -> active project paths

OUTPUT:
    - cache_paths: dict -> "targets" and "mua" paths, plus "features" for an
      experiment whose configuration names a pooled feature archive
"""
def resolve_cache_paths(cfg, paths):
    data_root = Path(paths["data_path"])
    cache_paths = {
        "targets": data_root / "data" / (
            f"tvsd_monkeyF_{cfg.area}_{cfg.time_start_ms:g}-"
            f"{cfg.time_end_ms:g}ms_{cfg.target_fs}Hz_baseline_corrected.npy"
        ),
        "mua": data_root / "data" / cfg.mua_file_name,
    }
    # An experiment reading a cached convolutional map names no pooled archive,
    # and then there is no feature path to resolve.
    archive_name = getattr(cfg, "feature_archive_name", None)
    if archive_name:
        cache_paths["features"] = data_root / "models" / archive_name
    # end if this experiment reads a pooled feature archive
    return cache_paths
# EOF


"""
load_cached_targets
Load the prepared neural targets and the presentation metadata.

The ANN features are loaded separately, so an experiment whose input is not a
pooled feature archive -- a cached convolutional map, say -- reads the response
side of the protocol through this function without paying for an archive it
never uses.

INPUT:
    - cfg: Cfg -> cache names and target-window settings
    - paths: dict -> active project paths

OUTPUT:
    - targets: np.memmap -> [presentations, time, channels] baseline-corrected MUA
    - allmat: np.ndarray -> ALLMAT metadata rows [presentations, 6]
"""
def load_cached_targets(cfg, paths):
    cache_paths = resolve_cache_paths(cfg, paths)
    for required_path in (cache_paths["targets"], cache_paths["mua"]):
        if not required_path.is_file():
            raise FileNotFoundError(
                f"Missing cache {required_path}. Run the marimo notebook's "
                "preparation cells first."
            )
        # end if a required cache is absent
    # end for required cache

    targets = np.load(cache_paths["targets"], mmap_mode="r")
    # Only ALLMAT is read; the 58 GB MUA array itself is never touched here.
    allmat, _ = load_tvsd_metadata(cache_paths["mua"])
    return targets, allmat
# EOF


"""
load_cached_data
Load the prepared neural targets, DINO features, and presentation metadata.

INPUT:
    - cfg: Cfg -> cache names and target-window settings
    - paths: dict -> active project paths

OUTPUT:
    - targets: np.memmap -> [presentations, time, channels] baseline-corrected MUA
    - train_features: np.ndarray -> [22248, layers, embedding]
    - test_features: np.ndarray -> [100, layers, embedding]
    - allmat: np.ndarray -> ALLMAT metadata rows [presentations, 6]
"""
def load_cached_data(cfg, paths):
    targets, allmat = load_cached_targets(cfg, paths)
    cache_paths = resolve_cache_paths(cfg, paths)
    if "features" not in cache_paths:
        raise ValueError(
            "cfg.feature_archive_name must name a pooled feature archive."
        )
    # end if the configuration names no feature archive
    feature_archive_path = cache_paths["features"]
    if not feature_archive_path.is_file():
        raise FileNotFoundError(
            f"Missing cache {feature_archive_path}. Run the marimo notebook's "
            "preparation cells first."
        )
    # end if the feature archive is absent

    with np.load(feature_archive_path) as feature_archive:
        train_features = feature_archive["train_features"].astype(
            np.float32, copy=False
        )
        test_features = feature_archive["test_features"].astype(
            np.float32, copy=False
        )
        saved_layer_names = feature_archive["layer_names"].tolist()
    # end with DINO feature archive
    # The archive may hold more depths than an experiment asks for, so select
    # cfg.layer_names out of it in the requested order. An exact match is the
    # identity operation, which keeps earlier experiments byte-identical.
    missing_layers = [
        layer_name
        for layer_name in cfg.layer_names
        if layer_name not in saved_layer_names
    ]
    if missing_layers:
        raise ValueError(
            f"Cached layers {saved_layer_names} do not contain {missing_layers}."
        )
    # end if a requested layer is absent from the archive
    layer_positions = [
        saved_layer_names.index(layer_name) for layer_name in cfg.layer_names
    ]
    train_features = train_features[:, layer_positions]
    test_features = test_features[:, layer_positions]
    return targets, train_features, test_features, allmat
# EOF


"""
build_datasets
Split the unique-image presentations and wrap every subset in a TVSD dataset.

The 100 repeated test images stay untouched, exactly as in the notebook, so
this script's numbers are directly comparable to the existing baseline results.

INPUT:
    - cfg: Cfg -> split fractions and seed
    - targets: np.ndarray -> [presentations, time, channels] target cache
    - train_features: np.ndarray -> ordered train-stimulus features
    - test_features: np.ndarray -> ordered repeated-test-stimulus features
    - allmat: np.ndarray -> ALLMAT metadata rows

OUTPUT:
    - datasets: dict -> train, validation, and test TVSDTrialDataset objects
    - indices: dict -> presentation indices behind each subset
    - standardization: tuple -> train-only channel mean and scale
"""
def build_datasets(cfg, targets, train_features, test_features, allmat):
    official_training_indices = np.flatnonzero(allmat[:, 1] > 0)
    test_trial_indices = np.flatnonzero(allmat[:, 2] > 0)

    split_rng = np.random.default_rng(cfg.random_seed)
    shuffled_indices = split_rng.permutation(official_training_indices)
    n_validation = max(
        1, round(len(official_training_indices) * cfg.validation_fraction)
    )
    validation_trial_indices = shuffled_indices[:n_validation]
    training_trial_indices = shuffled_indices[n_validation:]
    if cfg.smoke_test:
        # A short run only needs enough presentations to exercise every path.
        training_trial_indices = training_trial_indices[:1024]
        validation_trial_indices = validation_trial_indices[:256]
    # end if a smoke run was requested

    channel_mean, channel_scale = compute_tvsd_channel_standardization(
        targets, training_trial_indices
    )
    dataset_inputs = {
        "train_inputs": train_features,
        "test_inputs": test_features,
        "targets": targets,
        "metadata": allmat,
        "input_mode": "activations",
        "channel_mean": channel_mean,
        "channel_scale": channel_scale,
    }
    indices = {
        "train": training_trial_indices,
        "validation": validation_trial_indices,
        "test": test_trial_indices,
    }
    datasets = {
        subset_name: TVSDTrialDataset(
            **dataset_inputs, trial_indices=subset_indices
        )
        for subset_name, subset_indices in indices.items()
    }
    return datasets, indices, (channel_mean, channel_scale)
# EOF


"""
build_loaders
Wrap the three datasets in loaders; only the training subset is shuffled.

INPUT:
    - cfg: Cfg -> batch size, worker count, and seed
    - datasets: dict -> train, validation, and test datasets

OUTPUT:
    - loaders: dict -> matching DataLoader objects
"""
def build_loaders(cfg, datasets):
    loader_generator = torch.Generator().manual_seed(cfg.random_seed)
    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            generator=loader_generator,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        ),
    }
# EOF


"""
standardize_targets
Apply the train-only per-channel standardization used by TVSDTrialDataset.

INPUT:
    - raw_targets: np.ndarray -> [presentations, time, channels] raw MUA
    - channel_mean: np.ndarray -> [channels] train mean
    - channel_scale: np.ndarray -> [channels] train scale

OUTPUT:
    - standardized: np.ndarray -> targets on the model's output scale
"""
def standardize_targets(raw_targets, channel_mean, channel_scale):
    raw_targets = np.asarray(raw_targets, dtype=np.float32)
    return (
        raw_targets - channel_mean[None, None, :]
    ) / channel_scale[None, None, :]
# EOF


"""
fit_ridge_map
Fit the notebook's RidgeCV map from concatenated ANN layers to the flattened
[time x channel] response, and return both validation and test predictions.

Feature standardization is estimated on the fit split alone, so no validation
or test statistic leaks into the map.

INPUT:
    - cfg: Cfg -> unused beyond documenting the shared protocol
    - train_features: np.ndarray -> ordered train-stimulus features
    - test_features: np.ndarray -> ordered repeated-test-stimulus features
    - allmat: np.ndarray -> ALLMAT metadata rows
    - indices: dict -> presentation indices per subset
    - fit_targets: np.ndarray -> standardized fit targets
    - validation_targets: np.ndarray -> standardized validation targets

OUTPUT:
    - ridge_fit: dict -> trial_predictions, validation_predictions,
      validation_mse, alpha, and n_coefficients
"""
def fit_ridge_map(
    cfg,
    train_features,
    test_features,
    allmat,
    indices,
    fit_targets,
    validation_targets,
):
    fit_image_ids = allmat[indices["train"], 1] - 1
    validation_image_ids = allmat[indices["validation"], 1] - 1
    flat_fit_features = train_features[fit_image_ids].reshape(
        len(fit_image_ids), -1
    )
    flat_validation_features = train_features[validation_image_ids].reshape(
        len(validation_image_ids), -1
    )
    flat_test_features = test_features.reshape(len(test_features), -1)

    feature_mean = flat_fit_features.mean(axis=0, keepdims=True)
    feature_scale = flat_fit_features.std(axis=0, keepdims=True) + 1e-6
    flat_fit_features = (flat_fit_features - feature_mean) / feature_scale
    flat_validation_features = (
        flat_validation_features - feature_mean
    ) / feature_scale
    flat_test_features = (flat_test_features - feature_mean) / feature_scale

    n_time, n_channels = fit_targets.shape[1], fit_targets.shape[2]
    ridge_model = RidgeCV(alphas=np.logspace(1.0, 7.0, 13))
    ridge_model.fit(
        flat_fit_features, fit_targets.reshape(len(fit_image_ids), -1)
    )
    validation_predictions = ridge_model.predict(flat_validation_features)
    validation_mse = float(
        np.mean(
            (
                validation_predictions
                - validation_targets.reshape(len(validation_image_ids), -1)
            )
            ** 2
        )
    )
    image_predictions = ridge_model.predict(flat_test_features).reshape(
        len(test_features), n_time, n_channels
    )

    # Broadcast the deterministic per-image map back to trial level so that
    # ridge and the decoders go through the identical aggregation code path.
    test_image_ids = allmat[indices["test"], 2] - 1
    return {
        "trial_predictions": image_predictions[test_image_ids],
        "validation_predictions": validation_predictions.reshape(
            len(validation_image_ids), n_time, n_channels
        ),
        "validation_mse": validation_mse,
        "alpha": float(ridge_model.alpha_),
        "n_coefficients": int(
            ridge_model.coef_.size + np.size(ridge_model.intercept_)
        ),
    }
# EOF


"""
fit_ridge_reference
Legacy four-value view of fit_ridge_map, kept for the earlier TVSD experiment
scripts that unpack a tuple.

INPUT:
    - see fit_ridge_map

OUTPUT:
    - trial_predictions: np.ndarray -> [test presentations, time, channels]
    - validation_mse: float -> standardized validation MSE
    - alpha: float -> selected ridge penalty
    - n_coefficients: int -> fitted ridge coefficients plus intercepts
"""
def fit_ridge_reference(
    cfg,
    train_features,
    test_features,
    allmat,
    indices,
    fit_targets,
    validation_targets,
):
    ridge_fit = fit_ridge_map(
        cfg,
        train_features,
        test_features,
        allmat,
        indices,
        fit_targets,
        validation_targets,
    )
    return (
        ridge_fit["trial_predictions"],
        ridge_fit["validation_mse"],
        ridge_fit["alpha"],
        ridge_fit["n_coefficients"],
    )
# EOF


"""
fit_ridge_from_presentations
Fit RidgeCV on per-presentation design matrices rather than per-stimulus ones.

fit_ridge_map maps one feature vector per *stimulus*, which is right when the
input is the image alone: the 30 repetitions of a test image then share one
prediction. A decoder that also reads the presentation's own early response has
a different input on every repetition, so its linear reference has to be fitted
and evaluated per presentation. With an image-only design matrix the two agree,
which is what makes this the matched reference for both.

Standardization uses the fit split alone, so nothing leaks from validation or
test into the map.

INPUT:
    - subset_inputs: dict -> split name to [presentations, features] design matrix
    - subset_targets: dict -> split name to [presentations, time, channels] targets

OUTPUT:
    - ridge_fit: dict -> trial_predictions, validation_predictions,
      validation_mse, alpha, and n_coefficients
"""
def fit_ridge_from_presentations(subset_inputs, subset_targets):
    fit_inputs = np.asarray(subset_inputs["train"], dtype=np.float32)
    feature_mean = fit_inputs.mean(axis=0, keepdims=True)
    feature_scale = fit_inputs.std(axis=0, keepdims=True) + 1e-6
    standardized = {
        split_name: (
            np.asarray(split_inputs, dtype=np.float32) - feature_mean
        ) / feature_scale
        for split_name, split_inputs in subset_inputs.items()
    }

    n_time, n_channels = subset_targets["train"].shape[1:]
    ridge_model = RidgeCV(alphas=np.logspace(1.0, 7.0, 13))
    ridge_model.fit(
        standardized["train"],
        subset_targets["train"].reshape(len(standardized["train"]), -1),
    )
    predictions = {
        split_name: ridge_model.predict(split_inputs).reshape(
            len(split_inputs), n_time, n_channels
        )
        for split_name, split_inputs in standardized.items()
        if split_name in ("validation", "test")
    }
    validation_mse = float(
        np.mean(
            (predictions["validation"] - subset_targets["validation"]) ** 2
        )
    )
    return {
        "trial_predictions": predictions["test"],
        "validation_predictions": predictions["validation"],
        "validation_mse": validation_mse,
        "alpha": float(ridge_model.alpha_),
        "n_coefficients": int(
            ridge_model.coef_.size + np.size(ridge_model.intercept_)
        ),
    }
# EOF


"""
predict_test_trials
Run the selected decoder over every repeated-test presentation.

The noise-layer variant stays stochastic in eval mode, so each of an image's 30
presentations produces its own prediction. That is exactly what the minimum
aggregation later consumes.

INPUT:
    - model: BaselineModel -> trained decoder
    - test_loader: DataLoader -> repeated-test presentations
    - device: torch.device -> compute device

OUTPUT:
    - trial_predictions: np.ndarray -> [presentations, time, channels]
    - trial_targets: np.ndarray -> [presentations, time, channels]
    - mean_attention: np.ndarray -> [time, attended items] mean layer attention
"""
def predict_test_trials(model, test_loader, device):
    prediction_batches, target_batches, attention_batches = [], [], []
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            predictions, attention = model(
                inputs.to(device), use_precomputed_features=True
            )
            prediction_batches.append(predictions.cpu().numpy())
            target_batches.append(targets.numpy())
            attention_batches.append(
                aggregate_attention_by_layer(attention).cpu()
            )
        # end for repeated-test batch
    # end with no gradient tracking

    mean_attention = torch.cat(attention_batches, dim=0).mean(dim=0).numpy()
    return (
        np.concatenate(prediction_batches, axis=0),
        np.concatenate(target_batches, axis=0),
        mean_attention,
    )
# EOF


"""
score_predictions
Score trial-level predictions after collapsing each image's repetitions.

INPUT:
    - trial_predictions: np.ndarray -> [presentations, time, channels]
    - trial_targets: np.ndarray -> [presentations, time, channels]
    - image_ids: np.ndarray -> zero-based test-image id per presentation
    - reducer: str -> "mean" or "min" over the 30 repetitions
    - fit_target_mean: np.ndarray -> [time, channels] fit-split mean predictor
    - ceiling: np.ndarray -> [time, channels] stim_r reference values
    - response_slice: slice -> time bins counted as driven response

OUTPUT:
    - metrics: dict -> scalar scores for this variant and aggregation
    - channel_time_correlations: np.ndarray -> [time, channels] stim_r
"""
def score_predictions(
    trial_predictions,
    trial_targets,
    image_ids,
    reducer,
    fit_target_mean,
    ceiling,
    response_slice,
):
    image_predictions = aggregate_trials_by_image(
        trial_predictions, image_ids, N_TEST_IMAGES, reducer=reducer
    )
    image_targets = aggregate_trials_by_image(
        trial_targets, image_ids, N_TEST_IMAGES, reducer=reducer
    )

    test_mse = float(np.mean((image_predictions - image_targets) ** 2))
    predict_mean_mse = float(
        np.mean((fit_target_mean[None] - image_targets) ** 2)
    )
    channel_time_correlations = stimulus_correlation(
        image_predictions, image_targets
    )
    response_correlations = channel_time_correlations[response_slice]
    response_ceiling = ceiling[response_slice]

    # Only the repetition mean has a real noise ceiling: Spearman-Brown
    # extrapolation assumes averaging. The split-half reliability of a minimum
    # is near zero, so dividing by it would produce meaningless ratios above
    # one. For "min" the reliability is reported as context instead.
    fraction_of_ceiling = None
    if reducer == "mean":
        # Ratio of the means, not the mean of per-cell ratios. Roughly 5% of
        # (time, site) cells have reliability near zero, and dividing by those
        # inflates a per-cell average well past one.
        fraction_of_ceiling = round(
            float(np.nanmean(response_correlations) / np.nanmean(response_ceiling)),
            3,
        )
    # end if the aggregation has a well-defined ceiling

    metrics = {
        "aggregation": reducer,
        "test_mse": round(test_mse, 5),
        "variance_explained_vs_mean": round(1 - test_mse / predict_mean_mse, 4),
        "mean_stim_r_response": round(
            float(np.nanmean(response_correlations)), 4
        ),
        "median_stim_r_response": round(
            float(np.nanmedian(response_correlations)), 4
        ),
        "fraction_of_ceiling": fraction_of_ceiling,
        "target_reliability": round(float(np.nanmean(response_ceiling)), 4),
    }
    return metrics, channel_time_correlations
# EOF
"""
select_window_bin_indices
Map a response window in milliseconds onto indices of the target cache's bins.

Window edges are rounded to the nearest bin boundary, so the number of selected
bins matches the requested width whenever that width is a multiple of the bin
size. The covered range is returned as well, because a request whose edges do
not fall on bin boundaries is served by a slightly shifted window.

INPUT:
    - n_timepoints: int -> number of bins in the target cache
    - target_fs: int -> target sampling rate in Hz
    - cache_start_ms: float -> time of the cache's first bin edge
    - window_start_ms: float -> inclusive window start
    - window_end_ms: float -> exclusive window end

OUTPUT:
    - bin_indices: np.ndarray -> selected zero-based bin indices
    - covered_ms: tuple[float, float] -> the window actually covered
"""
def select_window_bin_indices(
    n_timepoints,
    target_fs,
    cache_start_ms,
    window_start_ms,
    window_end_ms,
):
    if window_end_ms <= window_start_ms:
        raise ValueError("window_end_ms must be greater than window_start_ms.")
    # end if the requested window is empty

    bin_width_ms = 1000.0 / target_fs
    # np.floor(x + 0.5) rounds halves away from zero, unlike Python's round().
    first_bin = int(
        np.floor((window_start_ms - cache_start_ms) / bin_width_ms + 0.5)
    )
    last_bin = int(
        np.floor((window_end_ms - cache_start_ms) / bin_width_ms + 0.5)
    )
    if first_bin < 0 or last_bin > n_timepoints or last_bin <= first_bin:
        raise ValueError(
            f"Window {window_start_ms:g}-{window_end_ms:g} ms maps to bins "
            f"[{first_bin}, {last_bin}), outside the cached "
            f"{n_timepoints} bins."
        )
    # end if the window falls outside the cached response

    bin_indices = np.arange(first_bin, last_bin)
    covered_ms = (
        cache_start_ms + first_bin * bin_width_ms,
        cache_start_ms + last_bin * bin_width_ms,
    )
    return bin_indices, covered_ms
# EOF


"""
average_targets_over_window
Collapse the selected response bins into one population vector per presentation.

Streaming over the memory-mapped cache keeps peak memory at one chunk, and the
retained singleton time axis lets the window-averaged targets flow through the
same datasets and scoring code as the time-resolved ones.

INPUT:
    - targets: np.ndarray -> [presentations, time, channels] target cache
    - bin_indices: array-like -> response bins to average
    - chunk_size: int -> presentations read per pass

OUTPUT:
    - window_targets: np.ndarray -> [presentations, 1, channels] window means
"""
def average_targets_over_window(targets, bin_indices, chunk_size=2048):
    if targets.ndim != 3:
        raise ValueError("targets must have shape [presentations, time, channels].")
    # end if target axes are invalid
    bin_indices = np.asarray(bin_indices, dtype=int)
    if bin_indices.ndim != 1 or bin_indices.size == 0:
        raise ValueError("bin_indices must be a non-empty vector.")
    # end if no response bins were selected
    if bin_indices.min() < 0 or bin_indices.max() >= targets.shape[1]:
        raise IndexError("bin_indices exceed the cached time axis.")
    # end if a requested bin is out of range
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    # end if the streaming chunk is invalid

    # The whole window is one group, which is the single-vector special case
    # of the general re-binning below.
    return average_targets_over_bin_groups(
        targets, bin_indices, len(np.asarray(bin_indices)), chunk_size
    )
# EOF


"""
average_targets_over_bin_groups
Re-bin the cached response by averaging consecutive groups of cached bins.

The cache is stored at the preprocessing rate (10 ms bins at 100 Hz), while an
experiment usually wants coarser bins: a group size of two turns them into the
20 ms bins used by the architecture search. Streaming over the memory-mapped
cache keeps peak memory at one chunk.

INPUT:
    - targets: np.ndarray -> [presentations, time, channels] target cache
    - bin_indices: array-like -> contiguous cached bins to re-bin, in order
    - group_size: int -> cached bins averaged into one output bin
    - chunk_size: int -> presentations read per pass

OUTPUT:
    - grouped_targets: np.ndarray -> [presentations, groups, channels] means
"""
def average_targets_over_bin_groups(
    targets, bin_indices, group_size, chunk_size=2048
):
    if targets.ndim != 3:
        raise ValueError("targets must have shape [presentations, time, channels].")
    # end if target axes are invalid
    bin_indices = np.asarray(bin_indices, dtype=int)
    if bin_indices.ndim != 1 or bin_indices.size == 0:
        raise ValueError("bin_indices must be a non-empty vector.")
    # end if no response bins were selected
    if bin_indices.min() < 0 or bin_indices.max() >= targets.shape[1]:
        raise IndexError("bin_indices exceed the cached time axis.")
    # end if a requested bin is out of range
    if group_size <= 0 or len(bin_indices) % group_size != 0:
        raise ValueError(
            f"{len(bin_indices)} selected bins do not form complete groups of "
            f"{group_size}."
        )
    # end if the selection leaves a partial output bin
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    # end if the streaming chunk is invalid

    n_presentations, _, n_channels = targets.shape
    n_groups = len(bin_indices) // group_size
    grouped_targets = np.empty(
        (n_presentations, n_groups, n_channels), dtype=np.float32
    )
    for chunk_start in range(0, n_presentations, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_presentations)
        target_chunk = np.asarray(
            targets[chunk_start:chunk_end], dtype=np.float32
        )[:, bin_indices, :]
        # [chunk, groups, group_size, channels] -> mean over the group axis.
        grouped_targets[chunk_start:chunk_end] = target_chunk.reshape(
            chunk_end - chunk_start, n_groups, group_size, n_channels
        ).mean(axis=2)
    # end for presentation chunk
    return grouped_targets
# EOF


"""
evaluate_cached_feature_decoder
Run a cached-feature decoder over one loader and collect its predictions.

A batch is read as (*inputs, target), so a decoder taking a second input -- the
observed early response, say -- goes through this function unchanged; with the
usual two-tensor batches the call is exactly model(features).

INPUT:
    - model: nn.Module -> decoder returning (predictions, diagnostics)
    - loader: DataLoader -> evaluation batches of (*inputs, targets)
    - device: torch.device -> compute device

OUTPUT:
    - predictions: np.ndarray -> [presentations, time, channels]
    - targets: np.ndarray -> [presentations, time, channels]
"""
def evaluate_cached_feature_decoder(model, loader, device):
    model.eval()
    prediction_batches, target_batches = [], []
    with torch.no_grad():
        for batch in loader:
            *inputs, targets = batch
            predictions, _ = model(*[tensor.to(device) for tensor in inputs])
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
collect_l2_parameters
Select the weights an explicit L2 penalty applies to.

AdamW's weight decay is *decoupled*: it shrinks weights directly, outside the
loss, and is never rescaled by Adam's second-moment estimate. Ridge instead
minimizes ||y - Xw||^2 + alpha ||w||^2, so its penalty reaches the optimizer as
a gradient like any other term. This selects what that term should cover:

    - "readout": the decoder's ridge-analogous linear maps, which each model
      names through ridge_like_parameters -- the maps in from the features and
      out to the population, with the recurrence or transition left alone;
    - "all": every weight matrix, biases and normalization scales excluded, the
      same way ridge leaves its intercept unpenalized.

INPUT:
    - model: nn.Module -> decoder being optimized
    - scope: str -> "none", "readout", or "all"

OUTPUT:
    - parameters: list[torch.nn.Parameter] -> weights the penalty covers
"""
def collect_l2_parameters(model, scope):
    if scope == "none":
        return []
    # end if no penalty was requested
    if scope == "all":
        # A 1-D parameter is a bias, a norm scale, or a learned embedding, none
        # of which is part of the linear map ridge penalizes.
        return [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.ndim > 1
        ]
    # end if the penalty covers every weight matrix
    if scope != "readout":
        raise ValueError(
            f"Unknown l2_scope {scope!r}; choose none, readout, or all."
        )
    # end if the requested scope is invalid
    if not hasattr(model, "ridge_like_parameters"):
        raise AttributeError(
            f"{type(model).__name__} does not define ridge_like_parameters, "
            'so the "readout" scope is undefined for it.'
        )
    # end if the model names no ridge-analogous maps
    return list(model.ridge_like_parameters())
# EOF


"""
ridge_equivalent_l2_lambda
The penalty weight that matches a ridge alpha under a mean-squared-error loss.

RidgeCV minimizes a *summed* squared error plus alpha ||w||^2, while the
decoders minimize the error *averaged* over presentations, bins and channels.
Dividing the ridge objective by that count puts the two on one scale, so this
is the value to centre an L2 sweep on rather than guessing a grid.

INPUT:
    - alpha: float -> the fitted RidgeCV penalty
    - n_presentations: int -> fit-split presentations
    - n_timepoints: int -> target bins
    - n_neurons: int -> target channels

OUTPUT:
    - l2_lambda: float -> equivalent penalty weight for a mean MSE loss
"""
def ridge_equivalent_l2_lambda(alpha, n_presentations, n_timepoints, n_neurons):
    return float(alpha) / (n_presentations * n_timepoints * n_neurons)
# EOF


"""
train_cached_feature_decoder
Optimize a decoder and restore its best validation checkpoint.

Selection follows cfg.selection_metric: validation MSE rewards the shrinkage
that ridge already applies, so "stim_r" -- the image selectivity these
experiments report -- is the usual choice.

The optimized loss is plain MSE unless a cost function is passed in. Whatever
is optimized, the reported train and validation MSE stay the *unweighted* data
term, so runs under different objectives remain comparable epoch by epoch; the
optimized quantity is reported next to them as the loss.

INPUT:
    - model: nn.Module -> decoder to optimize, already on the device
    - loaders: dict -> "train" and "validation" DataLoader objects
    - cfg: Cfg -> epochs, patience, optimizer, and selection settings
    - device: torch.device -> compute device
    - verbose: bool -> print one line per epoch
    - cost_function: callable | None -> (predictions, targets) -> scalar loss;
      None optimizes plain MSE

OUTPUT:
    - history: list[dict] -> per-epoch train and validation MSE, the optimized
      loss, and validation stim_r
    - best_epoch: int -> selected checkpoint epoch
    - best_validation_mse: float -> validation MSE at that checkpoint
    - best_validation_stim_r: float -> validation stim_r at that checkpoint
"""
def train_cached_feature_decoder(
    model, loaders, cfg, device, verbose=True, cost_function=None
):
    if cfg.selection_metric not in {"mse", "stim_r"}:
        raise ValueError("selection_metric must be either 'mse' or 'stim_r'.")
    # end if the selection criterion is unsupported

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(3, cfg.patience // 3),
        min_lr=cfg.learning_rate / 100.0,
    )
    if cost_function is None:
        cost_function = torch.nn.MSELoss()
    # end if no explicit objective was requested

    # An explicit L2 term reproduces ridge's objective; cfg may omit both
    # fields, in which case nothing is penalized and the loss is plain MSE.
    l2_lambda = float(getattr(cfg, "l2_lambda", 0.0))
    if l2_lambda < 0.0:
        raise ValueError("l2_lambda must be non-negative.")
    # end if the penalty weight is invalid
    penalized_parameters = (
        collect_l2_parameters(model, getattr(cfg, "l2_scope", "readout"))
        if l2_lambda > 0.0
        else []
    )

    best_state = copy.deepcopy(model.state_dict())
    # Both criteria are minimized once stim_r is negated.
    best_selection_score = np.inf
    best_validation_mse, best_validation_stim_r = np.inf, -np.inf
    best_epoch, epochs_without_improvement = 0, 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_squared_error, train_loss_sum, train_values = 0.0, 0.0, 0
        for batch in loaders["train"]:
            # Same convention as the evaluation loop: every tensor but the
            # last is a model input.
            *inputs, targets = batch
            inputs = [tensor.to(device) for tensor in inputs]
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions, _ = model(*inputs)
            loss = cost_function(predictions, targets)
            # The unweighted MSE is accounted separately from the optimized
            # loss, so a weighted or correlation objective still reports the
            # error scale every other experiment prints.
            batch_mse = torch.nn.functional.mse_loss(predictions, targets).detach()
            if penalized_parameters:
                # Reported train_mse stays the data term alone, so runs with
                # and without a penalty remain comparable epoch by epoch.
                penalty = sum(
                    parameter.square().sum() for parameter in penalized_parameters
                )
                (loss + l2_lambda * penalty).backward()
            else:
                loss.backward()
            # end if an explicit L2 penalty is active
            if cfg.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.gradient_clip
                )
            # end if gradient clipping is enabled
            optimizer.step()
            train_squared_error += batch_mse.item() * targets.numel()
            train_loss_sum += loss.item() * targets.numel()
            train_values += targets.numel()
        # end for training batch

        validation_predictions, validation_targets = (
            evaluate_cached_feature_decoder(model, loaders["validation"], device)
        )
        train_mse = train_squared_error / train_values
        train_loss = train_loss_sum / train_values
        validation_mse = float(
            np.mean((validation_predictions - validation_targets) ** 2)
        )
        # The optimized objective on the validation split, which for a weighted
        # loss is the quantity the training loss is directly comparable to.
        validation_loss = float(
            cost_function(
                torch.from_numpy(validation_predictions),
                torch.from_numpy(validation_targets),
            )
        )
        validation_stim_r = float(
            np.nanmean(
                stimulus_correlation(validation_predictions, validation_targets)
            )
        )
        selection_score = (
            validation_mse
            if cfg.selection_metric == "mse"
            else -validation_stim_r
        )
        # The schedule follows whichever criterion is being selected on.
        scheduler.step(selection_score)
        history.append(
            {
                "epoch": epoch,
                "train_mse": train_mse,
                "train_loss": train_loss,
                "validation_mse": validation_mse,
                "validation_loss": validation_loss,
                "validation_stim_r": validation_stim_r,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if verbose:
            print(
                f"    epoch {epoch:03d}/{cfg.epochs:03d} | train {train_mse:.6f}"
                f" | validation MSE {validation_mse:.6f} | validation stim_r "
                f"{validation_stim_r:.4f}"
            )
        # end if per-epoch reporting is requested

        if selection_score < best_selection_score - 1e-8:
            best_selection_score = selection_score
            best_validation_mse = validation_mse
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
            break
        # end if early-stopping patience is exhausted
    # end for optimization epoch

    model.load_state_dict(best_state)
    return (
        history,
        best_epoch,
        float(best_validation_mse),
        float(best_validation_stim_r),
    )
# EOF


"""
gather_presentation_features
Expand ordered per-stimulus features to one row per presentation.

INPUT:
    - train_features: np.ndarray -> [22248, layers, embedding] train stimuli
    - test_features: np.ndarray -> [100, layers, embedding] repeated-test stimuli
    - allmat: np.ndarray -> ALLMAT metadata rows
    - trial_indices: np.ndarray -> presentation indices of one subset

OUTPUT:
    - features: np.ndarray -> [presentations, layers, embedding]
"""
def gather_presentation_features(
    train_features, test_features, allmat, trial_indices
):
    train_ids = allmat[trial_indices, 1]
    test_ids = allmat[trial_indices, 2]
    # ALLMAT guarantees exactly one of the two identifiers is non-zero.
    if np.all(train_ids > 0):
        return train_features[train_ids - 1]
    # end if the subset holds unique-image presentations
    if np.all(test_ids > 0):
        return test_features[test_ids - 1]
    # end if the subset holds repeated-test presentations
    raise ValueError("A subset mixes train and repeated-test presentations.")
# EOF


"""
make_tensor_loader
Wrap materialized features and standardized targets in a deterministic loader.

Targets are already in memory after window averaging, so a TensorDataset avoids
the per-sample Python indexing that the time-resolved experiments need.

INPUT:
    - features: np.ndarray -> [presentations, layers, embedding]
    - targets: np.ndarray -> [presentations, 1, channels] standardized targets
    - cfg: Cfg -> batch size, worker count, and seed
    - shuffle: bool -> whether to reshuffle every epoch

OUTPUT:
    - loader: DataLoader -> batches of (features, targets)
"""
def make_tensor_loader(features, targets, cfg, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(targets, dtype=np.float32)),
    )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        generator=torch.Generator().manual_seed(cfg.random_seed),
    )
# EOF


"""
make_multi_input_loader
Wrap several materialized input arrays and one target array in a loader.

make_tensor_loader is the single-input case of this; a decoder that also reads
the observed early response needs two input tensors per presentation, in the
(*inputs, target) order that the shared training and evaluation loops expect.

INPUT:
    - inputs: list[np.ndarray] -> per-presentation input arrays, in model order
    - targets: np.ndarray -> [presentations, time, channels] standardized targets
    - cfg: Cfg -> batch size, worker count, and seed
    - shuffle: bool -> whether to reshuffle every epoch

OUTPUT:
    - loader: DataLoader -> batches of (*inputs, targets)
"""
def make_multi_input_loader(inputs, targets, cfg, shuffle):
    tensors = [
        torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
        for array in [*inputs, targets]
    ]
    lengths = {len(tensor) for tensor in tensors}
    if len(lengths) != 1:
        raise ValueError(f"Inputs and targets disagree on length: {lengths}.")
    # end if the presentation axes disagree
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        generator=torch.Generator().manual_seed(cfg.random_seed),
    )
# EOF


"""
load_pooled_spatial_features
Read a cached convolutional feature map and pool it onto a coarse grid.

extract_tvsd_spatial_features.py caches the last conv map of a backbone --
AlexNet conv5 is 256 x 13 x 13 -- which is far too wide to hand to a ridge or
to a recurrent decoder. Adaptive average pooling to a pool_size x pool_size
grid keeps coarse retinotopy while producing the [items, cells, channels] shape
every cached-feature decoder in this project already consumes, with the grid
cells taking the place of the ANN depths. pool_size 1 is plain global pooling.

The cache is float16 and a couple of gigabytes, so it is read in chunks and
only the pooled result is materialized.

INPUT:
    - cfg: Cfg -> spatial_stem naming the cached maps
    - paths: dict -> active project paths
    - pool_size: int -> side of the pooled grid
    - chunk_size: int -> stimuli read per pass

OUTPUT:
    - pooled: dict -> "train" and "test" [stimuli, pool_size ** 2, channels]
    - feature_shape: tuple -> the cached [channels, height, width]
"""
def load_pooled_spatial_features(cfg, paths, pool_size, chunk_size=512):
    if pool_size <= 0:
        raise ValueError("pool_size must be positive.")
    # end if the requested grid is empty
    cache_dir = Path(paths["data_path"]) / "models"
    pooled, feature_shape = {}, None
    for split_name in ("train", "test"):
        cache_path = cache_dir / f"{cfg.spatial_stem}_{split_name}.npy"
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"Missing {cache_path}. Run extract_tvsd_spatial_features.py "
                "for this backbone first."
            )
        # end if the cache is absent
        feature_maps = np.load(cache_path, mmap_mode="r")
        feature_shape = tuple(feature_maps.shape[1:])
        n_stimuli, n_channels = len(feature_maps), feature_shape[0]
        split_features = np.empty(
            (n_stimuli, pool_size * pool_size, n_channels), dtype=np.float32
        )
        for chunk_start in range(0, n_stimuli, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_stimuli)
            feature_chunk = torch.from_numpy(
                np.asarray(
                    feature_maps[chunk_start:chunk_end], dtype=np.float32
                )
            )
            # [chunk, channels, pool, pool] -> [chunk, cells, channels], so the
            # grid cell is the axis a decoder treats as its layer axis.
            grid = torch.nn.functional.adaptive_avg_pool2d(
                feature_chunk, pool_size
            )
            split_features[chunk_start:chunk_end] = (
                grid.flatten(start_dim=2).transpose(1, 2).numpy()
            )
        # end for stimulus chunk
        pooled[split_name] = split_features
    # end for stimulus split
    return pooled, feature_shape
# EOF


"""
standardize_features_on_fit_split
Z-score per-presentation features per (cell, channel) on the fit split alone.

The pooled feature archives are already comparable across depths, but a raw
ReLU conv map is not: its channels differ in scale by orders of magnitude, and
a LayerNorm over the channel axis leaves the largest of them dominating. Ridge
standardizes its own design matrix internally, so applying the same statistics
here is what puts every compared method on identical inputs.

Only the "train" split defines the statistics, so nothing leaks from validation
or test into any method's input scale.

INPUT:
    - subset_features: dict -> split name to [presentations, cells, channels]

OUTPUT:
    - standardized: dict -> the same splits, on the fit split's scale
"""
def standardize_features_on_fit_split(subset_features):
    fit_features = subset_features["train"]
    feature_mean = fit_features.mean(axis=0, keepdims=True)
    feature_scale = fit_features.std(axis=0, keepdims=True) + 1e-6
    return {
        split_name: (split_features - feature_mean) / feature_scale
        for split_name, split_features in subset_features.items()
    }
# EOF


"""
preprocess_neural_targets_on_fit_split
Apply optional feature-wise centering and robust neuron scaling to TVSD targets.

The shared neural preprocessing utilities expect [neurons, time, samples], so
this function transposes the TVSD [samples, time, neurons] arrays before and
after transformation. Only the training subset defines the statistics.

INPUT:
    - subset_targets: dict -> split name to [samples, time, neurons] targets
    - cfg: Cfg -> optional centering, percentile, and clipping settings

OUTPUT:
    - processed_targets: dict -> every split on the training-fitted scale
    - preprocessing_stats: NeuralPreprocessingStats | None -> fitted statistics
"""
def preprocess_neural_targets_on_fit_split(subset_targets, cfg):
    center_features = getattr(cfg, "center_neural_features", False)
    robust_minmax_neurons = getattr(cfg, "robust_minmax_neurons", False)
    if not center_features and not robust_minmax_neurons:
        return subset_targets, None
    # end if optional target preprocessing is disabled

    training_activity = subset_targets["train"].transpose(2, 1, 0)
    preprocessing_stats = fit_neural_preprocessing(
        training_activity,
        center_features=center_features,
        robust_minmax_neurons=robust_minmax_neurons,
        robust_percentile_range=getattr(
            cfg,
            "robust_percentile_range",
            (1.0, 99.0),
        ),
        clip_robust_minmax=getattr(cfg, "clip_robust_minmax", True),
    )
    processed_targets = {
        split_name: apply_neural_preprocessing(
            split_targets.transpose(2, 1, 0),
            preprocessing_stats,
        ).transpose(2, 1, 0)
        for split_name, split_targets in subset_targets.items()
    }
    return processed_targets, preprocessing_stats
# EOF


"""
prepare_search_data
Load the caches once and build everything the search and ridge both consume.

Re-binning, the seeded split, train-only standardization, the noise ceiling,
and the loaders are all fixed here, so every architecture and every
hyperparameter draw sees byte-identical data.

INPUT:
    - cfg: Cfg -> window, split, and loader settings
    - paths: dict -> active project paths
    - device: torch.device -> compute device, reported alongside the split

OUTPUT:
    - data: dict -> loaders, shapes, scoring, the standardized subset targets
      and per-presentation features, the stimulus-ordered features ridge needs,
      and the window description
"""
def prepare_search_data(cfg, paths, device):
    targets, train_features, test_features, allmat = load_cached_data(cfg, paths)
    return prepare_timebin_data(
        cfg, targets, train_features, test_features, allmat, device
    )
# EOF


"""
prepare_timebin_data
Build the shared protocol from already-loaded targets and stimulus features.

This is prepare_search_data with the loading step lifted out, so an experiment
whose input is not a pooled feature archive -- a pooled convolutional map, say
-- gets byte-identical re-binning, splitting, standardization, noise ceiling
and loaders by passing its own stimulus-ordered features in.

INPUT:
    - cfg: Cfg -> window, split, and loader settings
    - targets: np.ndarray -> [presentations, time, channels] target cache
    - train_features: np.ndarray -> ordered train-stimulus features
    - test_features: np.ndarray -> ordered repeated-test-stimulus features
    - allmat: np.ndarray -> ALLMAT metadata rows
    - device: torch.device -> compute device, reported alongside the split

OUTPUT:
    - data: dict -> see prepare_search_data, plus the target cache itself and
      the train-only channel standardization the targets were put on
"""
def prepare_timebin_data(
    cfg, targets, train_features, test_features, allmat, device
):
    # Re-bin the cached 10 ms response into the requested output bins.
    bin_indices, covered_ms = select_window_bin_indices(
        targets.shape[1],
        cfg.target_fs,
        cfg.time_start_ms,
        cfg.window_start_ms,
        cfg.window_end_ms,
    )
    cached_bin_ms = 1000.0 / cfg.target_fs
    group_size = int(round(cfg.timebin_ms / cached_bin_ms))
    if group_size < 1:
        raise ValueError(
            f"timebin_ms must be at least the cached {cached_bin_ms:g} ms bin."
        )
    # end if the requested bins are finer than the cache
    timebin_targets = average_targets_over_bin_groups(
        targets, bin_indices, group_size
    )
    n_timepoints, n_neurons = timebin_targets.shape[1], timebin_targets.shape[2]
    bin_edges_ms = covered_ms[0] + cfg.timebin_ms * np.arange(n_timepoints + 1)

    _, indices, (channel_mean, channel_scale) = build_datasets(
        cfg, timebin_targets, train_features, test_features, allmat
    )
    print(
        f"device {device} | window {covered_ms[0]:g}-{covered_ms[1]:g} ms -> "
        f"{n_timepoints} bins of {cfg.timebin_ms:g} ms "
        f"({group_size} cached bins each) | {n_neurons} {cfg.area} sites"
    )
    # The input axis is read off the features themselves rather than off
    # cfg.layer_names, so pooled grid cells work here as well as ANN depths.
    n_input_groups = train_features.shape[1]
    print(
        f"fit {len(indices['train']):,} | validation "
        f"{len(indices['validation']):,} | test {len(indices['test']):,} "
        f"presentations | {n_input_groups} {cfg.model_name} feature groups"
    )

    # Materialize the standardized targets and per-presentation features once;
    # ridge and every decoder consume exactly the same arrays.
    subset_targets = {
        subset_name: standardize_targets(
            timebin_targets[subset_indices], channel_mean, channel_scale
        )
        for subset_name, subset_indices in indices.items()
    }
    subset_targets, neural_preprocessing_stats = (
        preprocess_neural_targets_on_fit_split(subset_targets, cfg)
    )
    subset_features = {
        subset_name: gather_presentation_features(
            train_features, test_features, allmat, subset_indices
        )
        for subset_name, subset_indices in indices.items()
    }
    fit_target_mean = subset_targets["train"].mean(axis=0)
    test_image_ids = allmat[indices["test"], 2] - 1

    # Every bin of the searched window is driven response, so nothing is cut.
    scoring = {
        "test_image_ids": test_image_ids,
        "fit_target_mean": fit_target_mean,
        "ceiling": split_half_reliability(
            subset_targets["test"],
            test_image_ids,
            N_TEST_IMAGES,
            reducer="mean",
            n_resamples=cfg.noise_ceiling_resamples,
            seed=cfg.random_seed,
        ),
        "response_slice": slice(0, n_timepoints),
    }
    loaders = {
        subset_name: make_tensor_loader(
            subset_features[subset_name],
            subset_targets[subset_name],
            cfg,
            shuffle=subset_name == "train",
        )
        for subset_name in ("train", "validation", "test")
    }
    shapes = {
        "n_layers": n_input_groups,
        "feature_dim": train_features.shape[2],
        "n_timepoints": n_timepoints,
        "n_neurons": n_neurons,
    }
    return {
        "loaders": loaders,
        "shapes": shapes,
        "scoring": scoring,
        "indices": indices,
        "targets": targets,
        "channel_standardization": (channel_mean, channel_scale),
        "neural_preprocessing_stats": neural_preprocessing_stats,
        "subset_targets": subset_targets,
        "subset_features": subset_features,
        "fit_target_mean": fit_target_mean,
        "train_features": train_features,
        "test_features": test_features,
        "allmat": allmat,
        "bin_indices": bin_indices,
        "covered_ms": covered_ms,
        "bin_edges_ms": bin_edges_ms,
    }
# EOF


"""
decompose_test_mse
Separate the test MSE into the part trial noise fixes and the part a model owns.

With 30 repetitions the image-averaged target still carries noise. The noise
ceiling gives the reliability of that average, so var(target) splits into a
predictable part and an irreducible one, and the distance between a model's MSE
and the irreducible floor is the only part any decoder can compete over. The
shrinkage term reports how much of the remaining error is a scale mismatch: the
MSE that would remain after rescaling every (bin, site) prediction optimally.

INPUT:
    - image_predictions: np.ndarray -> [images, time, neurons]
    - image_targets: np.ndarray -> [images, time, neurons]
    - ceiling: np.ndarray -> [time, neurons] stim_r noise ceiling

OUTPUT:
    - decomposition: dict -> variance split, MSE floor, slope, and rescaled MSE
"""
def decompose_test_mse(image_predictions, image_targets, ceiling):
    target_variance = image_targets.var(axis=0)
    # The ceiling is a correlation, so its square is the reliable fraction of
    # the image-averaged variance.
    reliable_fraction = np.clip(ceiling, 0.0, 1.0) ** 2
    noise_variance = target_variance * (1.0 - reliable_fraction)

    centered_predictions = image_predictions - image_predictions.mean(axis=0)
    centered_targets = image_targets - image_targets.mean(axis=0)
    prediction_variance = centered_predictions.var(axis=0)
    covariance = (centered_predictions * centered_targets).mean(axis=0)
    usable = prediction_variance > 1e-12
    optimal_scale = np.ones_like(prediction_variance)
    optimal_scale[usable] = covariance[usable] / prediction_variance[usable]
    rescaled = (
        centered_predictions * optimal_scale + image_targets.mean(axis=0)
    )
    return {
        "test_mse": round(float(np.mean((image_predictions - image_targets) ** 2)), 5),
        "irreducible_mse": round(float(np.mean(noise_variance)), 5),
        "target_variance": round(float(np.mean(target_variance)), 5),
        "predictable_variance": round(
            float(np.mean(target_variance * reliable_fraction)), 5
        ),
        "mse_above_floor": round(
            float(np.mean((image_predictions - image_targets) ** 2))
            - float(np.mean(noise_variance)),
            5,
        ),
        "mean_optimal_scale": round(float(np.mean(optimal_scale[usable])), 4),
        "mse_after_rescaling": round(
            float(np.mean((rescaled - image_targets) ** 2)), 5
        ),
    }
# EOF


"""
score_and_decompose
Score one set of trial predictions and add the MSE decomposition.

INPUT:
    - trial_predictions: np.ndarray -> [presentations, time, neurons]
    - subset_targets: np.ndarray -> [presentations, time, neurons]
    - scoring: dict -> image ids, fit-split mean, ceiling, response slice
    - name: str -> label written into the row
    - reducer: str -> "mean" or "min" over the repetitions of each image

OUTPUT:
    - row: dict -> metrics and MSE decomposition
    - site_correlations: np.ndarray -> [time, neurons] test stim_r
"""
def score_and_decompose(
    trial_predictions, subset_targets, scoring, name, reducer="mean"
):
    metrics, site_correlations = score_predictions(
        trial_predictions,
        subset_targets,
        scoring["test_image_ids"],
        reducer,
        scoring["fit_target_mean"],
        scoring["ceiling"],
        scoring["response_slice"],
    )
    image_predictions = aggregate_trials_by_image(
        trial_predictions, scoring["test_image_ids"], N_TEST_IMAGES, reducer=reducer
    )
    image_targets = aggregate_trials_by_image(
        subset_targets, scoring["test_image_ids"], N_TEST_IMAGES, reducer=reducer
    )
    row = {
        "model": name,
        **metrics,
        **decompose_test_mse(image_predictions, image_targets, scoring["ceiling"]),
    }
    return row, site_correlations
# EOF


"""
build_variant_model
Instantiate one decoder variant on the shared frozen encoder.

INPUT:
    - variant: str -> "baseline", "noise_layer", or "temporal_noise"
    - encoder: imgANN -> wrapped frozen image encoder
    - cfg: Cfg -> architecture and noise settings
    - n_timepoints: int -> number of target time bins
    - n_neurons: int -> number of MUA channels

OUTPUT:
    - model: BaselineModel -> the requested decoder
"""
def build_variant_model(variant, encoder, cfg, n_timepoints, n_neurons):
    shared_kwargs = {
        "temporal_embedding_dim": cfg.temporal_embedding_dim,
        "value_dim": cfg.value_dim,
        "n_timepoints": n_timepoints,
        "temporal_compression_ratio": 1,
        "n_neurons": n_neurons,
        "mlp_hidden_dim": cfg.mlp_hidden_dim,
        "dropout": cfg.dropout,
        "attention_granularity": cfg.attention_granularity,
    }
    if variant == "baseline":
        return BaselineModel(encoder, layers=cfg.layer_names, **shared_kwargs)
    # end if the plain reference decoder was requested
    if variant == "noise_layer":
        return NoiseLayerBaselineModel(
            encoder,
            layers=cfg.layer_names,
            match_noise_norm=cfg.match_noise_norm,
            noise_in_eval=cfg.noise_layer_in_eval,
            **shared_kwargs,
        )
    # end if the Gaussian-sphere pseudo-layer was requested
    if variant == "temporal_noise":
        return TemporalNoiseBaselineModel(
            encoder,
            layers=cfg.layer_names,
            temporal_noise_std=cfg.temporal_noise_std,
            relative_noise=cfg.relative_temporal_noise,
            noise_in_eval=cfg.temporal_noise_in_eval,
            **shared_kwargs,
        )
    # end if the temporal-embedding jitter was requested
    raise KeyError(f"Unknown variant '{variant}'.")
# EOF


"""
train_baseline_variant
Optimize a BaselineModel-style decoder on cached features and restore its best
validation checkpoint.

BaselineModel takes ``use_precomputed_features``, so it cannot go through
train_cached_feature_decoder. Selection follows cfg.selection_metric: validation
MSE rewards shrinkage, so "stim_r" -- the image selectivity these experiments
report -- is the usual choice. Stochastic variants are evaluated in eval mode,
which is where their noise is switched off unless noise_in_eval was requested.

INPUT:
    - model: BaselineModel -> decoder to optimize, already on the device
    - loaders: dict -> "train" and "validation" DataLoader objects
    - cfg: Cfg -> epochs, patience, optimizer, and selection settings
    - device: torch.device -> compute device
    - verbose: bool -> print one line per epoch

OUTPUT:
    - history: list[dict] -> per-epoch train MSE, validation MSE and stim_r
    - best_epoch: int -> selected checkpoint epoch
    - best_validation_mse: float -> validation MSE at that checkpoint
    - best_validation_stim_r: float -> validation stim_r at that checkpoint
"""
def train_baseline_variant(model, loaders, cfg, device, verbose=True):
    if cfg.selection_metric not in {"mse", "stim_r"}:
        raise ValueError("selection_metric must be either 'mse' or 'stim_r'.")
    # end if the selection criterion is unsupported

    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    optimizer.zero_grad(set_to_none=True)
    cost_function = neural_activity_timebin_mse_loss

    best_state = copy.deepcopy(model.state_dict())
    # Both criteria are minimized once stim_r is negated.
    best_selection_score = np.inf
    best_validation_mse, best_validation_stim_r = np.inf, -np.inf
    best_epoch, epochs_without_improvement = 0, 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        train_mse = training_step(
            model,
            loaders["train"],
            optimizer,
            cost_function,
            use_precomputed_features=True,
            device=device,
        )
        validation_predictions, validation_targets, _ = predict_test_trials(
            model, loaders["validation"], device
        )
        validation_mse = float(
            np.mean((validation_predictions - validation_targets) ** 2)
        )
        validation_stim_r = float(
            np.nanmean(
                stimulus_correlation(validation_predictions, validation_targets)
            )
        )
        selection_score = (
            validation_mse
            if cfg.selection_metric == "mse"
            else -validation_stim_r
        )
        history.append(
            {
                "epoch": epoch,
                "train_mse": train_mse,
                "validation_mse": validation_mse,
                "validation_stim_r": validation_stim_r,
            }
        )
        if verbose:
            print(
                f"    epoch {epoch:03d}/{cfg.epochs:03d} | train {train_mse:.6f}"
                f" | validation MSE {validation_mse:.6f} | validation stim_r "
                f"{validation_stim_r:.4f}"
            )
        # end if per-epoch reporting is requested

        if selection_score < best_selection_score - 1e-8:
            best_selection_score = selection_score
            best_validation_mse = validation_mse
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
            break
        # end if early-stopping patience is exhausted
    # end for optimization epoch

    model.load_state_dict(best_state)
    return (
        history,
        best_epoch,
        float(best_validation_mse),
        float(best_validation_stim_r),
    )
# EOF


# EOC
