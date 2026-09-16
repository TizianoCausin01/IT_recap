"""
Decoding utilities for the trial-averaged three0 natraster.

The published temporal-architecture sweep predicts three0 AIT from mean-pooled
DINOv3 features with a concatenated-layer RidgeCV. That reference is limited by
three things this module addresses: a truncated alpha grid, a validation split
that is never folded back into the fit, and an independent regression per
(time, unit) output even though the 750 outputs share a handful of image-driven
components.

The decoder here is a reduced-rank kernel ridge. Targets are re-weighted by the
inverse of their trial-mean noise standard deviation, projected on the leading
components of the fit split, regressed from an averaged multi-backbone linear
kernel with one cross-validated alpha per component, and mapped back. Every
hyperparameter is selected inside the fitting rows, so the held-out images are
touched exactly once.
"""

from pathlib import Path

import numpy as np


# The stimulus set and recording the temporal-architecture sweep uses.
THREE0_MONKEY = "three0"
THREE0_DATE = "250313"
THREE0_STIMULI = "talia_20each_tizi"


"""
load_three0_targets
Load the trial-averaged three0 responses for one area, ordered by the monkey's
presentation order, together with the matching ANN feature ordering.

INPUT:
    - paths: dict -> active project paths
    - brain_area: str -> configured cortical-area name, None for the full probe
    - target_fs: int -> target sampling frequency in Hz
    - time_start_ms: float -> beginning of the retained window
    - time_end_ms: float -> end of the retained window
    - stimuli_folder: str -> ImageFolder name under Stimuli

OUTPUT:
    - targets: np.ndarray -> [images, time, units] trial-averaged responses
    - ann_index: np.ndarray -> ANN presentation index of every target row
    - image_paths: list[str] -> stimulus file of every target row
"""
def load_three0_targets(
    paths,
    brain_area="AIT",
    target_fs=100,
    time_start_ms=0.0,
    time_end_ms=300.0,
    stimuli_folder=THREE0_STIMULI,
):
    from torchvision.datasets import ImageFolder

    from IT_recap.hf_feature_extraction import is_valid_image_file
    from project_specific_utils.dataloader import (
        load_img_natraster,
        map_image_order_from_ann_to_monkey,
    )

    stimuli_root = Path(paths["livingstone_lab"]) / "Stimuli" / stimuli_folder
    image_dataset = ImageFolder(
        stimuli_root, is_valid_file=is_valid_image_file, allow_empty=True
    )
    ann_index = np.asarray(
        map_image_order_from_ann_to_monkey(
            paths, THREE0_MONKEY, THREE0_DATE, image_dataset
        ),
        dtype=int,
    )
    raster = load_img_natraster(
        paths=paths,
        monkey_name=THREE0_MONKEY,
        date=THREE0_DATE,
        original_fs=1000,
        new_fs=target_fs,
        time_start_ms=time_start_ms,
        time_end_ms=time_end_ms,
        brain_area=brain_area,
    )
    # TimeSeries stores [units, time, images]; the decoders want image-major.
    targets = raster.get_array().transpose(2, 1, 0).astype(np.float32)
    if len(ann_index) != len(targets):
        raise ValueError(
            f"Found {len(ann_index)} mapped images and {len(targets)} targets."
        )
    # end if the neural and stimulus spaces are misaligned
    image_paths = [image_dataset.samples[i][0] for i in ann_index]
    return targets, ann_index, image_paths
# EOF


"""
load_three0_trial_noise
Measure the residual noise variance of the trial mean for every output, using
the repeated single-trial rasters that the natraster average is built from.

This is what turns a raw MSE into an interpretable number: the noise variance
of the trial mean is the part of the test error no image-computable model can
remove, so it fixes both the noise ceiling and the reducible error.

INPUT:
    - paths: dict -> active project paths
    - brain_area: str -> configured cortical-area name
    - target_fs: int -> target sampling frequency in Hz
    - time_start_ms: float -> beginning of the retained window
    - time_end_ms: float -> end of the retained window
    - raster_file: str -> repeated-presentation raster under data_path/data
    - image_names_file: str -> per-trial image names for that raster
    - cache_path: Path | None -> optional .npz cache of the computed variance

OUTPUT:
    - noise_variance: np.ndarray -> [images, time, units] variance of the mean
    - image_names: np.ndarray -> stimulus name of every image row
"""
def load_three0_trial_noise(
    paths,
    brain_area="AIT",
    target_fs=100,
    time_start_ms=0.0,
    time_end_ms=300.0,
    raster_file="rasters_three0_250313to21.mat",
    image_names_file="allimages_three0_250313to21.mat",
    cache_path=None,
):
    import h5py

    from project_specific_utils.dataloader import (
        BrainAreas,
        decode_matlab_strings,
        rename_talia_dataset,
    )

    if cache_path is not None and Path(cache_path).is_file():
        with np.load(cache_path, allow_pickle=True) as cached:
            return cached["noise_variance"], cached["image_names"]
        # end with cached noise variance
    # end if the cache is already on disk

    data_dir = Path(paths["data_path"]) / "data"
    with h5py.File(data_dir / image_names_file, "r") as name_file:
        trial_names = decode_matlab_strings(
            name_file, name_file["allimages"][:]
        )
    # end with per-trial image names
    # The natraster row order comes from sorted, renamed unique image names.
    trial_names = np.asarray(rename_talia_dataset(trial_names))
    image_names, trial_image = np.unique(trial_names, return_inverse=True)

    area_limits = BrainAreas(THREE0_MONKEY).get_brain_areas_idx()[brain_area]
    start_sample = round(time_start_ms)
    end_sample = round(time_end_ms)
    bin_width = 1000 // target_fs
    n_bins = (end_sample - start_sample) // bin_width

    with h5py.File(data_dir / raster_file, "r") as raster_file_handle:
        raster = raster_file_handle["rasters"]
        n_trials = raster.shape[0]
        binned = np.empty((n_trials, n_bins, 0), np.float32)
        blocks = []
        for unit_start, unit_end in area_limits:
            block = np.empty((n_trials, n_bins, unit_end - unit_start), np.float32)
            # Read in chunks: the full raster is several GB on disk.
            for chunk in range(0, n_trials, 1000):
                window = raster[
                    chunk : chunk + 1000,
                    start_sample:end_sample,
                    unit_start:unit_end,
                ]
                block[chunk : chunk + 1000] = window.reshape(
                    len(window), n_bins, bin_width, -1
                ).mean(2)
            # end for trial chunk
            blocks.append(block)
        # end for area channel range
        binned = np.concatenate(blocks, axis=2)
    # end with repeated-presentation raster

    n_images = len(image_names)
    noise_variance = np.empty((n_images, n_bins, binned.shape[2]), np.float64)
    for image in range(n_images):
        trials = binned[trial_image == image]
        noise_variance[image] = trials.var(0, ddof=1) / len(trials)
    # end for image

    if cache_path is not None:
        np.savez_compressed(
            cache_path, noise_variance=noise_variance, image_names=image_names
        )
    # end if a cache was requested
    return noise_variance, image_names
# EOF


"""
make_image_splits
Reproduce the holdout image split of run_temporal_architecture_experiments.py.

INPUT:
    - ann_index: np.ndarray -> ANN presentation index of every target row
    - seed: int -> split seed
    - validation_fraction: float -> fraction of images held for validation
    - test_fraction: float -> fraction of images held for test

OUTPUT:
    - splits: dict[str, np.ndarray] -> train, validation, and test row indices
"""
def make_image_splits(ann_index, seed=0, validation_fraction=0.2, test_fraction=0.2):
    ann_index = np.asarray(ann_index, dtype=int)
    unique_images = np.unique(ann_index)
    shuffled = np.random.default_rng(seed).permutation(unique_images)
    n_validation = max(1, round(len(unique_images) * validation_fraction))
    n_test = max(1, round(len(unique_images) * test_fraction))
    if n_validation + n_test >= len(unique_images):
        raise ValueError("Validation and test fractions leave no training data.")
    # end if the holdout fractions exhaust the data
    select = lambda images: np.flatnonzero(np.isin(ann_index, images))
    return {
        "train": select(shuffled[n_validation + n_test :]),
        "validation": select(shuffled[:n_validation]),
        "test": select(shuffled[n_validation : n_validation + n_test]),
    }
# EOF


"""
normalized_linear_kernel
Build one image-by-image linear kernel from a feature block.

Features are z-scored over all images (a transductive step that uses no
targets) and divided by the square root of their dimensionality, so blocks of
very different width contribute comparably. The kernel trace is then fixed so a
single alpha grid is meaningful for every backbone.

INPUT:
    - features: np.ndarray -> [images, dimensions] feature block

OUTPUT:
    - kernel: np.ndarray -> [images, images] normalized linear kernel
"""
def normalized_linear_kernel(features):
    features = np.asarray(features, dtype=np.float64)
    scale = features.std(0)
    scale[scale == 0] = 1.0
    features = (features - features.mean(0)) / scale
    features = features / np.sqrt(features.shape[1])
    kernel = features @ features.T
    return kernel / np.trace(kernel) * len(kernel)
# EOF


"""
kernel_ridge_path
Predict held-out rows for a whole alpha grid from one precomputed kernel.

Centering is done in feature space through the kernel, and the single
eigendecomposition of the fitting block is shared across alphas, which is what
makes the per-component alpha search cheap.

INPUT:
    - kernel: np.ndarray -> [images, images] kernel over all rows
    - fit_rows: np.ndarray -> rows used to fit
    - fit_targets: np.ndarray -> [len(fit_rows), outputs] targets to regress
    - eval_rows: np.ndarray -> rows to predict
    - alphas: np.ndarray -> ridge penalties to evaluate

OUTPUT:
    - predictions: np.ndarray -> [alphas, len(eval_rows), outputs]
"""
def kernel_ridge_path(kernel, fit_rows, fit_targets, eval_rows, alphas):
    fit_block = kernel[np.ix_(fit_rows, fit_rows)]
    eval_block = kernel[np.ix_(eval_rows, fit_rows)]
    column_mean = fit_block.mean(0)
    grand_mean = fit_block.mean()
    centered_fit = fit_block - column_mean[None, :] - column_mean[:, None] + grand_mean
    centered_eval = (
        eval_block
        - column_mean[None, :]
        - eval_block.mean(1, keepdims=True)
        + grand_mean
    )
    eigenvalues, eigenvectors = np.linalg.eigh(centered_fit)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    target_mean = fit_targets.mean(0)
    projected = eigenvectors.T @ (fit_targets - target_mean)
    return np.stack(
        [
            centered_eval
            @ (eigenvectors @ (projected / (eigenvalues + alpha)[:, None]))
            + target_mean
            for alpha in alphas
        ]
    )
# EOF


"""
fit_reduced_rank_decoder
Fit the reduced-rank kernel-ridge decoder and predict held-out images.

INPUT:
    - kernel: np.ndarray -> [images, images] kernel over all rows
    - targets: np.ndarray -> [images, outputs] flattened trial-averaged targets
    - fit_rows: np.ndarray -> rows used to fit and to select hyperparameters
    - eval_rows: np.ndarray -> rows to predict
    - rank: int -> number of target components to regress
    - output_weights: np.ndarray | None -> per-output weight, None for uniform
    - alphas: np.ndarray -> ridge penalties to search
    - n_folds: int -> folds of the inner alpha search
    - seed: int -> seed of the inner fold assignment

OUTPUT:
    - predictions: np.ndarray -> [len(eval_rows), outputs] predicted responses
    - selected_alphas: np.ndarray -> alpha chosen for every component
"""
def fit_reduced_rank_decoder(
    kernel,
    targets,
    fit_rows,
    eval_rows,
    rank=20,
    output_weights=None,
    alphas=None,
    n_folds=5,
    seed=1,
):
    alphas = np.logspace(0, 9, 46) if alphas is None else np.asarray(alphas)
    if output_weights is None:
        output_weights = np.ones(targets.shape[1])
    # end if no output weighting was requested
    # A rank above the fitting-block rank would only add empty components.
    rank = min(rank, len(fit_rows) - 1, targets.shape[1])

    weighted = targets * output_weights
    weighted_mean = weighted[fit_rows].mean(0)
    _, _, right_vectors = np.linalg.svd(
        weighted[fit_rows] - weighted_mean, full_matrices=False
    )
    basis = right_vectors[:rank]
    scores = (weighted - weighted_mean) @ basis.T

    # One alpha per component, scored on inner folds of the fitting rows only.
    order = np.random.default_rng(seed).permutation(len(fit_rows))
    fold_error = np.zeros((len(alphas), rank))
    for fold in range(n_folds):
        held = order[fold::n_folds]
        kept = np.setdiff1d(order, held)
        fold_error += (
            (
                kernel_ridge_path(
                    kernel,
                    fit_rows[kept],
                    scores[fit_rows[kept]],
                    fit_rows[held],
                    alphas,
                )
                - scores[fit_rows[held]]
            )
            ** 2
        ).sum(1)
    # end for inner fold
    selected = fold_error.argmin(0)

    path = kernel_ridge_path(
        kernel, fit_rows, scores[fit_rows], eval_rows, alphas
    )
    predicted_scores = np.stack(
        [path[selected[component], :, component] for component in range(rank)], 1
    )
    predictions = (predicted_scores @ basis + weighted_mean) / output_weights
    return predictions, alphas[selected]
# EOF


"""
score_predictions
Score predicted responses against the trial-averaged targets.

The noise floor is the mean variance of the trial mean over the evaluated
images, so `reducible_fraction` reports how much of the removable error the
decoder actually removed and `fraction_of_ceiling` how much of the explainable
variance it explains.

INPUT:
    - predictions: np.ndarray -> [images, time, units] predicted responses
    - targets: np.ndarray -> [images, time, units] measured responses
    - fit_target_mean: np.ndarray -> [time, units] mean of the fitting rows
    - noise_floor: float | None -> mean variance of the trial mean, or None

OUTPUT:
    - scores: dict -> mse, r2, mean output correlation, and ceiling fractions
"""
def score_predictions(predictions, targets, fit_target_mean, noise_floor=None):
    residual = predictions - targets
    mse = float((residual**2).mean())
    total = float(((targets - fit_target_mean) ** 2).mean())
    scores = {"mse": mse, "r2": 1.0 - mse / total}

    flat_predictions = predictions.reshape(len(predictions), -1)
    flat_targets = targets.reshape(len(targets), -1)
    centered_predictions = flat_predictions - flat_predictions.mean(0)
    centered_targets = flat_targets - flat_targets.mean(0)
    denominator = np.sqrt(
        (centered_predictions**2).sum(0) * (centered_targets**2).sum(0)
    )
    valid = denominator > 0
    correlations = np.full(flat_targets.shape[1], np.nan)
    correlations[valid] = (
        centered_predictions[:, valid] * centered_targets[:, valid]
    ).sum(0) / denominator[valid]
    scores["mean_output_correlation"] = float(np.nanmean(correlations))

    if noise_floor is not None:
        ceiling = 1.0 - noise_floor / total
        scores["noise_ceiling_r2"] = float(ceiling)
        scores["fraction_of_ceiling"] = float(scores["r2"] / ceiling)
    # end if a noise floor was supplied
    return scores
# EOF
