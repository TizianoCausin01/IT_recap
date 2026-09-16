"""
Decode frozen DINOv3 layer representations from the TVSD IT population.

Every other TVSD experiment in this project runs the encoding direction, and
reversing it changes what the hard parts are. Three of them matter here.

The target is noiseless. A DINOv3 layer is a deterministic function of the
image, so there is no trial noise on the output side and no stim_r ceiling of
the usual kind. Validation MSE is therefore a legitimate selection criterion
again -- the shrinkage bias that makes it the wrong criterion for noisy neural
targets simply is not present.

The held-out split is large. Each of the 22,248 unique-image presentations is
its own sample with its own noiseless target, so a 10% held-out split is ~2,200
images rather than the 100 repeated ones, and it supports model selection that
the repeated-test set never could.

The ceiling moved to the input. What limits decoding is trial noise in the
*neural* response, so the ceiling is estimated by decoding from averages of n
repetitions of the 100 repeated test images and extrapolating n to infinity.
For a linear map with additive input noise the score follows
r(n) = r_inf / sqrt(1 + k/n), i.e. 1/r(n)^2 is linear in 1/n, and the intercept
of that line is the ceiling. It bounds *this decoder* given a clean input, not
every possible decoder, which is exactly the quantity that separates "the model
is weak" from "the response is noisy".
"""

import copy
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from IT_recap.neural_prediction_training import stimulus_correlation
from IT_recap.tvsd import compute_tvsd_channel_standardization
from IT_recap.tvsd_experiments import (
    average_targets_over_bin_groups,
    collect_l2_parameters,
    load_cached_targets,
    resolve_cache_paths,
    select_window_bin_indices,
)


# Column of ALLMAT holding the one-based train and test stimulus identifiers.
TRAIN_ID_COLUMN = 1
TEST_ID_COLUMN = 2


"""
fit_pca_basis
Fit a PCA basis on a centred training matrix via the economy SVD.

INPUT:
    - train_matrix: np.ndarray -> [samples, features] fit-split rows only
    - n_components: int -> components retained

OUTPUT:
    - basis: dict -> mean, components [n_components, features], explained
      variance per component, and the total variance of the fit split
"""
def fit_pca_basis(train_matrix, n_components):
    train_matrix = np.asarray(train_matrix, dtype=np.float64)
    if train_matrix.ndim != 2:
        raise ValueError("train_matrix must have shape [samples, features].")
    # end if the matrix axes are invalid
    n_components = int(n_components)
    if not 0 < n_components <= min(train_matrix.shape):
        raise ValueError(
            f"n_components must lie in (0, {min(train_matrix.shape)}]."
        )
    # end if the requested rank is unattainable

    feature_mean = train_matrix.mean(axis=0, keepdims=True)
    centered = train_matrix - feature_mean
    if centered.shape[0] >= centered.shape[1]:
        # Tall matrices -- 22k images by 1024 features, or 178k time bins by
        # 320 channels -- are far cheaper through the feature covariance than
        # through an SVD of the data matrix itself.
        covariance = (centered.T @ centered) / (len(centered) - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        explained_variance = np.maximum(eigenvalues[order], 0.0)
        right_vectors = eigenvectors[:, order].T
    else:
        _, singular_values, right_vectors = np.linalg.svd(
            centered, full_matrices=False
        )
        explained_variance = np.square(singular_values) / (len(centered) - 1)
    # end if the fit split has more samples than features
    return {
        "mean": feature_mean.astype(np.float32),
        "components": right_vectors[:n_components].astype(np.float32),
        "explained_variance": explained_variance[:n_components],
        "total_variance": float(explained_variance.sum()),
    }
# EOF


"""
project_pca
Project rows onto a fitted PCA basis.

INPUT:
    - matrix: np.ndarray -> [samples, features]
    - basis: dict -> output of fit_pca_basis

OUTPUT:
    - scores: np.ndarray -> [samples, n_components]
"""
def project_pca(matrix, basis):
    centered = np.asarray(matrix, dtype=np.float32) - basis["mean"]
    return (centered @ basis["components"].T).astype(np.float32)
# EOF


"""
shuffle_time_axis
Give every presentation an independent random permutation of its time bins.

This is the temporal control: the marginal content of each bin survives, so a
decoder that only needs the response averaged over time is unaffected, while
anything reading the order of the response loses it. The permutation is drawn
once and reused across splits, so train and evaluation see the same destroyed
temporal structure rather than two different ones.

INPUT:
    - sequences: np.ndarray -> [presentations, time, channels]
    - seed: int -> permutation reproducibility

OUTPUT:
    - shuffled: np.ndarray -> [presentations, time, channels]
"""
def shuffle_time_axis(sequences, seed=0):
    rng = np.random.default_rng(seed)
    n_presentations, n_steps = sequences.shape[0], sequences.shape[1]
    permutations = np.argsort(
        rng.random((n_presentations, n_steps)), axis=1
    )
    rows = np.arange(n_presentations)[:, None]
    return sequences[rows, permutations]
# EOF


"""
prepare_decoding_data
Build the neural inputs, the PCA-reduced DINO targets, and the splits.

The neural cache is re-binned to cfg.timebin_ms, standardized per channel on
the training presentations alone, and optionally reduced on the channel axis by
a PCA also fitted on training presentations only. The DINO targets are reduced
per layer by a PCA fitted on the training *images* alone, so neither the
held-out images nor the repeated test images touch any fitted statistic.

INPUT:
    - cfg: object -> window, binning, split, PCA, and layer settings
    - paths: dict -> active project paths

OUTPUT:
    - data: dict -> per-split neural inputs and targets, the repeated-test
      trials and their image targets, the target layout, and the PCA bases
"""
def prepare_decoding_data(cfg, paths):
    neural_cache, allmat = load_cached_targets(cfg, paths)
    cache_paths = resolve_cache_paths(cfg, paths)
    with np.load(cache_paths["features"]) as feature_archive:
        train_features = feature_archive["train_features"]
        test_features = feature_archive["test_features"]
        saved_layer_names = list(feature_archive["layer_names"])
    # end with cached DINO feature archive

    missing_layers = [
        name for name in cfg.layer_names if name not in saved_layer_names
    ]
    if missing_layers:
        raise ValueError(
            f"Cached layers {saved_layer_names} do not contain {missing_layers}."
        )
    # end if a requested target layer is absent
    layer_positions = [saved_layer_names.index(name) for name in cfg.layer_names]
    if len(cfg.target_ranks) != len(cfg.layer_names):
        raise ValueError(
            "target_ranks must have one retained rank per target layer."
        )
    # end if the rank specification does not match the layer list

    # Re-bin the 10 ms cache to the requested bin width over the response window.
    bin_indices, covered_ms = select_window_bin_indices(
        neural_cache.shape[1],
        cfg.target_fs,
        cfg.time_start_ms,
        cfg.window_start_ms,
        cfg.window_end_ms,
    )
    group_size = int(round(cfg.timebin_ms * cfg.target_fs / 1000.0))
    neural = average_targets_over_bin_groups(
        neural_cache, bin_indices, group_size
    )
    bin_edges_ms = np.linspace(
        covered_ms[0], covered_ms[1], neural.shape[1] + 1
    )

    # Split the unique-image presentations three ways; the 100 repeated test
    # images stay whole and are used only for the repetition ceiling.
    unique_rows = np.flatnonzero(allmat[:, TRAIN_ID_COLUMN] > 0)
    repeated_rows = np.flatnonzero(allmat[:, TEST_ID_COLUMN] > 0)
    split_rng = np.random.default_rng(cfg.random_seed)
    shuffled_rows = split_rng.permutation(unique_rows)
    n_validation = max(1, round(len(unique_rows) * cfg.validation_fraction))
    n_heldout = max(1, round(len(unique_rows) * cfg.heldout_fraction))
    split_rows = {
        "validation": shuffled_rows[:n_validation],
        "heldout": shuffled_rows[n_validation:n_validation + n_heldout],
        "train": shuffled_rows[n_validation + n_heldout:],
    }
    if cfg.smoke_test:
        # Enough presentations to exercise every path and nothing more.
        split_rows["train"] = split_rows["train"][:2048]
        split_rows["validation"] = split_rows["validation"][:512]
        split_rows["heldout"] = split_rows["heldout"][:512]
    # end if a smoke run was requested

    channel_mean, channel_scale = compute_tvsd_channel_standardization(
        neural, split_rows["train"]
    )
    neural = (neural - channel_mean) / channel_scale
    if cfg.shuffle_time:
        neural = shuffle_time_axis(neural, seed=cfg.random_seed)
    # end if the temporal control is active

    # Ridge is always fitted on the unreduced channels, so that the baseline is
    # the strongest linear map available rather than one handicapped by a
    # reduction chosen for the recurrent model's sake.
    neural_full = neural
    input_basis = None
    if cfg.input_pca_rank:
        # The channel PCA is fitted on every training time bin pooled together,
        # so it describes the channel covariance rather than any one bin.
        train_bins = neural[split_rows["train"]].reshape(-1, neural.shape[2])
        input_basis = fit_pca_basis(train_bins, cfg.input_pca_rank)
        flat = project_pca(neural.reshape(-1, neural.shape[2]), input_basis)
        neural = flat.reshape(len(neural), neural.shape[1], cfg.input_pca_rank)
    # end if the input is reduced on the channel axis

    # One target block per layer: PCA fitted on the training images alone, then
    # rescaled so the three blocks contribute comparably to a plain MSE.
    train_image_ids = allmat[split_rows["train"], TRAIN_ID_COLUMN] - 1
    fit_image_ids = np.unique(train_image_ids)
    target_bases, target_dims, block_scales = [], [], []
    for position, rank in zip(layer_positions, cfg.target_ranks):
        basis = fit_pca_basis(train_features[fit_image_ids, position], rank)
        scores = project_pca(train_features[fit_image_ids, position], basis)
        if cfg.target_scaling == "component":
            # Every PC gets unit variance: the loss weights a tiny component as
            # heavily as the first one.
            scale = scores.std(axis=0) + 1e-6
        elif cfg.target_scaling == "layer_rms":
            # One scalar per layer: the blocks are comparable to each other, but
            # the PCA variance ordering inside a block survives, which keeps MSE
            # equal to variance-weighted R2 on that layer.
            scale = np.full(rank, float(np.sqrt(np.mean(scores ** 2))) + 1e-6)
        else:
            raise ValueError("target_scaling must be 'component' or 'layer_rms'.")
        # end if the target scaling mode is unsupported
        target_bases.append(basis)
        target_dims.append(int(rank))
        block_scales.append(scale.astype(np.float32))
    # end for target layer

    component_scale = np.concatenate(block_scales)
    boundaries = np.cumsum([0] + target_dims)
    target_slices = [
        (int(boundaries[idx]), int(boundaries[idx + 1]))
        for idx in range(len(target_dims))
    ]

    """
    encode_targets
    Project raw layer features of a set of images onto every target basis.
    """
    def encode_targets(features, image_ids):
        blocks = [
            project_pca(features[image_ids, position], basis)
            for position, basis in zip(layer_positions, target_bases)
        ]
        return (np.concatenate(blocks, axis=1) / component_scale).astype(np.float32)
    # end def encode_targets

    splits = {}
    for split_name, rows in split_rows.items():
        image_ids = allmat[rows, TRAIN_ID_COLUMN] - 1
        split_neural = neural[rows]
        splits[split_name] = {
            "neural": split_neural,
            # The same rows without the channel PCA; the identical array when
            # no reduction is active, so nothing is copied twice for nothing.
            "neural_full": split_neural
            if input_basis is None
            else neural_full[rows],
            "targets": encode_targets(train_features, image_ids),
            "image_ids": image_ids,
        }
    # end for unique-image split

    repeated_image_ids = allmat[repeated_rows, TEST_ID_COLUMN] - 1
    repeated_neural = neural[repeated_rows]
    repeated = {
        "neural": repeated_neural,
        "neural_full": repeated_neural
        if input_basis is None
        else neural_full[repeated_rows],
        "image_ids": repeated_image_ids,
        "image_targets": encode_targets(
            test_features, np.arange(len(test_features))
        ),
    }

    return {
        "splits": splits,
        "repeated": repeated,
        "target_slices": target_slices,
        "target_dims": target_dims,
        "layer_names": list(cfg.layer_names),
        "component_scale": component_scale,
        "target_bases": target_bases,
        "input_basis": input_basis,
        "bin_edges_ms": bin_edges_ms,
        "n_channels": int(neural.shape[2]),
        "n_timepoints": int(neural.shape[1]),
    }
# EOF


"""
score_decoding
Score predicted PCA targets against the true ones, per target layer.

Two numbers per layer, because they answer different questions. `r2` is the
variance-weighted fraction of the retained PCs' variance that the decoder
explains -- the quantity that converts back to the raw DINO feature space.
`mean_component_r` is the unweighted mean correlation over components, which
treats a barely-varying PC as seriously as the first one and is therefore the
harsher of the two.

INPUT:
    - predictions: np.ndarray -> [images, total components]
    - targets: np.ndarray -> [images, total components]
    - target_slices: list -> column range of each target layer
    - layer_names: list -> name of each target layer

OUTPUT:
    - scores: dict -> per-layer and pooled metrics
"""
def score_decoding(predictions, targets, target_slices, layer_names):
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    scores = {}
    for (start, end), layer_name in zip(target_slices, layer_names):
        block_predictions = predictions[:, start:end]
        block_targets = targets[:, start:end]
        residual = np.square(block_predictions - block_targets).sum()
        total = np.square(
            block_targets - block_targets.mean(axis=0, keepdims=True)
        ).sum()
        # stimulus_correlation expects a [images, time, channels] layout, so the
        # component axis stands in for channels with a singleton time axis.
        component_r = stimulus_correlation(
            block_predictions[:, None, :], block_targets[:, None, :]
        )[0]
        scores[layer_name] = {
            "r2": float(1.0 - residual / total),
            "mean_component_r": float(np.nanmean(component_r)),
            "first_component_r": float(component_r[0]),
            "mse": float(np.mean(np.square(block_predictions - block_targets))),
        }
    # end for target layer
    scores["mean_r2"] = float(
        np.mean([scores[name]["r2"] for name in layer_names])
    )
    scores["mean_component_r"] = float(
        np.mean([scores[name]["mean_component_r"] for name in layer_names])
    )
    scores["mse"] = float(np.mean(np.square(predictions - targets)))
    return scores
# EOF


"""
fit_ridge_decoder
Fit a ridge map from the flattened neural response to every target component.

The penalty is chosen on the validation split rather than by generalized
cross-validation, and the selected alpha is reported together with whether it
landed on the edge of the grid: a penalty pinned at an edge means the grid, not
the data, picked it, and the comparison against any decoder is then void.

One eigendecomposition of the (features x features) Gram matrix serves the
whole grid, which is what makes a 40-point alpha sweep cheap enough to be wide.

INPUT:
    - fit_inputs: np.ndarray -> [presentations, time, channels] training input
    - fit_targets: np.ndarray -> [presentations, components] training targets
    - eval_inputs: dict -> split name to [presentations, time, channels]
    - validation_targets: np.ndarray -> targets of the validation split
    - alphas: np.ndarray -> ridge penalties searched
    - target_slices: list | None -> column range of each target layer; given,
      every layer picks its own penalty

OUTPUT:
    - ridge_fit: dict -> per-split predictions, selected alpha(s), edge flag,
      and the validation MSE curve over the grid
"""
def fit_ridge_decoder(
    fit_inputs,
    fit_targets,
    eval_inputs,
    validation_targets,
    alphas,
    target_slices=None,
):
    design = np.asarray(fit_inputs, dtype=np.float64).reshape(len(fit_inputs), -1)
    feature_mean = design.mean(axis=0, keepdims=True)
    feature_scale = design.std(axis=0, keepdims=True) + 1e-6
    design = (design - feature_mean) / feature_scale
    targets = np.asarray(fit_targets, dtype=np.float64)
    target_mean = targets.mean(axis=0, keepdims=True)
    targets = targets - target_mean

    # Eigendecomposing X'X once turns every alpha into a rescaling of the same
    # rotated statistics instead of a fresh solve.
    gram = design.T @ design
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    rotated_cross = eigenvectors.T @ (design.T @ targets)

    standardized_eval = {
        split_name: (
            np.asarray(split_inputs, dtype=np.float64).reshape(
                len(split_inputs), -1
            )
            - feature_mean
        )
        / feature_scale
        for split_name, split_inputs in eval_inputs.items()
    }
    validation_targets = np.asarray(validation_targets, dtype=np.float64)

    # The three target depths differ by a factor of four in how well they are
    # decodable, so one shared penalty is not the best linear map available.
    # Selecting a penalty per depth is what makes this the strongest baseline
    # rather than a conveniently weak one.
    blocks = target_slices or [(0, targets.shape[1])]
    validation_curve = np.zeros((len(alphas), len(blocks)))
    for alpha_index, alpha in enumerate(alphas):
        alpha_weights = eigenvectors @ (
            rotated_cross / (eigenvalues + alpha)[:, None]
        )
        predictions = (
            standardized_eval["validation"] @ alpha_weights + target_mean
        )
        for block_index, (start, end) in enumerate(blocks):
            validation_curve[alpha_index, block_index] = np.mean(
                np.square(
                    predictions[:, start:end] - validation_targets[:, start:end]
                )
            )
        # end for target block
    # end for candidate penalty

    best_indices = np.argmin(validation_curve, axis=0)
    best_alphas = [float(alphas[index]) for index in best_indices]
    # Each block's columns are solved at its own penalty and written back into
    # one weight matrix, so every downstream user sees a single linear map.
    weights = np.zeros((design.shape[1], targets.shape[1]))
    for block_index, (start, end) in enumerate(blocks):
        block_weights = eigenvectors @ (
            rotated_cross[:, start:end]
            / (eigenvalues + best_alphas[block_index])[:, None]
        )
        weights[:, start:end] = block_weights
    # end for target block
    best_alpha = best_alphas[0] if len(blocks) == 1 else best_alphas
    return {
        "predictions": {
            split_name: (split_design @ weights + target_mean).astype(np.float32)
            for split_name, split_design in standardized_eval.items()
        },
        "weights": weights,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "target_mean": target_mean,
        "alpha": best_alpha,
        "alpha_at_grid_edge": bool(
            np.any((best_indices == 0) | (best_indices == len(alphas) - 1))
        ),
        "validation_curve": validation_curve.tolist(),
        "n_coefficients": int(weights.size + target_mean.size),
    }
# EOF


"""
make_ridge_predictor
Wrap a fitted ridge map as a callable on raw [presentations, time, channels].

INPUT:
    - ridge_fit: dict -> output of fit_ridge_decoder

OUTPUT:
    - predict: callable -> neural sequences to predicted components
"""
def make_ridge_predictor(ridge_fit):
    def predict(sequences):
        design = np.asarray(sequences, dtype=np.float64).reshape(
            len(sequences), -1
        )
        design = (design - ridge_fit["feature_mean"]) / ridge_fit["feature_scale"]
        return design @ ridge_fit["weights"] + ridge_fit["target_mean"]
    # end def predict
    return predict
# EOF


"""
repetition_ceiling
Extrapolate decoding score to a noiseless input using the repeated test images.

Each of the 100 repeated images has 30 repetitions. Decoding from the mean of n
of them and sweeping n traces how much of the shortfall is trial noise in the
input. For an additive-noise input the score obeys r(n) = r_inf / sqrt(1 + k/n),
so 1/r(n)^2 is linear in 1/n and the fitted intercept gives r_inf.

The caveat to read this with: the decoder was trained on single repetitions, so
feeding it averages is a mild input-distribution shift. It makes r_inf a
slightly conservative estimate of what a decoder trained on clean input would
reach, not an optimistic one.

INPUT:
    - predict: callable -> neural sequences to predicted components
    - trial_neural: np.ndarray -> [presentations, time, channels] repeated trials
    - image_ids: np.ndarray -> zero-based image identifier per presentation
    - image_targets: np.ndarray -> [images, components] noiseless targets
    - target_slices: list -> column range of each target layer
    - layer_names: list -> name of each target layer
    - rep_counts: sequence[int] -> repetition counts averaged over
    - n_resamples: int -> random repetition draws per count
    - seed: int -> draw reproducibility

OUTPUT:
    - ceiling: dict -> the r(n) curve per layer, the fitted r_inf, and the
      single-repetition fraction of that ceiling
"""
def repetition_ceiling(
    predict,
    trial_neural,
    image_ids,
    image_targets,
    target_slices,
    layer_names,
    rep_counts=(1, 2, 3, 5, 10, 15, 30),
    n_resamples=20,
    seed=0,
):
    rng = np.random.default_rng(seed)
    n_images = len(image_targets)
    rows_by_image = [
        np.flatnonzero(image_ids == image_id) for image_id in range(n_images)
    ]
    max_reps = min(len(rows) for rows in rows_by_image)
    rep_counts = [int(n) for n in rep_counts if n <= max_reps]

    curve = {layer_name: [] for layer_name in layer_names}
    for n_reps in rep_counts:
        resample_scores = {layer_name: [] for layer_name in layer_names}
        # A full average over every repetition has nothing left to resample.
        n_draws = 1 if n_reps == max_reps else n_resamples
        for _ in range(n_draws):
            averaged = np.stack(
                [
                    trial_neural[rng.choice(rows, n_reps, replace=False)].mean(
                        axis=0
                    )
                    for rows in rows_by_image
                ]
            )
            scores = score_decoding(
                predict(averaged), image_targets, target_slices, layer_names
            )
            for layer_name in layer_names:
                resample_scores[layer_name].append(
                    scores[layer_name]["mean_component_r"]
                )
            # end for target layer
        # end for repetition draw
        for layer_name in layer_names:
            curve[layer_name].append(float(np.mean(resample_scores[layer_name])))
        # end for target layer
    # end for repetition count

    inverse_counts = 1.0 / np.asarray(rep_counts, dtype=float)
    ceiling = {"rep_counts": rep_counts, "curve": curve}
    for layer_name in layer_names:
        observed = np.asarray(curve[layer_name], dtype=float)
        if np.any(observed <= 0) or len(rep_counts) < 2:
            # A non-positive score makes the 1/r^2 transform meaningless.
            ceiling[layer_name] = {
                "r_infinite_reps": float("nan"),
                "fraction_of_ceiling": float("nan"),
            }
            continue
        # end if the curve cannot be transformed
        slope, intercept = np.polyfit(inverse_counts, 1.0 / observed ** 2, 1)
        r_infinite = float(1.0 / np.sqrt(intercept)) if intercept > 0 else float("nan")
        ceiling[layer_name] = {
            "r_infinite_reps": r_infinite,
            "single_rep_r": float(observed[0]),
            "full_average_r": float(observed[-1]),
            "fraction_of_ceiling": float(observed[0] / r_infinite)
            if r_infinite == r_infinite
            else float("nan"),
            "noise_slope": float(slope),
        }
    # end for target layer
    return ceiling
# EOF


"""
make_decoding_loader
Wrap neural sequences and their targets in a DataLoader.

INPUT:
    - neural: np.ndarray -> [presentations, time, channels]
    - targets: np.ndarray -> [presentations, components]
    - cfg: object -> batch size, worker count, and seed
    - shuffle: bool -> whether to shuffle presentations

OUTPUT:
    - loader: DataLoader -> (neural, targets) batches
"""
def make_decoding_loader(neural, targets, cfg, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(neural, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(targets, dtype=np.float32)),
    )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        generator=torch.Generator().manual_seed(cfg.random_seed) if shuffle else None,
    )
# EOF


"""
multi_target_mse
Weighted sum of the per-target-layer mean squared errors.

INPUT:
    - predictions: torch.Tensor -> [batch, total components]
    - targets: torch.Tensor -> [batch, total components]
    - target_slices: list -> column range of each target layer
    - weights: sequence[float] -> one weight per target layer

OUTPUT:
    - loss: torch.Tensor -> scalar weighted loss
    - per_layer: list[torch.Tensor] -> the unweighted per-layer errors
"""
def multi_target_mse(predictions, targets, target_slices, weights):
    per_layer = [
        torch.nn.functional.mse_loss(
            predictions[:, start:end], targets[:, start:end]
        )
        for start, end in target_slices
    ]
    loss = sum(
        weight * layer_loss for weight, layer_loss in zip(weights, per_layer)
    )
    return loss, per_layer
# EOF


"""
evaluate_neural_decoder
Run a decoder over one loader and collect predictions, targets, and attention.

INPUT:
    - model: nn.Module -> decoder returning (predictions, diagnostics)
    - loader: DataLoader -> (neural, targets) batches
    - device: torch.device -> compute device

OUTPUT:
    - predictions: np.ndarray -> [presentations, components]
    - targets: np.ndarray -> [presentations, components]
    - attention: np.ndarray | None -> [presentations, targets, time]
"""
def evaluate_neural_decoder(model, loader, device):
    model.eval()
    prediction_batches, target_batches, attention_batches = [], [], []
    with torch.no_grad():
        for neural, targets in loader:
            predictions, diagnostics = model(neural.to(device))
            prediction_batches.append(predictions.cpu())
            target_batches.append(targets)
            if diagnostics.get("attention") is not None:
                attention_batches.append(diagnostics["attention"].cpu())
            # end if the decoder pools by attention
        # end for evaluation batch
    # end with inference mode
    attention = (
        torch.cat(attention_batches).numpy() if attention_batches else None
    )
    return (
        torch.cat(prediction_batches).numpy(),
        torch.cat(target_batches).numpy(),
        attention,
    )
# EOF


"""
train_neural_decoder
Train a decoder with early stopping on the validation split.

Selection is on validation MSE. That is the wrong criterion for the encoding
experiments in this project, where trial noise in the neural target rewards
shrunken predictions, but the targets here are deterministic image features, so
MSE carries no such bias and is simply the objective being optimized.

INPUT:
    - model: nn.Module -> decoder returning (predictions, diagnostics)
    - loaders: dict -> train and validation DataLoaders
    - cfg: object -> optimization settings
    - target_slices: list -> column range of each target layer
    - device: torch.device -> compute device
    - verbose: bool -> whether to print per-epoch progress

OUTPUT:
    - history: list[dict] -> per-epoch training record
    - summary: dict -> best epoch, best validation MSE, and wall-clock seconds
"""
def train_neural_decoder(
    model, loaders, cfg, target_slices, device, verbose=True
):
    weights = list(cfg.loss_weights) or [1.0] * len(target_slices)
    if len(weights) != len(target_slices):
        raise ValueError("loss_weights must have one entry per target layer.")
    # end if the loss weighting does not match the targets

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

    # An explicit sum-of-squares penalty is not the same thing as AdamW's
    # decoupled weight decay, and in the encoding direction the explicit form
    # at ridge's own lambda is what lifted decoders past ridge. It is offered
    # here for the same reason, on the maps the model names as ridge-analogous.
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
    best_validation_mse, best_epoch = np.inf, 0
    epochs_without_improvement = 0
    history = []
    started_at = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss_sum, train_samples = 0.0, 0
        for neural, targets in loaders["train"]:
            neural, targets = neural.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions, _ = model(neural)
            loss, _ = multi_target_mse(
                predictions, targets, target_slices, weights
            )
            if penalized_parameters:
                # The reported train loss stays the data term alone, so runs
                # with and without a penalty stay comparable epoch by epoch.
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
            train_loss_sum += float(loss.detach()) * len(neural)
            train_samples += len(neural)
        # end for training batch

        validation_predictions, validation_targets, _ = evaluate_neural_decoder(
            model, loaders["validation"], device
        )
        validation_mse = float(
            np.mean(np.square(validation_predictions - validation_targets))
        )
        # Per-layer validation errors are tracked separately, because a single
        # aggregate hides one target collapsing while the others improve.
        per_layer_mse = [
            float(
                np.mean(
                    np.square(
                        validation_predictions[:, start:end]
                        - validation_targets[:, start:end]
                    )
                )
            )
            for start, end in target_slices
        ]
        scheduler.step(validation_mse)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_sum / train_samples,
                "validation_mse": validation_mse,
                "validation_mse_per_layer": per_layer_mse,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if verbose:
            layer_text = " ".join(f"{value:.4f}" for value in per_layer_mse)
            print(
                f"  epoch {epoch:3d} | train {history[-1]['train_loss']:.4f} | "
                f"val {validation_mse:.4f} | per-layer {layer_text}"
            )
        # end if progress is printed

        if validation_mse < best_validation_mse - cfg.min_improvement:
            best_validation_mse = validation_mse
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        # end if the epoch improved on the best validation error
        if epochs_without_improvement >= cfg.patience:
            break
        # end if early stopping triggered
    # end for epoch

    model.load_state_dict(best_state)
    return history, {
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_validation_mse": best_validation_mse,
        "train_seconds": time.time() - started_at,
    }
# EOF


"""
attention_entropy
Entropy in nats of each target layer's mean temporal attention distribution.

A value near log(T) means the layer reads the whole response; a value near zero
means it has collapsed onto a single bin, which is worth comparing against a
best-single-bin linear baseline before calling it a finding.

INPUT:
    - attention: np.ndarray -> [presentations, targets, time]

OUTPUT:
    - entropies: np.ndarray -> [targets] entropy of the trial-averaged pattern
"""
def attention_entropy(attention):
    mean_attention = np.asarray(attention, dtype=np.float64).mean(axis=0)
    safe = np.clip(mean_attention, 1e-12, None)
    return -(safe * np.log(safe)).sum(axis=-1)
# EOF


"""
count_trainable_parameters
Count the parameters an optimizer will update.

INPUT:
    - model: nn.Module -> any model

OUTPUT:
    - count: int -> trainable parameter count
"""
def count_trainable_parameters(model):
    return int(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )
# EOF


"""
temporal_centroid
Centre of mass, in milliseconds, of a non-negative weighting over time bins.

Applied to an attention distribution it gives the single number that makes the
depth-ordering claim testable: shallow features attended earlier than deep ones
means a smaller centroid. Applied to a per-bin decodability curve it gives the
same summary of the linear reference, so the two are directly comparable.

INPUT:
    - weights: np.ndarray -> [..., time] non-negative weights
    - bin_centers_ms: np.ndarray -> [time] centre of each bin in milliseconds

OUTPUT:
    - centroid_ms: np.ndarray -> [...] weighted mean time
"""
def temporal_centroid(weights, bin_centers_ms):
    weights = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
    total = weights.sum(axis=-1)
    return np.divide(
        (weights * np.asarray(bin_centers_ms)).sum(axis=-1),
        total,
        out=np.full(total.shape, np.nan),
        where=total > 0,
    )
# EOF
