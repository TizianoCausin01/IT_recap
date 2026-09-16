"""
Level-one ridge maps for the TVSD stacked decoders.

The time-bin architecture search fits a single ridge from all selected ANN
depths at once, and hands that same concatenated feature vector to every
decoder it compares. This module builds the other arrangement: one ridge per
depth, each one predicting the complete [time, neurons] response on its own.
A decoder stacked on top then reads L predictions of the target rather than L
feature vectors, and its only remaining job is to decide which depth to believe
in which time bin, and by how much.

The fit split's maps are cross-fitted: every fit presentation is predicted by a
ridge that never saw it. Without that the stacked decoder would be trained on
level-one predictions that are far better than the ones it meets at test time,
and it would learn to trust them accordingly.
"""

import numpy as np
from sklearn.linear_model import RidgeCV

from IT_recap.neural_prediction_training import stimulus_correlation


# The penalty grid the joint ridge reference already searches, so the level-one
# maps are regularized on exactly the same scale as the model they compete with.
RIDGE_ALPHAS = np.logspace(1.0, 7.0, 13)


"""
fit_ridge_layer_map
Fit one RidgeCV from a single feature group to the complete response.

Standardization uses the fit split alone, matching fit_ridge_map, so nothing
leaks from validation or test into the level-one maps.

INPUT:
    - fit_features: np.ndarray -> [fit presentations, embedding]
    - fit_targets: np.ndarray -> [fit presentations, time, neurons]
    - predict_features: dict -> name to [presentations, embedding] to predict
    - alphas: np.ndarray -> penalty grid searched by generalized cross-validation

OUTPUT:
    - predictions: dict -> name to [presentations, time, neurons]
    - alpha: float -> selected ridge penalty
"""
def fit_ridge_layer_map(
    fit_features, fit_targets, predict_features, alphas=RIDGE_ALPHAS
):
    fit_features = np.asarray(fit_features, dtype=np.float32)
    feature_mean = fit_features.mean(axis=0, keepdims=True)
    feature_scale = fit_features.std(axis=0, keepdims=True) + 1e-6

    n_timepoints, n_neurons = fit_targets.shape[1], fit_targets.shape[2]
    ridge_model = RidgeCV(alphas=alphas)
    ridge_model.fit(
        (fit_features - feature_mean) / feature_scale,
        fit_targets.reshape(len(fit_features), -1),
    )
    predictions = {
        name: ridge_model.predict(
            (np.asarray(split_features, dtype=np.float32) - feature_mean)
            / feature_scale
        )
        .reshape(len(split_features), n_timepoints, n_neurons)
        .astype(np.float32)
        for name, split_features in predict_features.items()
    }
    return predictions, float(ridge_model.alpha_)
# EOF


"""
cross_fit_layer_ridge
Predict every split from one feature group, the fit split out of fold.

Fit-split presentations show each training image exactly once, so splitting
them into folds is also a split over images: a held-out presentation shares no
stimulus with the ridge that predicts it. Validation and test are predicted by
the ridge fitted on the whole fit split, which is the map a deployed level-one
model would use.

INPUT:
    - subset_features: dict -> split name to [presentations, embedding]
    - subset_targets: dict -> split name to [presentations, time, neurons]
    - n_folds: int -> cross-fitting folds over the fit split
    - seed: int -> fold assignment seed

OUTPUT:
    - predictions: dict -> split name to [presentations, time, neurons]
    - diagnostics: dict -> selected alpha, per-fold alphas, validation scores
"""
def cross_fit_layer_ridge(subset_features, subset_targets, n_folds, seed):
    if n_folds < 2:
        raise ValueError("n_folds must be at least two to cross-fit.")
    # end if cross-fitting was requested with a single fold

    fit_features = subset_features["train"]
    fit_targets = subset_targets["train"]
    fold_rng = np.random.default_rng(seed)
    fold_of = fold_rng.permutation(len(fit_features)) % n_folds

    out_of_fold = np.empty_like(fit_targets, dtype=np.float32)
    fold_alphas = []
    for fold in range(n_folds):
        held_out = np.flatnonzero(fold_of == fold)
        kept = np.flatnonzero(fold_of != fold)
        fold_predictions, fold_alpha = fit_ridge_layer_map(
            fit_features[kept],
            fit_targets[kept],
            {"held_out": fit_features[held_out]},
        )
        out_of_fold[held_out] = fold_predictions["held_out"]
        fold_alphas.append(fold_alpha)
    # end for cross-fitting fold

    # The map used at prediction time is the one fitted on everything.
    other_splits = {
        split_name: split_features
        for split_name, split_features in subset_features.items()
        if split_name != "train"
    }
    full_predictions, alpha = fit_ridge_layer_map(
        fit_features, fit_targets, other_splits
    )
    predictions = {"train": out_of_fold, **full_predictions}

    validation_predictions = predictions["validation"]
    diagnostics = {
        "alpha": alpha,
        "fold_alphas": fold_alphas,
        "validation_mse": round(
            float(
                np.mean((validation_predictions - subset_targets["validation"]) ** 2)
            ),
            5,
        ),
        "validation_stim_r": round(
            float(
                np.nanmean(
                    stimulus_correlation(
                        validation_predictions, subset_targets["validation"]
                    )
                )
            ),
            4,
        ),
    }
    return predictions, diagnostics
# EOF


"""
build_ridge_stack
Turn every feature group into its own ridge map and stack the predictions.

The stacked array keeps the [presentations, groups, embedding] layout every
cached-feature decoder consumes, with the flattened [time, neurons] prediction
of one depth taking the place of that depth's feature vector. The decoder above
is therefore unchanged; only the meaning of its input axis is.

INPUT:
    - subset_features: dict -> split name to [presentations, layers, embedding]
    - subset_targets: dict -> split name to [presentations, time, neurons]
    - n_folds: int -> cross-fitting folds over the fit split
    - seed: int -> fold assignment seed
    - verbose: bool -> print one line per fitted depth

OUTPUT:
    - stacked: dict -> split name to [presentations, layers, time * neurons]
    - diagnostics: list[dict] -> per-layer alpha and validation scores
"""
def build_ridge_stack(subset_features, subset_targets, n_folds, seed, verbose=True):
    n_layers = subset_features["train"].shape[1]
    n_timepoints, n_neurons = subset_targets["train"].shape[1:]

    stacked = {
        split_name: np.empty(
            (len(split_features), n_layers, n_timepoints * n_neurons),
            dtype=np.float32,
        )
        for split_name, split_features in subset_features.items()
    }
    diagnostics = []
    for layer in range(n_layers):
        layer_features = {
            split_name: split_features[:, layer]
            for split_name, split_features in subset_features.items()
        }
        predictions, layer_diagnostics = cross_fit_layer_ridge(
            layer_features, subset_targets, n_folds, seed
        )
        for split_name, split_predictions in predictions.items():
            stacked[split_name][:, layer] = split_predictions.reshape(
                len(split_predictions), -1
            )
        # end for predicted split
        layer_diagnostics["layer"] = layer
        diagnostics.append(layer_diagnostics)
        if verbose:
            print(
                f"  depth {layer}: alpha {layer_diagnostics['alpha']:.3g} | "
                f"validation stim_r {layer_diagnostics['validation_stim_r']:.4f}"
            )
        # end if per-depth reporting is requested
    # end for feature group
    return stacked, diagnostics
# EOF


"""
average_stacked_predictions
Average the level-one maps back into one [presentations, time, neurons] guess.

This is the stacking baseline that costs nothing: if a trained decoder on top of
the ridge maps cannot beat their plain mean, whatever it learned about which
depth to trust is not worth its parameters.

INPUT:
    - stacked_split: np.ndarray -> [presentations, layers, time * neurons]
    - n_timepoints: int -> number of target bins
    - n_neurons: int -> number of recorded sites

OUTPUT:
    - predictions: np.ndarray -> [presentations, time, neurons]
"""
def average_stacked_predictions(stacked_split, n_timepoints, n_neurons):
    return stacked_split.mean(axis=1).reshape(-1, n_timepoints, n_neurons)
# EOF
