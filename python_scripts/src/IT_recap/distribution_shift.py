"""
Measure whether the TVSD train and test splits are drawn from the same
distribution, and whether any difference costs the decoder anything.

The protocol does not split presentations at random: the 22,248 training images
are shown once each, while the 100 test images are shown 30 times and scored
after averaging those repetitions. Three things can therefore differ between the
splits independently of any model, and each leaves its own signature:

    - the stimuli themselves (a covariate shift on the model input);
    - when in the recording they were run (a day or drift confound on the
      target, which looks like a shift but is not stimulus-driven);
    - how much trial noise survives in the target (single trials while fitting,
      30-repetition averages while scoring).

The first three functions quantify those separately. matched_holdout_stim_r is
the decisive one: it scores a single ridge map on 100 *held-out training* images
and on the 100 test images under the same single-trial conditions, so the two
numbers differ only by the shift. The usual repetition-averaged test score is
reported alongside, which is the part explained by averaging rather than by any
distribution difference.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from scipy.linalg import cho_factor, cho_solve

from IT_recap.neural_prediction_training import (
    aggregate_trials_by_image,
    stimulus_correlation,
)


"""
summarize_presentation_protocol
Compare the two splits on the metadata alone: repetitions, recording day, and
position within the session.

A shift here is the cheapest one to find and the easiest to mistake for a
stimulus effect: if the test presentations sit on different days, or late in
each session, then any target difference between the splits may be electrode
drift or adaptation rather than anything about the images.

INPUT:
    - allmat: np.ndarray -> ALLMAT metadata rows [presentations, 6] of
      trial_idx, train_idx, test_idx, rep, count, day

OUTPUT:
    - report: dict -> per-split counts, the per-day table, the total-variation
      distance between the two day histograms, and within-session positions
"""
def summarize_presentation_protocol(allmat):
    train_rows = np.flatnonzero(allmat[:, 1] > 0)
    test_rows = np.flatnonzero(allmat[:, 2] > 0)
    days = np.unique(allmat[:, 5])

    per_day = []
    for day in days:
        on_day = allmat[:, 5] == day
        per_day.append(
            {
                "day": int(day),
                "n_train": int(np.sum(on_day & (allmat[:, 1] > 0))),
                "n_test": int(np.sum(on_day & (allmat[:, 2] > 0))),
            }
        )
    # end for recording day

    train_histogram = np.array([row["n_train"] for row in per_day], dtype=float)
    test_histogram = np.array([row["n_test"] for row in per_day], dtype=float)
    train_histogram /= train_histogram.sum()
    test_histogram /= test_histogram.sum()
    # Total variation over days: 0 when both splits are recorded on the same
    # days in the same proportions, 1 when they share no day at all.
    day_tv_distance = float(0.5 * np.abs(train_histogram - test_histogram).sum())

    # Position of each presentation within its own day, so a test block run at
    # the end of every session shows up even though trial_idx is not comparable
    # across days.
    day_max_trial = {
        int(day): float(allmat[allmat[:, 5] == day, 0].max()) for day in days
    }
    within_day_position = allmat[:, 0] / np.array(
        [day_max_trial[int(day)] for day in allmat[:, 5]]
    )

    return {
        "n_presentations": int(len(allmat)),
        "n_train_presentations": int(len(train_rows)),
        "n_test_presentations": int(len(test_rows)),
        "n_train_images": int(allmat[:, 1].max()),
        "n_test_images": int(allmat[:, 2].max()),
        "train_reps_per_image": float(len(train_rows) / allmat[:, 1].max()),
        "test_reps_per_image": float(len(test_rows) / allmat[:, 2].max()),
        "n_days": int(len(days)),
        "n_days_with_train": int(sum(row["n_train"] > 0 for row in per_day)),
        "n_days_with_test": int(sum(row["n_test"] > 0 for row in per_day)),
        "day_tv_distance": day_tv_distance,
        "median_within_day_position_train": float(
            np.median(within_day_position[train_rows])
        ),
        "median_within_day_position_test": float(
            np.median(within_day_position[test_rows])
        ),
        "per_day": per_day,
    }
# EOF


"""
domain_classifier_auc
Cross-validated AUC of a logistic classifier asked to tell two groups of
stimuli apart.

This is the standard covariate-shift test: an AUC a balanced classifier cannot
push above chance means the two groups are indistinguishable in this feature
space. The absolute value is not interpretable on its own -- any two finite
high-dimensional samples are separable -- so it is only ever read against the
null this module builds by splitting the training images against themselves.

INPUT:
    - features_a: np.ndarray -> [n_a, components] first group
    - features_b: np.ndarray -> [n_b, components] second group
    - n_folds: int -> stratified cross-validation folds
    - seed: int -> fold shuffling seed

OUTPUT:
    - auc: float -> out-of-fold ROC AUC for the group label
"""
def domain_classifier_auc(features_a, features_b, n_folds=5, seed=0):
    design = np.concatenate([features_a, features_b], axis=0)
    labels = np.concatenate(
        [np.zeros(len(features_a), dtype=int), np.ones(len(features_b), dtype=int)]
    )
    classifier = LogisticRegression(max_iter=2000, C=1.0)
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    out_of_fold = cross_val_predict(
        classifier, design, labels, cv=folds, method="predict_proba"
    )[:, 1]
    return float(roc_auc_score(labels, out_of_fold))
# EOF


"""
stimulus_feature_shift
Test whether the test images differ from the training images in the encoder's
feature space, against a null built from the training images alone.

The reference pool defines the space -- its PCA basis and its mean and scale --
and the probe pool, disjoint from it, supplies the "training" groups. The
observed comparison is a probe group against the test images; the null is one
probe group against another. Both are the same size and live in the same basis,
so the null absorbs the small-sample separability that makes a raw AUC
uninterpretable. Two geometric statistics come along for free: how far a test
image sits from the nearest training image, and how much of its variance falls
outside the training subspace.

INPUT:
    - train_features: np.ndarray -> [train images, groups, dim] cached features
    - test_features: np.ndarray -> [test images, groups, dim] cached features
    - n_components: int -> PCA components retained from the reference pool
    - n_resamples: int -> probe groups drawn for the observed and null tests
    - seed: int -> resampling seed

OUTPUT:
    - report: dict -> observed and null AUCs, nearest-neighbour distances, and
      the out-of-subspace variance fraction for both splits
"""
def stimulus_feature_shift(
    train_features, test_features, n_components=64, n_resamples=10, seed=0
):
    n_test_images = len(test_features)
    train_flat = train_features.reshape(len(train_features), -1).astype(np.float64)
    test_flat = test_features.reshape(n_test_images, -1).astype(np.float64)

    # The probe pool has to hold two disjoint groups for the null plus one for
    # the observed test, and stays out of the basis so nothing leaks into it.
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(train_flat))
    n_probe = max(3 * n_test_images, len(train_flat) // 5)
    probe_pool, reference_pool = shuffled[:n_probe], shuffled[n_probe:]

    reference = train_flat[reference_pool]
    feature_mean = reference.mean(axis=0, keepdims=True)
    feature_scale = reference.std(axis=0, keepdims=True) + 1e-6
    standardize = lambda block: (block - feature_mean) / feature_scale

    # PCA by SVD of the standardized reference pool; the components are the
    # directions the training images actually vary along.
    centered_reference = standardize(reference)
    _, singular_values, right_vectors = np.linalg.svd(
        centered_reference, full_matrices=False
    )
    components = right_vectors[:n_components]
    probe_scores = standardize(train_flat[probe_pool]) @ components.T
    test_scores = standardize(test_flat) @ components.T
    reference_scores = centered_reference @ components.T

    observed_aucs, null_aucs = [], []
    for resample in range(n_resamples):
        # Three disjoint probe groups: one plays "training" for both tests, the
        # second is the null's second group, the third is unused padding that
        # keeps every group the same size as the test set.
        picks = rng.permutation(len(probe_scores))[: 2 * n_test_images]
        group_a = probe_scores[picks[:n_test_images]]
        group_b = probe_scores[picks[n_test_images:]]
        observed_aucs.append(
            domain_classifier_auc(group_a, test_scores, seed=seed + resample)
        )
        null_aucs.append(
            domain_classifier_auc(group_a, group_b, seed=seed + resample)
        )
    # end for resample

    # Coverage: distance from a held-out image to its nearest reference image.
    # Probe images give the matched baseline, since they are held out too.
    def nearest_reference_distance(scores):
        squared = (
            np.square(scores).sum(axis=1)[:, None]
            - 2.0 * scores @ reference_scores.T
            + np.square(reference_scores).sum(axis=1)[None, :]
        )
        return np.sqrt(np.maximum(squared.min(axis=1), 0.0))
    # end nearest_reference_distance

    probe_distances = nearest_reference_distance(probe_scores[:n_test_images])
    test_distances = nearest_reference_distance(test_scores)

    # Energy outside the retained subspace, as a fraction of each image's total
    # standardized variance: a test set the basis does not span scores higher.
    def out_of_subspace_fraction(block):
        standardized = standardize(block)
        total = np.square(standardized).sum(axis=1)
        retained = np.square(standardized @ components.T).sum(axis=1)
        return (total - retained) / total
    # end out_of_subspace_fraction

    return {
        "n_components": int(n_components),
        "variance_retained": float(
            np.square(singular_values[:n_components]).sum()
            / np.square(singular_values).sum()
        ),
        "observed_auc_mean": float(np.mean(observed_aucs)),
        "observed_auc_sd": float(np.std(observed_aucs)),
        "null_auc_mean": float(np.mean(null_aucs)),
        "null_auc_sd": float(np.std(null_aucs)),
        "auc_gap": float(np.mean(observed_aucs) - np.mean(null_aucs)),
        "median_nn_distance_heldout_train": float(np.median(probe_distances)),
        "median_nn_distance_test": float(np.median(test_distances)),
        "nn_distance_ratio": float(
            np.median(test_distances) / np.median(probe_distances)
        ),
        "out_of_subspace_heldout_train": float(
            np.mean(out_of_subspace_fraction(train_flat[probe_pool[:n_test_images]]))
        ),
        "out_of_subspace_test": float(np.mean(out_of_subspace_fraction(test_flat))),
    }
# EOF


"""
neural_target_shift
Compare the recorded responses of the two splits, before and after matching on
recording day, and report how much trial noise the averaging removes.

Two numbers matter. The raw offset says whether the test responses sit at a
different level than the ones the decoder was fitted to, in units of the
training spread the standardization divides by. The day-matched offset repeats
that within each day and averages: if it collapses, the difference is drift
between sessions rather than anything the images did. The spread ratio is the
separate target-side mismatch -- the fitting target is one trial, the scoring
target is a 30-repetition mean, and those have different variance.

INPUT:
    - window_responses: np.ndarray -> [presentations, channels] window-averaged
      responses
    - allmat: np.ndarray -> ALLMAT metadata rows [presentations, 6]
    - n_test_images: int -> size of the repeated-test pool

OUTPUT:
    - report: dict -> raw and day-matched standardized offsets, and the
      single-trial versus repetition-averaged spreads
"""
def neural_target_shift(window_responses, allmat, n_test_images=100):
    train_rows = np.flatnonzero(allmat[:, 1] > 0)
    test_rows = np.flatnonzero(allmat[:, 2] > 0)
    train_mean = window_responses[train_rows].mean(axis=0)
    train_sd = window_responses[train_rows].std(axis=0) + 1e-9
    raw_offset = (window_responses[test_rows].mean(axis=0) - train_mean) / train_sd

    # Repeat the same difference inside each day, then pool by how many test
    # presentations that day contributes.
    day_offsets, day_weights = [], []
    for day in np.unique(allmat[:, 5]):
        on_day = allmat[:, 5] == day
        day_train = np.flatnonzero(on_day & (allmat[:, 1] > 0))
        day_test = np.flatnonzero(on_day & (allmat[:, 2] > 0))
        if len(day_train) < 2 or len(day_test) < 2:
            continue
        # end if this day cannot support a within-day comparison
        day_offsets.append(
            (
                window_responses[day_test].mean(axis=0)
                - window_responses[day_train].mean(axis=0)
            )
            / train_sd
        )
        day_weights.append(len(day_test))
    # end for recording day
    day_matched_offset = np.average(
        np.stack(day_offsets), axis=0, weights=np.asarray(day_weights, dtype=float)
    )

    # Spread of what the decoder is fitted against versus what it is scored on.
    test_image_ids = allmat[test_rows, 2] - 1
    image_means = aggregate_trials_by_image(
        window_responses[test_rows][:, None, :], test_image_ids, n_test_images
    )[:, 0, :]
    single_trial_sd = window_responses[test_rows].std(axis=0)
    averaged_sd = image_means.std(axis=0)

    return {
        "mean_abs_raw_offset": float(np.mean(np.abs(raw_offset))),
        "p95_abs_raw_offset": float(np.percentile(np.abs(raw_offset), 95)),
        "mean_abs_day_matched_offset": float(np.mean(np.abs(day_matched_offset))),
        "p95_abs_day_matched_offset": float(
            np.percentile(np.abs(day_matched_offset), 95)
        ),
        "n_days_compared": int(len(day_offsets)),
        "mean_train_single_trial_sd": float(np.mean(train_sd)),
        "mean_test_single_trial_sd": float(np.mean(single_trial_sd)),
        "mean_test_averaged_sd": float(np.mean(averaged_sd)),
        "averaged_over_single_trial_sd": float(
            np.mean(averaged_sd) / np.mean(single_trial_sd)
        ),
    }
# EOF


"""
fit_ridge_gram
Fit a ridge map by normal equations, selecting the penalty on a held-out block.

RidgeCV takes the SVD route, which needs a dense copy of the design matrix and
is what makes a full-depth feature stack expensive here. Accumulating the Gram
matrix over chunks instead keeps peak memory at one chunk plus the [p, p]
matrix, and the penalty is then one Cholesky solve per candidate.

INPUT:
    - features: np.ndarray -> [presentations, p] design matrix, unstandardized
    - targets: np.ndarray -> [presentations, time, channels] responses
    - fit_rows: np.ndarray -> rows the map is fitted on
    - validation_rows: np.ndarray -> rows the penalty is selected on
    - alphas: np.ndarray -> ridge penalties to compare
    - chunk_size: int -> rows accumulated per pass

OUTPUT:
    - ridge_fit: dict -> predict callable, selected alpha, validation MSE per
      alpha, and whether the choice sits at an edge of the grid
"""
def fit_ridge_gram(
    features, targets, fit_rows, validation_rows, alphas, chunk_size=4096
):
    n_time, n_channels = targets.shape[1:]
    fit_features = features[fit_rows]
    feature_mean = fit_features.mean(axis=0, keepdims=True)
    feature_scale = fit_features.std(axis=0, keepdims=True) + 1e-6
    fit_targets = targets[fit_rows].reshape(len(fit_rows), -1).astype(np.float64)
    target_mean = fit_targets.mean(axis=0, keepdims=True)
    fit_targets = fit_targets - target_mean

    # Gram and cross-product accumulated in float64 over chunks; centring the
    # targets and standardizing the features removes the need for an intercept.
    n_features = features.shape[1]
    gram = np.zeros((n_features, n_features))
    cross = np.zeros((n_features, fit_targets.shape[1]))
    for start in range(0, len(fit_rows), chunk_size):
        stop = min(start + chunk_size, len(fit_rows))
        block = (
            (fit_features[start:stop] - feature_mean) / feature_scale
        ).astype(np.float64)
        gram += block.T @ block
        cross += block.T @ fit_targets[start:stop]
    # end for chunk

    standardized_validation = (
        (features[validation_rows] - feature_mean) / feature_scale
    ).astype(np.float64)
    validation_targets = (
        targets[validation_rows].reshape(len(validation_rows), -1).astype(np.float64)
        - target_mean
    )

    validation_mses, weight_sets = [], []
    for alpha in alphas:
        penalized = gram + alpha * np.eye(n_features)
        weights = cho_solve(cho_factor(penalized, lower=True), cross)
        weight_sets.append(weights)
        validation_mses.append(
            float(
                np.mean((standardized_validation @ weights - validation_targets) ** 2)
            )
        )
    # end for candidate penalty

    best = int(np.argmin(validation_mses))
    best_weights = weight_sets[best]

    def predict(block):
        standardized = ((block - feature_mean) / feature_scale).astype(np.float64)
        return (standardized @ best_weights + target_mean).reshape(
            len(block), n_time, n_channels
        )
    # end predict

    return {
        "predict": predict,
        "alpha": float(alphas[best]),
        "validation_mse": validation_mses[best],
        "validation_mse_per_alpha": validation_mses,
        "alpha_at_grid_edge": bool(best in (0, len(alphas) - 1)),
    }
# EOF


"""
single_trial_reliability
Correlate one repetition of each test image against a disjoint repetition.

This is the ceiling a single-trial score can reach, and it is what makes the
held-out-train and test single-trial numbers comparable: if the two splits carry
the same trial noise, the same decoder should reach the same fraction of it.

INPUT:
    - test_responses: np.ndarray -> [presentations, time, channels]
    - test_image_ids: np.ndarray -> zero-based image id per presentation
    - n_images: int -> size of the repeated-test pool
    - n_resamples: int -> repetition pairs drawn per image
    - seed: int -> resampling seed

OUTPUT:
    - reliability: float -> mean over (time, channel) of the rep-to-rep stim_r
"""
def single_trial_reliability(
    test_responses, test_image_ids, n_images, n_resamples=20, seed=0
):
    rng = np.random.default_rng(seed)
    correlations = []
    for _ in range(n_resamples):
        first, second = [], []
        for image_id in range(n_images):
            rows = np.flatnonzero(test_image_ids == image_id)
            picks = rng.permutation(rows)[:2]
            first.append(test_responses[picks[0]])
            second.append(test_responses[picks[1]])
        # end for test image
        correlations.append(
            np.nanmean(stimulus_correlation(np.stack(first), np.stack(second)))
        )
    # end for resample
    return float(np.mean(correlations))
# EOF


"""
matched_holdout_stim_r
Score one ridge map on held-out training images and on the test images under
identical single-trial conditions.

Everything except the identity of the images is matched: the same map, the same
number of images, one presentation each. A drop from the held-out training
images to the test images is therefore the distribution shift, and nothing else.
The repetition-averaged test score is reported next to it as the part that
averaging explains rather than any shift.

INPUT:
    - train_design: np.ndarray -> [train presentations, p] features
    - train_response: np.ndarray -> [train presentations, time, channels]
    - test_design: np.ndarray -> [test presentations, p] features
    - test_response: np.ndarray -> [test presentations, time, channels]
    - test_image_ids: np.ndarray -> zero-based image id per test presentation
    - n_images: int -> images held out, matched to the test pool size
    - n_resamples: int -> hold-out groups drawn
    - validation_fraction: float -> rows reserved for choosing the penalty
    - alphas: np.ndarray -> ridge penalties compared
    - seed: int -> resampling seed

OUTPUT:
    - report: dict -> per-resample and mean stim_r for the three evaluations,
      plus the selected penalties
"""
def matched_holdout_stim_r(
    train_design,
    train_response,
    test_design,
    test_response,
    test_image_ids,
    n_images=100,
    n_resamples=5,
    validation_fraction=0.1,
    alphas=None,
    seed=0,
):
    alphas = np.logspace(1.0, 9.0, 9) if alphas is None else np.asarray(alphas)
    rng = np.random.default_rng(seed)
    rows = []
    for resample in range(n_resamples):
        shuffled = rng.permutation(len(train_design))
        holdout_rows = shuffled[:n_images]
        remaining = shuffled[n_images:]
        n_validation = max(1, round(len(remaining) * validation_fraction))
        validation_rows, fit_rows = remaining[:n_validation], remaining[n_validation:]

        ridge_fit = fit_ridge_gram(
            train_design, train_response, fit_rows, validation_rows, alphas
        )
        # Held-out training images: one presentation each, exactly as recorded.
        holdout_stim_r = float(
            np.nanmean(
                stimulus_correlation(
                    ridge_fit["predict"](train_design[holdout_rows]),
                    train_response[holdout_rows],
                )
            )
        )

        # Test images, matched: one randomly chosen repetition per image.
        single_rep_rows = np.array(
            [
                rng.choice(np.flatnonzero(test_image_ids == image_id))
                for image_id in range(n_images)
            ]
        )
        test_single_stim_r = float(
            np.nanmean(
                stimulus_correlation(
                    ridge_fit["predict"](test_design[single_rep_rows]),
                    test_response[single_rep_rows],
                )
            )
        )

        # Test images as usually scored: every repetition averaged.
        test_predictions = ridge_fit["predict"](test_design)
        test_averaged_stim_r = float(
            np.nanmean(
                stimulus_correlation(
                    aggregate_trials_by_image(
                        test_predictions, test_image_ids, n_images
                    ),
                    aggregate_trials_by_image(
                        test_response, test_image_ids, n_images
                    ),
                )
            )
        )
        rows.append(
            {
                "resample": resample,
                "alpha": ridge_fit["alpha"],
                "alpha_at_grid_edge": ridge_fit["alpha_at_grid_edge"],
                "heldout_train_single_trial": holdout_stim_r,
                "test_single_trial": test_single_stim_r,
                "test_repetition_averaged": test_averaged_stim_r,
            }
        )
    # end for resample

    summary = {
        key: float(np.mean([row[key] for row in rows]))
        for key in (
            "heldout_train_single_trial",
            "test_single_trial",
            "test_repetition_averaged",
        )
    }
    summary["shift_cost_single_trial"] = (
        summary["heldout_train_single_trial"] - summary["test_single_trial"]
    )
    summary["sd_heldout_train_single_trial"] = float(
        np.std([row["heldout_train_single_trial"] for row in rows])
    )
    summary["sd_test_single_trial"] = float(
        np.std([row["test_single_trial"] for row in rows])
    )
    return {"rows": rows, "summary": summary}
# EOF
