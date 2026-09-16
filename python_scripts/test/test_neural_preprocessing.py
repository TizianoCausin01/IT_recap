from types import SimpleNamespace

import numpy as np

from IT_recap.tvsd_experiments import preprocess_neural_targets_on_fit_split
from project_specific_utils.dataloader import (
    apply_neural_preprocessing,
    fit_neural_preprocessing,
)


def test_center_and_robust_minmax_use_only_fitting_samples():
    neural_activity = np.asarray(
        [
            [[0.0, 2.0, 4.0, 100.0], [10.0, 14.0, 18.0, 200.0]],
            [[-4.0, 0.0, 4.0, -100.0], [2.0, 4.0, 6.0, -200.0]],
        ],
        dtype=np.float32,
    )
    fitting_indices = np.asarray([0, 1, 2])

    stats = fit_neural_preprocessing(
        neural_activity,
        fitting_sample_indices=fitting_indices,
        center_features=True,
        robust_minmax_neurons=True,
        robust_percentile_range=(0.0, 100.0),
        clip_robust_minmax=True,
    )
    processed = apply_neural_preprocessing(neural_activity, stats)

    expected_feature_mean = neural_activity[:, :, fitting_indices].mean(
        axis=2,
        keepdims=True,
    )
    np.testing.assert_allclose(stats.feature_mean, expected_feature_mean)
    np.testing.assert_allclose(processed[:, :, fitting_indices].min(axis=(1, 2)), 0)
    np.testing.assert_allclose(processed[:, :, fitting_indices].max(axis=(1, 2)), 1)
    np.testing.assert_allclose(processed[:, :, 3], [[1, 1], [0, 0]])


def test_disabled_preprocessing_returns_an_independent_float32_array():
    neural_activity = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    stats = fit_neural_preprocessing(neural_activity)

    processed = apply_neural_preprocessing(neural_activity, stats)

    np.testing.assert_array_equal(processed, neural_activity)
    assert processed.dtype == np.float32
    assert not np.shares_memory(processed, neural_activity)


def test_tvsd_preprocessing_reuses_training_statistics_for_every_split():
    training_activity = np.asarray(
        [
            [[0.0, 2.0, 4.0], [10.0, 14.0, 18.0]],
            [[-4.0, 0.0, 4.0], [2.0, 4.0, 6.0]],
        ],
        dtype=np.float32,
    )
    validation_activity = np.asarray(
        [[[100.0], [200.0]], [[-100.0], [-200.0]]],
        dtype=np.float32,
    )
    subset_targets = {
        "train": training_activity.transpose(2, 1, 0),
        "validation": validation_activity.transpose(2, 1, 0),
    }
    cfg = SimpleNamespace(
        center_neural_features=True,
        robust_minmax_neurons=True,
        robust_percentile_range=(0.0, 100.0),
        clip_robust_minmax=True,
    )

    processed, stats = preprocess_neural_targets_on_fit_split(
        subset_targets,
        cfg,
    )

    expected_feature_mean = training_activity.mean(axis=2, keepdims=True)
    np.testing.assert_allclose(stats.feature_mean, expected_feature_mean)
    assert processed["train"].shape == subset_targets["train"].shape
    assert processed["validation"].shape == subset_targets["validation"].shape
    np.testing.assert_allclose(processed["validation"][0, :, 0], 1.0)
    np.testing.assert_allclose(processed["validation"][0, :, 1], 0.0)
