import sys
from pathlib import Path

import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python_scripts" / "src"))

from IT_recap.tvsd_stacking import (  # noqa: E402
    average_stacked_predictions,
    build_ridge_stack,
    cross_fit_layer_ridge,
    fit_ridge_layer_map,
)
from model_classes.timebin_models import build_timebin_model  # noqa: E402


N_TIMEPOINTS, N_NEURONS = 3, 4


"""
make_linear_problem
Build a small dataset whose response is a fixed linear function of features.

INPUT:
    - n_presentations: int -> rows per split
    - n_layers: int -> feature groups
    - embedding: int -> width of each group

OUTPUT:
    - subset_features: dict -> split name to [presentations, layers, embedding]
    - subset_targets: dict -> split name to [presentations, time, neurons]
"""
def make_linear_problem(n_presentations=120, n_layers=2, embedding=5):
    rng = np.random.default_rng(0)
    weights = rng.standard_normal((n_layers * embedding, N_TIMEPOINTS * N_NEURONS))
    subset_features, subset_targets = {}, {}
    for split_name, split_size in (
        ("train", n_presentations),
        ("validation", n_presentations // 3),
        ("test", n_presentations // 4),
    ):
        features = rng.standard_normal((split_size, n_layers, embedding))
        responses = features.reshape(split_size, -1) @ weights
        responses = responses + 0.1 * rng.standard_normal(responses.shape)
        subset_features[split_name] = features.astype(np.float32)
        subset_targets[split_name] = responses.reshape(
            split_size, N_TIMEPOINTS, N_NEURONS
        ).astype(np.float32)
    # end for data split
    return subset_features, subset_targets
# EOF


def test_layer_map_predicts_every_requested_split():
    subset_features, subset_targets = make_linear_problem()
    predictions, alpha = fit_ridge_layer_map(
        subset_features["train"][:, 0],
        subset_targets["train"],
        {
            "validation": subset_features["validation"][:, 0],
            "test": subset_features["test"][:, 0],
        },
    )
    assert set(predictions) == {"validation", "test"}
    assert predictions["test"].shape == (
        len(subset_features["test"]),
        N_TIMEPOINTS,
        N_NEURONS,
    )
    assert alpha > 0
    # One of two independent feature groups explains part of the response, so
    # the map has to beat predicting the fit split's mean.
    fit_mean = subset_targets["train"].mean(axis=0)
    map_error = np.mean((predictions["test"] - subset_targets["test"]) ** 2)
    mean_error = np.mean((fit_mean - subset_targets["test"]) ** 2)
    assert map_error < mean_error
# EOF


def test_cross_fitting_predicts_the_fit_split_out_of_sample():
    subset_features, subset_targets = make_linear_problem()
    layer_features = {
        split_name: split_features[:, 0]
        for split_name, split_features in subset_features.items()
    }
    predictions, diagnostics = cross_fit_layer_ridge(
        layer_features, subset_targets, n_folds=3, seed=0
    )
    assert set(predictions) == {"train", "validation", "test"}
    assert predictions["train"].shape == subset_targets["train"].shape
    assert len(diagnostics["fold_alphas"]) == 3

    # The out-of-fold fit predictions must be worse than in-sample ones; that
    # gap is exactly the optimism the stacked decoder must not be trained on.
    in_sample, _ = fit_ridge_layer_map(
        layer_features["train"],
        subset_targets["train"],
        {"train": layer_features["train"]},
    )
    out_of_fold_error = np.mean(
        (predictions["train"] - subset_targets["train"]) ** 2
    )
    in_sample_error = np.mean((in_sample["train"] - subset_targets["train"]) ** 2)
    assert out_of_fold_error > in_sample_error
# EOF


def test_cross_fitting_rejects_a_single_fold():
    subset_features, subset_targets = make_linear_problem()
    layer_features = {
        split_name: split_features[:, 0]
        for split_name, split_features in subset_features.items()
    }
    with pytest.raises(ValueError, match="at least two"):
        cross_fit_layer_ridge(layer_features, subset_targets, n_folds=1, seed=0)
# EOF


def test_stack_has_one_flattened_response_map_per_depth():
    subset_features, subset_targets = make_linear_problem(n_layers=3)
    stacked, diagnostics = build_ridge_stack(
        subset_features, subset_targets, n_folds=2, seed=0, verbose=False
    )
    assert len(diagnostics) == 3
    assert [entry["layer"] for entry in diagnostics] == [0, 1, 2]
    for split_name, split_features in subset_features.items():
        assert stacked[split_name].shape == (
            len(split_features),
            3,
            N_TIMEPOINTS * N_NEURONS,
        )
    # end for data split

    # A depth's row of the stack is that depth's own ridge prediction, so it
    # must reproduce the map fitted on the fit split alone.
    predictions, _ = fit_ridge_layer_map(
        subset_features["train"][:, 1],
        subset_targets["train"],
        {"test": subset_features["test"][:, 1]},
    )
    assert np.allclose(
        stacked["test"][:, 1],
        predictions["test"].reshape(len(predictions["test"]), -1),
        atol=1e-5,
    )
# EOF


def test_average_stacked_predictions_restores_the_response_shape():
    stacked_split = np.arange(2 * 3 * N_TIMEPOINTS * N_NEURONS, dtype=np.float32)
    stacked_split = stacked_split.reshape(2, 3, N_TIMEPOINTS * N_NEURONS)
    averaged = average_stacked_predictions(
        stacked_split, N_TIMEPOINTS, N_NEURONS
    )
    assert averaged.shape == (2, N_TIMEPOINTS, N_NEURONS)
    assert np.allclose(
        averaged[0, 0, 0], stacked_split[0, :, 0].mean()
    )
# EOF


def test_the_stacked_decoder_consumes_the_stack_without_normalizing_it():
    subset_features, subset_targets = make_linear_problem(n_layers=3)
    stacked, _ = build_ridge_stack(
        subset_features, subset_targets, n_folds=2, seed=0, verbose=False
    )
    model = build_timebin_model(
        "baseline",
        n_layers=3,
        feature_dim=N_TIMEPOINTS * N_NEURONS,
        n_timepoints=N_TIMEPOINTS,
        n_neurons=N_NEURONS,
        time_embedding_dim=6,
        value_dim=8,
        mlp_hidden_dim=8,
        dropout=0.0,
        temporal_noise_std=0.25,
        normalize_features=False,
    )
    # Off, the decoder sees the maps in the target's own units; the LayerNorm
    # would divide each image's predicted response magnitude away.
    assert isinstance(model.feature_norm, torch.nn.Identity)
    inputs = torch.from_numpy(stacked["test"])
    predictions, _ = model(inputs)
    assert predictions.shape == (len(stacked["test"]), N_TIMEPOINTS, N_NEURONS)
    assert torch.isfinite(predictions).all()
# EOF


def test_normalization_stays_on_by_default():
    model = build_timebin_model(
        "gru",
        n_layers=2,
        feature_dim=6,
        n_timepoints=N_TIMEPOINTS,
        n_neurons=N_NEURONS,
        hidden_dim=8,
        time_embedding_dim=4,
        dropout=0.0,
    )
    assert isinstance(model.feature_norm, torch.nn.LayerNorm)
# EOF
