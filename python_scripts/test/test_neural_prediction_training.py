import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from IT_recap.neural_prediction_training import (
    collect_neural_predictions,
    collect_image_level_predictions,
    mean_stimulus_correlation,
    minimum_repetition_weighted_mse,
    neural_activity_timebin_mse_loss,
    neural_activity_weighted_mse_loss,
    minimum_repetition_test_step,
)


def test_collect_neural_predictions_preserves_loader_order():
    class OffsetPredictionModel(torch.nn.Module):
        def forward(self, inputs, use_precomputed_features):
            assert use_precomputed_features
            return inputs + 1.0, None
        # EOF

    inputs = torch.arange(6, dtype=torch.float32).reshape(3, 2, 1)
    targets = -inputs
    loader = DataLoader(
        TensorDataset(inputs, targets),
        batch_size=2,
        shuffle=False,
    )

    predictions, collected_targets = collect_neural_predictions(
        OffsetPredictionModel(),
        loader,
        use_precomputed_features=True,
    )

    np.testing.assert_array_equal(predictions, (inputs + 1.0).numpy())
    np.testing.assert_array_equal(collected_targets, targets.numpy())


def test_mean_stimulus_correlation_excludes_flat_cells():
    targets = np.array(
        [
            [[0.0, 1.0]],
            [[1.0, 1.0]],
            [[2.0, 1.0]],
        ]
    )
    predictions = np.array(
        [
            [[0.0, 4.0]],
            [[2.0, 4.0]],
            [[4.0, 4.0]],
        ]
    )

    # The first cell is perfectly correlated; the second is flat and omitted.
    assert mean_stimulus_correlation(predictions, targets) == pytest.approx(1.0)


def test_collect_image_level_predictions_averages_repetitions():
    class IdentityPredictionModel(torch.nn.Module):
        def forward(self, inputs, use_precomputed_features):
            return inputs, None
        # EOF

    predictions = torch.tensor([[[1.0]], [[5.0]], [[3.0]]])
    targets = torch.tensor([[[2.0]], [[8.0]], [[4.0]]])
    loader = DataLoader(
        TensorDataset(predictions, targets),
        batch_size=2,
        shuffle=False,
    )

    image_predictions, image_targets = collect_image_level_predictions(
        IdentityPredictionModel(),
        loader,
        use_precomputed_features=True,
        image_ids=np.array([9, 4, 9]),
    )

    # np.unique sorts IDs: image 4 is row 0; repeated image 9 is row 1.
    np.testing.assert_array_equal(image_predictions[:, 0, 0], [5.0, 2.0])
    np.testing.assert_array_equal(image_targets[:, 0, 0], [8.0, 3.0])


def test_uniform_weighted_mse_matches_plain_mse():
    predictions = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 0.0], [1.0, 5.0]],
        ]
    )
    targets = torch.zeros_like(predictions)
    weights = torch.ones(2, 2)

    weighted_loss = neural_activity_weighted_mse_loss(
        predictions, targets, weights
    )
    plain_loss = neural_activity_timebin_mse_loss(predictions, targets)

    torch.testing.assert_close(weighted_loss, plain_loss)


def test_weighted_mse_selects_requested_time_neuron_cell():
    predictions = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[3.0, 6.0], [5.0, 8.0]],
        ]
    )
    targets = torch.zeros_like(predictions)
    weights = torch.tensor([[0.0, 1.0], [0.0, 0.0]])

    loss = neural_activity_weighted_mse_loss(predictions, targets, weights)

    # Only neuron 1 at time 0 contributes: mean([2^2, 6^2]) = 20.
    torch.testing.assert_close(loss, torch.tensor(20.0))


@pytest.mark.parametrize(
    "weights",
    [
        torch.ones(3, 2),
        torch.tensor([[1.0, -1.0], [1.0, 1.0]]),
        torch.zeros(2, 2),
    ],
)
def test_weighted_mse_rejects_invalid_weights(weights):
    predictions = torch.zeros(2, 2, 2)

    with pytest.raises(ValueError):
        neural_activity_weighted_mse_loss(predictions, predictions, weights)


def test_minimum_repetition_mse_selects_one_whole_trial_per_image():
    predictions = np.array(
        [
            [[0.0, 2.0]],
            [[2.0, 0.0]],
            [[1.0, 1.0]],
            [[0.0, 4.0]],
            [[4.0, 0.0]],
        ]
    )
    targets = np.zeros_like(predictions)
    image_ids = np.array([7, 7, 7, 9, 9])
    weights = np.array([[1.0, 3.0]])

    mse = minimum_repetition_weighted_mse(
        predictions,
        targets,
        image_ids,
        weights,
    )

    # Image 7 has minimum trial MSE 1; image 9 has minimum 4. Each image
    # contributes once even though image 7 has an extra repetition.
    assert mse == pytest.approx(2.5)


def test_minimum_repetition_test_step_groups_across_batches():
    class IdentityPredictionModel(torch.nn.Module):
        def forward(self, inputs, use_precomputed_features):
            return inputs, None
        # EOF

    predictions = torch.tensor(
        [
            [[3.0]],
            [[4.0]],
            [[1.0]],
            [[2.0]],
        ]
    )
    targets = torch.zeros_like(predictions)
    loader = DataLoader(
        TensorDataset(predictions, targets),
        batch_size=2,
        shuffle=False,
    )

    mse = minimum_repetition_test_step(
        IdentityPredictionModel(),
        loader,
        use_precomputed_features=True,
        image_ids=np.array([0, 1, 0, 1]),
        weights=np.ones((1, 1)),
    )

    # The best repetitions are in the second batch: 1^2 and 2^2.
    assert mse == pytest.approx(2.5)
