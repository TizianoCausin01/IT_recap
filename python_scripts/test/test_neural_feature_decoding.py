import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python_scripts" / "src"))

from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from IT_recap.neural_feature_decoding import (  # noqa: E402
    NeuralLayerFeatureDataset,
    collect_feature_decoder_outputs,
    fit_feature_decoder,
    fit_layer_pcas,
    fit_neural_channel_standardizer,
    layer_balanced_mse_loss,
    score_layer_predictions,
)
from model_classes.temporal_models import (  # noqa: E402
    NeuralToLayerFeatureGRUModel,
    NoiseStateGRUFeatureModel,
    VariationalGRUEncoder,
)


def make_noise_state_model(noise_in_eval):
    return NoiseStateGRUFeatureModel(
        n_neural_channels=7,
        target_dims=[5, 4, 3],
        hidden_dim=8,
        hidden_dropout=0.0,
        head_dropout=0.0,
        variational_dropout=0.0,
        noise_in_eval=noise_in_eval,
    )
# EOF


def test_noise_state_model_attends_over_time_plus_one_noise_item():
    torch.manual_seed(0)
    model = make_noise_state_model(noise_in_eval=True).eval()
    neural_sequences = torch.randn(4, 9, 7)
    predictions, attention, hidden_states = model(
        neural_sequences,
        return_hidden_states=True,
    )

    assert predictions.shape == (4, 12)
    assert attention.shape == (4, 3, 10)
    assert hidden_states.shape == (4, 9, 8)
    torch.testing.assert_close(attention.sum(dim=-1), torch.ones(4, 3))

    # The appended noise state has the mean norm of the real GRU states.
    noise_state = model._build_attended_states(hidden_states)[:, -1]
    torch.testing.assert_close(
        noise_state.norm(dim=-1),
        hidden_states.norm(dim=-1).mean(dim=1),
    )

    # Fresh noise changes predictions, so evaluation is stochastic.
    second_predictions, _ = model(neural_sequences)
    assert not torch.allclose(predictions, second_predictions)
# EOF


def test_noise_state_model_is_deterministic_when_noise_is_off_in_eval():
    torch.manual_seed(0)
    model = make_noise_state_model(noise_in_eval=False).eval()
    neural_sequences = torch.randn(4, 9, 7)
    first_predictions, first_attention = model(neural_sequences)
    second_predictions, _ = model(neural_sequences)

    torch.testing.assert_close(first_predictions, second_predictions)
    assert first_attention.shape == (4, 3, 10)
# EOF


def test_fit_feature_decoder_records_callback_metrics_every_epoch():
    torch.manual_seed(0)
    model = make_noise_state_model(noise_in_eval=False)
    loader = DataLoader(
        TensorDataset(torch.randn(8, 9, 7), torch.randn(8, 12)),
        batch_size=4,
    )
    seen_epochs = []

    def epoch_callback(model, epoch):
        seen_epochs.append(epoch)
        return {"epoch_squared": float(epoch ** 2)}
    # EOF

    history = fit_feature_decoder(
        model,
        loader,
        loader,
        [slice(0, 5), slice(5, 9), slice(9, 12)],
        learning_rate=1e-3,
        weight_decay=0.0,
        max_epochs=3,
        patience=10,
        epoch_callback=epoch_callback,
    )

    # Epoch 0 is the untrained model, followed by every completed epoch.
    assert seen_epochs == [0, 1, 2, 3]
    assert history["epoch_squared"] == [0.0, 1.0, 4.0, 9.0]
# EOF


def test_collect_outputs_averages_repeated_noise_draws():
    torch.manual_seed(0)
    model = make_noise_state_model(noise_in_eval=True)
    loader = DataLoader(
        TensorDataset(torch.randn(6, 9, 7), torch.randn(6, 12)),
        batch_size=4,
    )
    predictions, targets, attention, hidden_states = (
        collect_feature_decoder_outputs(model, loader, n_repeats=5)
    )

    assert predictions.shape == targets.shape == (6, 12)
    assert attention.shape == (6, 3, 10)
    assert hidden_states.shape == (6, 9, 8)
    np.testing.assert_allclose(attention.sum(axis=-1), 1.0, rtol=1e-5)
# EOF


def test_neural_to_layer_model_preserves_all_states_and_separate_attention():
    torch.manual_seed(0)
    model = NeuralToLayerFeatureGRUModel(
        n_neural_channels=7,
        target_dims=[5, 4, 3],
        target_names=["shallow", "mid", "deep"],
        hidden_dim=8,
        bottleneck_dim=6,
        variational_dropout=0.1,
        hidden_dropout=0.0,
        head_dropout=0.0,
        head_type="linear",
    )
    neural_sequences = torch.randn(4, 9, 7)
    predictions, attention, hidden_states = model(
        neural_sequences,
        return_hidden_states=True,
    )

    assert predictions.shape == (4, 12)
    assert attention.shape == (4, 3, 9)
    assert hidden_states.shape == (4, 9, 8)
    torch.testing.assert_close(
        attention.sum(dim=-1),
        torch.ones(4, 3),
    )
    assert model.layer_queries.shape == (3, 8)
# EOF


def test_variational_gru_is_deterministic_only_in_evaluation_mode():
    encoder = VariationalGRUEncoder(
        input_dim=5,
        hidden_dim=8,
        n_layers=1,
        variational_dropout=0.5,
    )
    sequence = torch.randn(16, 7, 5)

    encoder.train()
    first_training_output = encoder(sequence)
    second_training_output = encoder(sequence)
    assert not torch.allclose(first_training_output, second_training_output)

    encoder.eval()
    first_evaluation_output = encoder(sequence)
    second_evaluation_output = encoder(sequence)
    torch.testing.assert_close(first_evaluation_output, second_evaluation_output)
# EOF


def test_layer_pca_and_dataset_use_expected_alignment():
    rng = np.random.default_rng(0)
    layer_features = rng.normal(size=(10, 3, 12)).astype(np.float32)
    image_targets, pcas, target_slices = fit_layer_pcas(
        layer_features,
        training_image_indices=np.arange(8),
        n_components=4,
        random_seed=0,
    )
    assert image_targets.shape == (10, 12)
    assert len(pcas) == 3
    assert target_slices == [slice(0, 4), slice(4, 8), slice(8, 12)]

    neural_activity = rng.normal(size=(6, 5, 20)).astype(np.float32)
    image_indices = np.repeat(np.arange(10), 2)
    training_trial_indices = np.arange(16)
    channel_mean, channel_std = fit_neural_channel_standardizer(
        neural_activity,
        training_trial_indices,
    )
    dataset = NeuralLayerFeatureDataset(
        neural_activity,
        image_targets,
        image_indices,
        channel_mean,
        channel_std,
    )
    neural_sequence, target = dataset[3]
    assert neural_sequence.shape == (5, 6)
    torch.testing.assert_close(target, torch.as_tensor(image_targets[1]))
# EOF


def test_layer_scores_are_perfect_for_exact_image_predictions():
    rng = np.random.default_rng(0)
    image_targets = rng.normal(size=(6, 8))
    image_indices = np.repeat(np.arange(6), 3)
    targets = image_targets[image_indices]
    scores = score_layer_predictions(
        targets,
        targets,
        image_indices,
        [slice(0, 5), slice(5, 8)],
    )

    for score_name in ("r2", "feature_r", "rdm_rho"):
        np.testing.assert_allclose(scores[score_name], [1.0, 1.0])
    # end for score type
# EOF


def test_layer_balanced_loss_does_not_weight_layers_by_width():
    predictions = torch.tensor([[2.0, 2.0, 2.0, 4.0]])
    targets = torch.zeros_like(predictions)
    loss = layer_balanced_mse_loss(
        predictions,
        targets,
        [slice(0, 3), slice(3, 4)],
    )

    # First-layer MSE is 4 and second-layer MSE is 16; layers average to 10.
    torch.testing.assert_close(loss, torch.tensor(10.0))
# EOF
