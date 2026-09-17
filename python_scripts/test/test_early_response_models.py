import sys
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python_scripts" / "src"))

from model_classes.early_response_models import (  # noqa: E402
    EARLY_RESPONSE_MODEL_CLASSES,
    EarlyResponseEncoder,
    build_early_response_model,
)
from model_classes.timebin_models import build_timebin_model  # noqa: E402
from IT_recap.tvsd_experiments import (  # noqa: E402
    collect_l2_parameters,
    ridge_equivalent_l2_lambda,
)
from IT_recap.neural_prediction_training import (  # noqa: E402
    optimally_rescaled_mse,
    population_pattern_correlation,
)


SHAPES = {
    "n_layers": 4,
    "feature_dim": 16,
    "n_timepoints": 5,
    "n_neurons": 7,
}
N_EARLY_BINS = 5

# One minimal set of width arguments per early-response architecture, and the
# parent arguments they have to stay compatible with.
ARCHITECTURE_KWARGS = {
    "gru": {"hidden_dim": 12, "time_embedding_dim": 6},
    "lds": {"state_dim": 9},
}


def make_model(architecture, **overrides):
    """Build one early-response decoder with its minimal width arguments."""
    return build_early_response_model(
        architecture,
        **SHAPES,
        **ARCHITECTURE_KWARGS[architecture],
        n_early_bins=N_EARLY_BINS,
        early_dim=8,
        **overrides,
    )
# EOF


def make_inputs(batch_size=4):
    """Cached features and an observed early trace for one batch."""
    return (
        torch.randn(batch_size, SHAPES["n_layers"], SHAPES["feature_dim"]),
        torch.randn(batch_size, N_EARLY_BINS, SHAPES["n_neurons"]),
    )
# EOF


@pytest.mark.parametrize("architecture", sorted(EARLY_RESPONSE_MODEL_CLASSES))
def test_every_architecture_predicts_all_time_bins(architecture):
    model = make_model(architecture)
    predictions, diagnostics = model(*make_inputs())
    assert predictions.shape == (4, SHAPES["n_timepoints"], SHAPES["n_neurons"])
    assert torch.isfinite(predictions).all()
    assert diagnostics is None
# EOF


@pytest.mark.parametrize("architecture", sorted(EARLY_RESPONSE_MODEL_CLASSES))
def test_diagnostics_expose_the_early_code(architecture):
    model = make_model(architecture)
    _, diagnostics = model(*make_inputs(), return_diagnostics=True)
    assert diagnostics["early_code"].shape == (4, 8)
# EOF


@pytest.mark.parametrize("architecture", sorted(EARLY_RESPONSE_MODEL_CLASSES))
def test_the_early_trace_changes_the_prediction(architecture):
    model = make_model(architecture).eval()
    features, early_response = make_inputs()
    with torch.no_grad():
        predictions = model(features, early_response)[0]
        other_predictions = model(features, torch.randn_like(early_response))[0]
    # end with no gradient tracking
    # The trace only sets the initial condition, so late bins may converge;
    # the trajectory as a whole must still depend on it.
    assert not torch.allclose(predictions, other_predictions, atol=1e-5)
# EOF


@pytest.mark.parametrize("architecture", sorted(EARLY_RESPONSE_MODEL_CLASSES))
def test_the_ablation_ignores_the_early_trace(architecture):
    model = make_model(architecture, use_early_response=False).eval()
    features, early_response = make_inputs()
    with torch.no_grad():
        predictions = model(features, early_response)[0]
        other_predictions = model(features, torch.randn_like(early_response))[0]
    # end with no gradient tracking
    assert torch.allclose(predictions, other_predictions)
    assert model.early_encoder is None
    # No parameter of the ablation reads the trace it was handed.
    assert not any(
        "early" in name for name, _ in model.named_parameters()
    )
# EOF


@pytest.mark.parametrize("architecture", sorted(EARLY_RESPONSE_MODEL_CLASSES))
def test_the_ablation_matches_the_parent_decoder(architecture):
    """The no-early variant must be the searched decoder, parameter for parameter."""
    parent = build_timebin_model(
        architecture, **SHAPES, **ARCHITECTURE_KWARGS[architecture]
    )
    ablation = make_model(architecture, use_early_response=False)
    parent_shapes = {
        name: tuple(parameter.shape)
        for name, parameter in parent.named_parameters()
    }
    ablation_shapes = {
        name: tuple(parameter.shape)
        for name, parameter in ablation.named_parameters()
    }
    assert parent_shapes == ablation_shapes
# EOF


@pytest.mark.parametrize("architecture", sorted(EARLY_RESPONSE_MODEL_CLASSES))
def test_gradients_reach_the_early_encoder(architecture):
    model = make_model(architecture)
    predictions = model(*make_inputs())[0]
    predictions.square().mean().backward()
    gradient = model.early_encoder.projection.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0.0
# EOF


@pytest.mark.parametrize("architecture", sorted(EARLY_RESPONSE_MODEL_CLASSES))
def test_feature_noise_is_training_only(architecture):
    model = make_model(architecture, input_noise_std=0.5).eval()
    features, early_response = make_inputs()
    with torch.no_grad():
        first = model(features, early_response)[0]
        second = model(features, early_response)[0]
    # end with no gradient tracking
    assert torch.allclose(first, second)
# EOF


@pytest.mark.parametrize("architecture", sorted(EARLY_RESPONSE_MODEL_CLASSES))
def test_a_misshaped_early_trace_is_rejected(architecture):
    model = make_model(architecture)
    features, _ = make_inputs()
    with pytest.raises(ValueError):
        model(features, torch.randn(4, N_EARLY_BINS + 1, SHAPES["n_neurons"]))
    # end with the expected failure
# EOF


def test_the_encoder_rejects_invalid_dimensions():
    with pytest.raises(ValueError):
        EarlyResponseEncoder(0, 7, 8)
    # end with the expected failure
# EOF


def test_an_unknown_architecture_is_rejected():
    with pytest.raises(KeyError):
        build_early_response_model("transformer", **SHAPES, n_early_bins=5)
    # end with the expected failure
# EOF


@pytest.mark.parametrize("architecture", sorted(EARLY_RESPONSE_MODEL_CLASSES))
def test_ridge_like_parameters_are_weight_matrices(architecture):
    model = make_model(architecture)
    parameters = model.ridge_like_parameters()
    identities = {id(parameter) for parameter in parameters}
    assert len(identities) == len(parameters)
    # Ridge leaves its intercept unpenalized, so no bias may appear here.
    assert all(parameter.ndim == 2 for parameter in parameters)
    owned = {id(parameter) for parameter in model.parameters()}
    assert identities <= owned
# EOF


@pytest.mark.parametrize("architecture", sorted(EARLY_RESPONSE_MODEL_CLASSES))
def test_the_early_encoder_map_is_penalized_only_when_present(architecture):
    with_trace = make_model(architecture)
    without_trace = make_model(architecture, use_early_response=False)
    assert (
        len(with_trace.ridge_like_parameters())
        == len(without_trace.ridge_like_parameters()) + 1
    )
    assert any(
        parameter is with_trace.early_encoder.projection.weight
        for parameter in with_trace.ridge_like_parameters()
    )
# EOF


@pytest.mark.parametrize("architecture", sorted(EARLY_RESPONSE_MODEL_CLASSES))
def test_l2_scopes_select_nested_parameter_sets(architecture):
    model = make_model(architecture)
    readout = collect_l2_parameters(model, "readout")
    every = collect_l2_parameters(model, "all")
    assert collect_l2_parameters(model, "none") == []
    # "all" must cover "readout" and reach further into the dynamics.
    assert {id(p) for p in readout} < {id(p) for p in every}
    assert all(parameter.ndim > 1 for parameter in every)
# EOF


def test_an_unknown_l2_scope_is_rejected():
    model = make_model("lds")
    with pytest.raises(ValueError):
        collect_l2_parameters(model, "everything")
    # end with the expected failure
# EOF


def test_the_ridge_equivalent_lambda_rescales_a_summed_objective():
    # RidgeCV sums its squared error while the decoders average theirs, so the
    # equivalent penalty is the alpha divided by the number of averaged values.
    assert ridge_equivalent_l2_lambda(3.2e7, 20000, 5, 320) == pytest.approx(1.0)
    assert ridge_equivalent_l2_lambda(0.0, 10, 2, 3) == 0.0
# EOF


def test_population_pattern_correlation_is_perfect_on_a_copy():
    rng = np.random.default_rng(0)
    targets = rng.normal(size=(40, 5, 32))
    correlations = population_pattern_correlation(targets, targets)
    assert correlations.shape == (40, 5)
    assert np.allclose(correlations, 1.0)
# EOF


def test_centring_removes_a_shared_per_site_profile():
    """The per-site profile inflates an uncentred score; centring strips it."""
    rng = np.random.default_rng(1)
    # A large profile shared by every image, plus a small image-specific part.
    site_profile = rng.normal(size=(1, 5, 32)) * 5.0
    deviation = rng.normal(size=(40, 5, 32))
    targets = site_profile + deviation
    # The prediction reproduces the profile exactly, and carries a third of the
    # image-specific signal buried in independent error. A scaled *copy* of the
    # signal would correlate at exactly 1.0, correlation being scale-free, so
    # the error has to be independent for the score to be informative.
    predictions = (
        site_profile + 0.33 * deviation + 0.9 * rng.normal(size=(40, 5, 32))
    )

    raw = float(
        np.nanmean(
            population_pattern_correlation(
                predictions, targets, center_across_images=False
            )
        )
    )
    centred = float(
        np.nanmean(
            population_pattern_correlation(
                predictions, targets, center_across_images=True
            )
        )
    )
    # Uncentred, the shared profile alone carries the score close to one.
    assert raw > 0.95
    # Centred, only the image-specific agreement survives: 0.33 of signal
    # against 0.9 of error is a correlation near 0.33 / sqrt(0.33^2 + 0.9^2).
    assert 0.25 < centred < 0.45
    assert raw - centred > 0.5
# EOF


def test_population_pattern_correlation_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        population_pattern_correlation(
            np.zeros((4, 5, 6)), np.zeros((4, 5, 7))
        )
    # end with the expected failure
# EOF


def test_rescaling_removes_a_pure_gain_error():
    """A prediction that is only mis-scaled has no error left after rescaling."""
    rng = np.random.default_rng(2)
    targets = rng.normal(size=(60, 5, 16))
    # Shrunk by half and offset: the shape is perfect, the scale is not.
    predictions = 0.5 * targets + 3.0

    plain = float(np.mean((predictions - targets) ** 2))
    rescaled = optimally_rescaled_mse(predictions, targets)
    assert plain > 9.0
    assert rescaled < 1e-9
# EOF


def test_rescaling_cannot_help_an_uninformative_prediction():
    rng = np.random.default_rng(3)
    targets = rng.normal(size=(60, 5, 16))
    predictions = rng.normal(size=(60, 5, 16))
    rescaled = optimally_rescaled_mse(predictions, targets)
    # The best a useless prediction can do is the target's own variance.
    assert rescaled == pytest.approx(targets.var(axis=0).mean(), rel=0.1)
# EOF


def build_early_attention_gru(n_early_bins=3, **noise_kwargs):
    from model_classes.early_response_models import (
        EarlyResponseShowAttendTellGRUModel,
    )

    torch.manual_seed(0)
    return EarlyResponseShowAttendTellGRUModel(
        None,
        layers=["layer_a", "layer_b"],
        n_timepoints=4,
        n_neurons=5,
        hidden_dim=8,
        attention_dim=6,
        n_early_bins=n_early_bins,
        early_dim=7,
        encoder_dim=16,
        **noise_kwargs,
    )
# EOF


def test_early_attention_gru_shapes_and_initial_state_reads_early_bins():
    model = build_early_attention_gru().eval()
    features = torch.randn(2, 2, 16)
    early_response = torch.randn(2, 3, 5)

    predictions, attention, hidden = model(
        features,
        early_response,
        use_precomputed_features=True,
        return_hidden_states=True,
    )
    assert predictions.shape == (2, 4, 5)
    assert attention.shape == (2, 4, 2, 16)
    assert hidden.shape == (2, 4, 8)

    # A different pre-response state must change the predicted trajectory.
    shifted_predictions, _ = model(
        features,
        torch.randn(2, 3, 5),
        use_precomputed_features=True,
    )
    assert not torch.allclose(predictions, shifted_predictions)
# EOF


def test_early_response_dataset_and_helpers_accept_three_tensor_batches():
    from torch.utils.data import DataLoader

    from IT_recap.dynamic_drsa import collect_gru_time_series
    from IT_recap.neural_prediction_training import (
        collect_concatenated_layer_regression_data,
        collect_neural_predictions,
        neural_activity_weighted_mse_loss,
        training_step,
    )
    from project_specific_utils.dataloader import (
        EarlyResponseInputDataset,
        NeuralInputDataset,
    )

    # 6 trials over 3 images; full window of 7 bins = 3 early + 4 target.
    neural_activity = np.random.default_rng(0).normal(size=(5, 7, 6))
    full_dataset = NeuralInputDataset(
        image_dataset=[None] * 3,
        activations=np.random.default_rng(1).normal(size=(3, 2, 16)),
        neural_activity=neural_activity,
        image_indices=[0, 1, 2, 0, 1, 2],
        input_mode="activations",
    )
    dataset = EarlyResponseInputDataset(full_dataset, n_early_bins=3)
    features, early_response, target = dataset[4]
    assert features.shape == (2, 16)
    assert early_response.shape == (3, 5)
    assert target.shape == (4, 5)
    np.testing.assert_allclose(
        early_response.numpy(), neural_activity[:, :3, 4].T, rtol=1e-6
    )
    np.testing.assert_allclose(
        target.numpy(), neural_activity[:, 3:, 4].T, rtol=1e-6
    )

    model = build_early_attention_gru()
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    loss = training_step(
        model,
        loader,
        optimizer,
        partial(neural_activity_weighted_mse_loss, weights=torch.ones(4, 5)),
        use_precomputed_features=True,
    )
    assert np.isfinite(loss)

    predictions, targets = collect_neural_predictions(model, loader, True)
    assert predictions.shape == targets.shape == (6, 4, 5)
    _, hidden, _ = collect_gru_time_series(model, loader, True)
    assert hidden.shape == (6, 4, 8)

    image_only, _ = collect_concatenated_layer_regression_data(
        model, loader, True
    )
    with_early, flat_targets = collect_concatenated_layer_regression_data(
        model, loader, True, append_extra_inputs=True
    )
    assert image_only.shape == (6, 32)
    assert with_early.shape == (6, 32 + 15)
    assert flat_targets.shape == (6, 20)
# EOF


def test_early_attention_gru_noise_layer_adds_one_attended_item():
    features = torch.randn(2, 2, 16)
    early_response = torch.randn(2, 3, 5)
    model = build_early_attention_gru(use_noise_layer=True).eval()

    predictions, attention = model(
        features, early_response, use_precomputed_features=True
    )
    assert predictions.shape == (2, 4, 5)
    assert attention.shape == (2, 4, 3, 16)
    assert len(model.get_layer_names()) == 3
    # noise_in_eval defaults to True, so evaluation stays stochastic.
    repeated_predictions, _ = model(
        features, early_response, use_precomputed_features=True
    )
    assert not torch.allclose(predictions, repeated_predictions)

    silent_model = build_early_attention_gru(
        use_noise_layer=True, noise_in_eval=False
    ).eval()
    first, _ = silent_model(features, early_response, use_precomputed_features=True)
    second, _ = silent_model(features, early_response, use_precomputed_features=True)
    torch.testing.assert_close(first, second)
# EOF


def test_early_attention_gru_temporal_embeddings_are_optional_and_used():
    features = torch.randn(2, 2, 16)
    early_response = torch.randn(2, 3, 5)
    plain_model = build_early_attention_gru()
    assert plain_model.temporal_query_embeddings is None

    model = build_early_attention_gru(use_temporal_embeddings=True).eval()
    assert model.temporal_query_embeddings.shape == (4, 6)
    predictions, attention = model(
        features, early_response, use_precomputed_features=True
    )
    assert attention.shape == (2, 4, 2, 16)

    # Changing one bin's embedding changes attention from that bin onwards only.
    with torch.no_grad():
        model.temporal_query_embeddings[2] += 5.0
    _, shifted_attention = model(
        features, early_response, use_precomputed_features=True
    )
    torch.testing.assert_close(attention[:, :2], shifted_attention[:, :2])
    assert not torch.allclose(attention[:, 2], shifted_attention[:, 2])

    # Embeddings receive gradients, i.e. they reach the loss.
    model.train()
    predictions, _ = model(features, early_response, use_precomputed_features=True)
    predictions.pow(2).mean().backward()
    assert model.temporal_query_embeddings.grad.abs().sum() > 0
# EOF
