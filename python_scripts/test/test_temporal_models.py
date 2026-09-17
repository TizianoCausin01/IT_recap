import sys
from pathlib import Path

import pytest
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python_scripts" / "src"))

from model_classes.temporal_models import (  # noqa: E402
    BaselineModel,
    ShowAttendTellGRUModel,
    TemporalNoiseBaselineModel,
)


ENCODER_DIM = 16
SHAPES = {
    "temporal_embedding_dim": 8,
    "value_dim": 8,
    "n_timepoints": 5,
    "temporal_compression_ratio": 1,
    "n_neurons": 4,
    "mlp_hidden_dim": 8,
}


class StubEncoder:
    """
    Minimal imgANN stand-in exposing only what BaselineModel reads at build
    time. Every test feeds cached activations, so the backbone never runs.
    """

    def __init__(self, encoder_dim=ENCODER_DIM):
        self.model = nn.Linear(encoder_dim, encoder_dim)
        self.encoder_dim = encoder_dim
        self.features = {}

    def set_relevant_layers(self, relevant_layers):
        self.layer_names = list(relevant_layers)

    def get_layer_output_shape(self, layer_name):
        return (1, self.encoder_dim)

    def create_forward_hook(self, layer_names=None):
        return None
# EOC


"""
build_model
Construct one decoder on the stub encoder at a given query-dropout setting.

INPUT:
    - model_class: type -> BaselineModel or one of its subclasses
    - temporal_embedding_dropout: float -> query dropout probability
    - kwargs: dict -> remaining class-specific arguments

OUTPUT:
    - model: BaselineModel -> decoder ready for cached-feature forward passes
"""
def build_model(model_class, temporal_embedding_dropout, **kwargs):
    torch.manual_seed(0)
    return model_class(
        StubEncoder(),
        layers=["layer_a", "layer_b"],
        temporal_embedding_dropout=temporal_embedding_dropout,
        **SHAPES,
        **kwargs,
    )
# EOF


def test_query_dropout_masks_coordinates_only_in_training():
    layer_features = torch.randn(6, 2, ENCODER_DIM)

    for model_class, kwargs in (
        (BaselineModel, {}),
        (TemporalNoiseBaselineModel, {"noise_in_eval": False}),
    ):
        model = build_model(model_class, 0.5, **kwargs)

        # key_query_dim is unset, so queries are the dropped embeddings
        # themselves and a masked coordinate is visible as an exact zero.
        model.train()
        training_queries = model._build_temporal_queries(
            layer_features.shape[0]
        )
        assert (training_queries == 0.0).any()

        model.eval()
        evaluation_queries = model._build_temporal_queries(
            layer_features.shape[0]
        )
        assert not (evaluation_queries == 0.0).any()
    # end for decoder class
# EOF


def test_query_dropout_is_independent_across_images():
    model = build_model(BaselineModel, 0.5)
    model.train()

    queries = model._build_temporal_queries(8)
    masks = (queries == 0.0).flatten(start_dim=1)

    # Sharing one mask across the batch would make every row identical.
    assert not bool((masks[0] == masks[1:]).all())
# EOF


def test_zero_query_dropout_leaves_predictions_unchanged():
    layer_features = torch.randn(3, 2, ENCODER_DIM)
    model = build_model(BaselineModel, 0.0)

    model.train()
    training_predictions, _ = model(
        layer_features,
        use_precomputed_features=True,
    )
    model.eval()
    evaluation_predictions, _ = model(
        layer_features,
        use_precomputed_features=True,
    )

    torch.testing.assert_close(training_predictions, evaluation_predictions)
# EOF


def test_query_dropout_reaches_feature_granularity_predictions():
    layer_features = torch.randn(3, 2, ENCODER_DIM)
    model = build_model(
        BaselineModel,
        0.5,
        attention_granularity="feature",
    )
    model.train()

    torch.manual_seed(1)
    first_predictions, _ = model(layer_features, use_precomputed_features=True)
    torch.manual_seed(2)
    second_predictions, _ = model(layer_features, use_precomputed_features=True)

    assert not torch.allclose(first_predictions, second_predictions)
    assert torch.isfinite(first_predictions).all()
# EOF


def test_show_attend_tell_gru_shapes_and_feature_attention():
    layer_features = torch.randn(3, 2, ENCODER_DIM)
    model = ShowAttendTellGRUModel(
        StubEncoder(),
        layers=["layer_a", "layer_b"],
        n_timepoints=5,
        n_neurons=4,
        hidden_dim=8,
        attention_dim=6,
        dropout=0.0,
    )

    predictions, attention = model(
        layer_features,
        use_precomputed_features=True,
    )

    assert predictions.shape == (3, 5, 4)
    assert attention.shape == (3, 5, 2, ENCODER_DIM)
    attention_mass = attention.flatten(start_dim=-2).sum(dim=-1)
    torch.testing.assert_close(attention_mass, torch.ones_like(attention_mass))
    assert torch.isfinite(predictions).all()
# EOF


def test_show_attend_tell_gru_supports_cached_features_without_encoder():
    layer_features = torch.randn(3, 2, ENCODER_DIM)
    model = ShowAttendTellGRUModel(
        None,
        layers=["layer_a", "layer_b"],
        n_timepoints=5,
        n_neurons=4,
        hidden_dim=8,
        attention_dim=6,
        encoder_dim=ENCODER_DIM,
    )

    predictions, attention = model(
        layer_features,
        use_precomputed_features=True,
    )
    assert predictions.shape == (3, 5, 4)
    assert attention.shape == (3, 5, 2, ENCODER_DIM)
    assert model.get_encoder() is None
# EOF


def test_cached_only_gru_rejects_online_image_inputs():
    model = ShowAttendTellGRUModel(
        None,
        layers=["layer_a", "layer_b"],
        n_timepoints=5,
        n_neurons=4,
        hidden_dim=8,
        attention_dim=6,
        encoder_dim=ENCODER_DIM,
    )

    with pytest.raises(ValueError, match="require an encoder"):
        model(torch.randn(2, 3, 224, 224), use_precomputed_features=False)
    # end with rejected online images
# EOF


def test_show_attend_tell_gru_has_one_recurrent_layer_and_no_time_parameters():
    model = ShowAttendTellGRUModel(
        StubEncoder(),
        layers=["layer_a", "layer_b"],
        n_timepoints=5,
        n_neurons=4,
        hidden_dim=8,
        attention_dim=6,
    )

    recurrent_modules = [
        module
        for module in model.modules()
        if isinstance(module, (nn.GRU, nn.GRUCell))
    ]
    assert len(recurrent_modules) == 1
    assert isinstance(recurrent_modules[0], nn.GRUCell)

    trainable_names = [
        name for name, _ in model.get_trainable_named_parameters()
    ]
    has_time_parameter = any(
        "temporal" in name or "time_embedding" in name
        for name in trainable_names
    )
    assert not has_time_parameter
# EOF
