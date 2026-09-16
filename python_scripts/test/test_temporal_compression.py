import sys
from pathlib import Path

import pytest
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python_scripts" / "src"))

from model_classes.temporal_models import (  # noqa: E402
    BaselineModel,
    NoiseLayerBaselineModel,
    TemporalNoiseBaselineModel,
)


class StubEncoder:
    """Minimal frozen-encoder interface for cached-feature tests."""

    def __init__(self, encoder_dim=12):
        self.model = nn.Linear(encoder_dim, encoder_dim)
        self.encoder_dim = encoder_dim
        self.features = {}

    def set_relevant_layers(self, relevant_layers):
        self.layer_names = list(relevant_layers)

    def get_layer_output_shape(self, layer_name):
        return (1, self.encoder_dim)

    def create_forward_hook(self, layer_names=None):
        return None
    # EOF
# EOC


"""
build_compressed_model
Construct a baseline subclass with two coarse queries and six fine readouts.

INPUT:
    - model_class: type -> BaselineModel or a temporal/noise subclass

OUTPUT:
    - model: BaselineModel -> compressed decoder for cached features
"""
def build_compressed_model(model_class, attention_granularity):
    class_kwargs = {}
    if model_class is NoiseLayerBaselineModel:
        class_kwargs["noise_in_eval"] = False
    elif model_class is TemporalNoiseBaselineModel:
        class_kwargs.update({
            "temporal_noise_std": 0.0,
            "noise_in_eval": False,
        })
    # end if a noise subclass needs deterministic test settings

    return model_class(
        StubEncoder(),
        layers=["shallow", "deep"],
        temporal_embedding_dim=8,
        value_dim=7,
        n_timepoints=6,
        temporal_compression_ratio=3,
        n_neurons=5,
        mlp_hidden_dim=9,
        dropout=0.0,
        attention_granularity=attention_granularity,
        **class_kwargs,
    )
# EOF


@pytest.mark.parametrize(
    "model_class",
    [BaselineModel, NoiseLayerBaselineModel, TemporalNoiseBaselineModel],
)
@pytest.mark.parametrize("attention_granularity", ["layer", "feature"])
def test_compression_expands_coarse_attention_into_all_fine_readouts(
    model_class,
    attention_granularity,
):
    model = build_compressed_model(
        model_class,
        attention_granularity,
    ).eval()
    cached_features = torch.randn(4, 2, 12)

    predictions, attention = model(
        cached_features,
        use_precomputed_features=True,
    )

    assert model.get_n_temporal_embeddings() == 2
    assert model.get_temporal_compression_ratio() == 3
    assert predictions.shape == (4, 6, 5)
    expected_attention_layers = 3 if model_class is NoiseLayerBaselineModel else 2
    expected_attention_shape = (4, 2, expected_attention_layers)
    if attention_granularity == "feature":
        expected_attention_shape = (*expected_attention_shape, 12)
    # end if attention operates over individual embedding coordinates
    assert attention.shape == expected_attention_shape
    assert len(model.timebin_readouts) == 6
    torch.testing.assert_close(
        attention.flatten(start_dim=2).sum(dim=-1),
        torch.ones(4, 2),
    )
# EOF


@pytest.mark.parametrize("compression_ratio", [0, 4, 2.5])
def test_compression_ratio_must_be_positive_integer_dividing_time(compression_ratio):
    with pytest.raises(ValueError):
        BaselineModel(
            StubEncoder(),
            layers=["shallow", "deep"],
            temporal_embedding_dim=8,
            value_dim=7,
            n_timepoints=6,
            temporal_compression_ratio=compression_ratio,
            n_neurons=5,
            mlp_hidden_dim=9,
        )
# EOF
