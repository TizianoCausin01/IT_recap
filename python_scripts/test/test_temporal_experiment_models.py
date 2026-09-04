import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python_scripts" / "src"))

from model_classes.temporal_experiment_models import MODEL_CLASSES  # noqa: E402


def test_all_temporal_experiment_model_shapes():
    layer_features = torch.randn(3, 4, 16)
    expected_prediction_shape = (3, 6, 5)
    common_kwargs = {
        "n_layers": 4,
        "feature_dim": 16,
        "n_timepoints": 6,
        "n_neurons": 5,
        "hidden_dim": 16,
        "time_embedding_dim": 8,
        "attention_dim": 8,
        "n_attention_heads": 4,
        "dropout": 0.1,
    }

    for model_class in MODEL_CLASSES.values():
        model = model_class(**common_kwargs)
        predictions, diagnostics = model(
            layer_features,
            return_diagnostics=True,
        )
        assert predictions.shape == expected_prediction_shape
        assert torch.isfinite(predictions).all()
        assert diagnostics is not None
    # end for experiment model
# EOF


def test_attention_models_normalize_the_attention_axis():
    layer_features = torch.randn(2, 4, 16)
    common_kwargs = {
        "n_layers": 4,
        "feature_dim": 16,
        "n_timepoints": 6,
        "n_neurons": 5,
        "hidden_dim": 16,
        "time_embedding_dim": 8,
        "attention_dim": 8,
        "n_attention_heads": 4,
        "dropout": 0.0,
    }
    attention_models = (
        "recurrent_layer_attention",
        "recurrent_feature_attention",
        "time_local_feature_attention",
        "temporal_transformer",
    )
    for model_name in attention_models:
        model = MODEL_CLASSES[model_name](**common_kwargs)
        _, diagnostics = model(layer_features, return_diagnostics=True)
        attention = diagnostics["attention"]
        if attention.ndim == 4:
            attention_sum = attention.flatten(start_dim=-2).sum(dim=-1)
        else:
            attention_sum = attention.sum(dim=-1)
        # end if attention covers layers and embedding features
        assert torch.allclose(attention_sum, torch.ones_like(attention_sum))
    # end for attention model
# EOF
