import sys
from pathlib import Path

import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python_scripts" / "src"))

from IT_recap.tvsd_experiments import (  # noqa: E402
    average_targets_over_bin_groups,
    average_targets_over_window,
)
from model_classes.timebin_models import (  # noqa: E402
    TIMEBIN_MODEL_CLASSES,
    build_timebin_model,
)


SHAPES = {
    "n_layers": 3,
    "feature_dim": 16,
    "n_timepoints": 5,
    "n_neurons": 7,
}

# One minimal set of width arguments per searched architecture.
ARCHITECTURE_KWARGS = {
    "baseline": {"time_embedding_dim": 8, "value_dim": 12, "mlp_hidden_dim": 10},
    "tiny_transformer": {
        "hidden_dim": 16,
        "n_attention_heads": 4,
        "n_transformer_layers": 2,
    },
    "gru": {"hidden_dim": 12, "time_embedding_dim": 6},
    "gru_attention": {"hidden_dim": 12, "attention_dim": 8},
    "lds": {"state_dim": 9},
}


def make_model(architecture, **overrides):
    """Build one searched decoder with its minimal width arguments."""
    return build_timebin_model(
        architecture,
        **SHAPES,
        **ARCHITECTURE_KWARGS[architecture],
        **overrides,
    )
# EOF


@pytest.mark.parametrize("architecture", sorted(TIMEBIN_MODEL_CLASSES))
def test_every_architecture_predicts_all_time_bins(architecture):
    model = make_model(architecture)
    predictions, diagnostics = model(torch.randn(4, 3, 16))
    assert predictions.shape == (4, 5, 7)
    assert torch.isfinite(predictions).all()
    assert diagnostics is None
# EOF


@pytest.mark.parametrize("architecture", sorted(TIMEBIN_MODEL_CLASSES))
def test_noise_is_sampled_in_training_and_silent_in_eval(architecture):
    model = make_model(
        architecture,
        dropout=0.0,
        input_noise_std=0.5,
        temporal_noise_std=0.5,
    )
    model.train()
    # Independent draws per forward pass make two training passes differ.
    assert not torch.allclose(
        model(torch.zeros(2, 3, 16))[0], model(torch.zeros(2, 3, 16))[0]
    )
    model.eval()
    assert torch.allclose(
        model(torch.zeros(2, 3, 16))[0], model(torch.zeros(2, 3, 16))[0]
    )
# EOF


@pytest.mark.parametrize("architecture", sorted(TIMEBIN_MODEL_CLASSES))
def test_zero_noise_leaves_the_architecture_deterministic(architecture):
    model = make_model(architecture, dropout=0.0).train()
    features = torch.randn(2, 3, 16)
    assert torch.allclose(model(features)[0], model(features)[0])
# EOF


def test_noise_scales_must_be_non_negative():
    with pytest.raises(ValueError):
        make_model("lds", input_noise_std=-0.1)
    # end with rejected input-noise scale
    with pytest.raises(ValueError):
        make_model("lds", temporal_noise_std=-0.1)
    # end with rejected temporal-noise scale
# EOF


def test_transformer_rejects_indivisible_head_counts():
    with pytest.raises(ValueError):
        build_timebin_model(
            "tiny_transformer",
            **SHAPES,
            hidden_dim=18,
            n_attention_heads=4,
        )
    # end with rejected attention width
# EOF


def test_lds_starts_as_contracting_dynamics():
    model = make_model("lds", spectral_init=0.9)
    # A radius below one keeps the five-step rollout from exploding at init.
    assert model.spectral_radius() < 1.0
# EOF


def test_unknown_architecture_is_rejected():
    with pytest.raises(KeyError):
        build_timebin_model("gated_gru", **SHAPES)
    # end with rejected architecture name
# EOF


def test_bin_groups_average_consecutive_cached_bins():
    targets = np.arange(2 * 6 * 3, dtype=np.float32).reshape(2, 6, 3)
    grouped = average_targets_over_bin_groups(targets, np.arange(2, 6), 2)
    assert grouped.shape == (2, 2, 3)
    expected = targets[:, 2:4].mean(axis=1)
    assert np.allclose(grouped[:, 0], expected)
# EOF


def test_window_average_is_the_single_group_case():
    targets = np.random.default_rng(0).normal(size=(3, 8, 4)).astype(np.float32)
    bin_indices = np.arange(1, 7)
    assert np.allclose(
        average_targets_over_window(targets, bin_indices),
        average_targets_over_bin_groups(targets, bin_indices, len(bin_indices)),
    )
# EOF


def test_partial_groups_are_rejected():
    targets = np.zeros((2, 6, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        average_targets_over_bin_groups(targets, np.arange(0, 5), 2)
    # end with rejected incomplete group
# EOF
