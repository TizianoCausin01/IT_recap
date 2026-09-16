import sys
from pathlib import Path

import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python_scripts" / "src"))

from IT_recap.tvsd_experiments import (  # noqa: E402
    average_targets_over_window,
    select_window_bin_indices,
)
from model_classes.window_models import (  # noqa: E402
    LayerAttentionPopulationDecoder,
)


DECODER_KWARGS = {
    "n_layers": 3,
    "feature_dim": 16,
    "n_neurons": 5,
    "hidden_dim": 8,
    "n_attention_heads": 2,
    "mlp_hidden_dim": 12,
    "dropout": 0.0,
}


def test_decoder_returns_a_singleton_time_axis():
    model = LayerAttentionPopulationDecoder(**DECODER_KWARGS)
    predictions, diagnostics = model(torch.randn(4, 3, 16))
    # The window average is one population vector, kept as [batch, 1, neurons].
    assert predictions.shape == (4, 1, 5)
    assert torch.isfinite(predictions).all()
    assert diagnostics is None
# EOF


def test_decoder_attention_normalizes_over_the_key_axis():
    model = LayerAttentionPopulationDecoder(**DECODER_KWARGS).eval()
    _, diagnostics = model(torch.randn(4, 3, 16), return_diagnostics=True)
    attention = diagnostics["attention"]
    assert attention.shape == (4, 3, 3)
    assert torch.allclose(attention.sum(dim=-1), torch.ones(4, 3), atol=1e-5)
# EOF


def test_mean_pooling_shrinks_the_readout_input():
    concat_model = LayerAttentionPopulationDecoder(**DECODER_KWARGS)
    mean_model = LayerAttentionPopulationDecoder(
        **DECODER_KWARGS, readout_pooling="mean"
    )
    assert concat_model.readout[0].in_features == 8 * 3
    assert mean_model.readout[0].in_features == 8
    assert mean_model(torch.randn(2, 3, 16))[0].shape == (2, 1, 5)
# EOF


def test_decoder_depends_on_depth_order():
    # Learned depth embeddings must break the permutation invariance that a
    # bare self-attention block over layer tokens would otherwise have.
    torch.manual_seed(0)
    model = LayerAttentionPopulationDecoder(**DECODER_KWARGS).eval()
    features = torch.randn(2, 3, 16)
    original, _ = model(features)
    permuted, _ = model(features[:, [2, 0, 1]])
    assert not torch.allclose(original, permuted, atol=1e-4)
# EOF


def test_decoder_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        LayerAttentionPopulationDecoder(**{**DECODER_KWARGS, "hidden_dim": 9})
    # end with indivisible attention heads
    with pytest.raises(ValueError):
        LayerAttentionPopulationDecoder(
            **DECODER_KWARGS, readout_pooling="median"
        )
    # end with unsupported pooling
    model = LayerAttentionPopulationDecoder(**DECODER_KWARGS)
    with pytest.raises(ValueError):
        model(torch.randn(4, 2, 16))
    # end with the wrong number of layer tokens
# EOF


def test_window_bins_match_the_requested_width():
    # 0-200 ms cached at 100 Hz: 75-175 ms snaps to the ten bins 8-17, which
    # cover 80-180 ms and keep the requested 100 ms width.
    bin_indices, covered_ms = select_window_bin_indices(20, 100, 0.0, 75.0, 175.0)
    assert bin_indices.tolist() == list(range(8, 18))
    assert covered_ms == (80.0, 180.0)

    aligned_indices, aligned_ms = select_window_bin_indices(
        20, 100, 0.0, 80.0, 180.0
    )
    assert aligned_indices.tolist() == bin_indices.tolist()
    assert aligned_ms == covered_ms
# EOF


def test_window_bins_reject_windows_outside_the_cache():
    with pytest.raises(ValueError):
        select_window_bin_indices(20, 100, 0.0, 175.0, 75.0)
    # end with a reversed window
    with pytest.raises(ValueError):
        select_window_bin_indices(20, 100, 0.0, 150.0, 260.0)
    # end with a window past the cached response
# EOF


def test_window_average_matches_a_direct_mean():
    rng = np.random.default_rng(0)
    targets = rng.normal(size=(7, 20, 4)).astype(np.float32)
    bin_indices = np.arange(8, 18)

    # A chunk size below the presentation count exercises the streaming path.
    window_targets = average_targets_over_window(targets, bin_indices, chunk_size=3)
    assert window_targets.shape == (7, 1, 4)
    assert np.allclose(
        window_targets[:, 0], targets[:, bin_indices].mean(axis=1), atol=1e-6
    )
# EOF


def test_window_average_rejects_out_of_range_bins():
    targets = np.zeros((3, 20, 4), dtype=np.float32)
    with pytest.raises(IndexError):
        average_targets_over_window(targets, np.arange(15, 25))
    # end with bins past the cached time axis
# EOF
