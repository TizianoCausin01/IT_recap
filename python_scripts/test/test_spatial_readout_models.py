import sys
from pathlib import Path

import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python_scripts" / "src"))

from model_classes.spatial_readout_models import (  # noqa: E402
    SPATIAL_READOUT_CLASSES,
    build_spatial_readout,
)


READOUT_KWARGS = {
    "n_channels": 8,
    "spatial_size": (5, 5),
    "n_timepoints": 3,
    "n_neurons": 4,
    "hidden_dim": 6,
    "dropout": 0.0,
}


@pytest.mark.parametrize("readout_name", sorted(SPATIAL_READOUT_CLASSES))
def test_readouts_predict_every_bin_and_site(readout_name):
    readout = build_spatial_readout(readout_name, **READOUT_KWARGS).eval()
    predictions, diagnostics = readout(torch.randn(7, 8, 5, 5))
    assert predictions.shape == (7, 3, 4)
    assert torch.isfinite(predictions).all()
    assert diagnostics is None
# EOF


@pytest.mark.parametrize("readout_name", sorted(SPATIAL_READOUT_CLASSES))
def test_untrained_readouts_barely_move_the_base_prediction(readout_name):
    readout = build_spatial_readout(readout_name, **READOUT_KWARGS).eval()
    # The head predicts a residual, so at initialization it must be a near
    # no-op: the combined model then starts exactly at the base decoder.
    predictions, _ = readout(torch.randn(7, 8, 5, 5))
    assert predictions.abs().max() < 0.05
# EOF


@pytest.mark.parametrize("readout_name", sorted(SPATIAL_READOUT_CLASSES))
def test_readouts_reject_the_wrong_map_geometry(readout_name):
    readout = build_spatial_readout(readout_name, **READOUT_KWARGS).eval()
    with pytest.raises(ValueError):
        readout(torch.randn(2, 8, 7, 7))
    # end with rejected feature map
# EOF


def test_spatial_masks_are_distributions_over_positions():
    readout = build_spatial_readout("factorized_spatial", **READOUT_KWARGS)
    masks = readout.spatial_masks()
    assert masks.shape == (4, 5, 5)
    assert torch.allclose(masks.sum(dim=(1, 2)), torch.ones(4), atol=1e-5)
# EOF


def test_a_sharp_mask_reads_one_position():
    readout = build_spatial_readout("factorized_spatial", **READOUT_KWARGS).eval()
    with torch.no_grad():
        # Site 0 is forced to read the top-left corner alone.
        readout.spatial_logits[0] = -50.0
        readout.spatial_logits[0, 0] = 50.0
    # end with forced mask
    mask = readout.spatial_masks()[0].detach()
    assert float(mask[0, 0]) > 0.99
    assert float(mask.sum()) == pytest.approx(1.0, abs=1e-5)
# EOF


def test_the_pooled_control_ignores_where_a_feature_sits():
    readout = build_spatial_readout("pooled_control", **READOUT_KWARGS).eval()
    feature_map = torch.randn(2, 8, 5, 5)
    # A spatial permutation leaves the global average, and so the control's
    # prediction, unchanged; that is exactly what makes it the control.
    shuffled = feature_map.flatten(start_dim=2)[..., torch.randperm(25)]
    shuffled = shuffled.reshape_as(feature_map)
    assert torch.allclose(
        readout(feature_map)[0], readout(shuffled)[0], atol=1e-5
    )
# EOF


def test_unknown_readout_is_rejected():
    with pytest.raises(KeyError):
        build_spatial_readout("gaussian_rf", **READOUT_KWARGS)
    # end with rejected readout name
# EOF


def test_invalid_dimensions_are_rejected():
    with pytest.raises(ValueError):
        build_spatial_readout(
            "factorized_spatial", **{**READOUT_KWARGS, "hidden_dim": 0}
        )
    # end with rejected width
# EOF


from model_classes.spatial_readout_models import (  # noqa: E402
    JointSpatialTemporalDecoder,
)


JOINT_KWARGS = {
    "temporal_architecture": "gru",
    "temporal_kwargs": {
        "n_layers": 2,
        "feature_dim": 12,
        "n_timepoints": 3,
        "n_neurons": 4,
        "hidden_dim": 8,
        "time_embedding_dim": 5,
    },
    "spatial_readout": "factorized_spatial",
    "spatial_kwargs": READOUT_KWARGS,
}


def joint_inputs():
    """Return one batch of pooled features and matching feature maps."""
    return torch.randn(6, 2, 12), torch.randn(6, 8, 5, 5)
# EOF


@pytest.mark.parametrize(
    "use_temporal,use_spatial", [(True, True), (True, False), (False, True)]
)
def test_joint_decoder_predicts_from_either_or_both_branches(
    use_temporal, use_spatial
):
    model = JointSpatialTemporalDecoder(
        **JOINT_KWARGS, use_temporal=use_temporal, use_spatial=use_spatial
    ).eval()
    predictions, diagnostics = model(*joint_inputs(), return_diagnostics=True)
    assert predictions.shape == (6, 3, 4)
    assert torch.isfinite(predictions).all()
    assert ("temporal" in diagnostics) is use_temporal
    assert ("spatial" in diagnostics) is use_spatial
# EOF


def test_joint_prediction_is_the_sum_of_its_branches():
    model = JointSpatialTemporalDecoder(**JOINT_KWARGS).eval()
    predictions, diagnostics = model(*joint_inputs(), return_diagnostics=True)
    assert torch.allclose(
        predictions, diagnostics["temporal"] + diagnostics["spatial"], atol=1e-6
    )
# EOF


def test_joint_decoder_needs_at_least_one_branch():
    with pytest.raises(ValueError):
        JointSpatialTemporalDecoder(
            **JOINT_KWARGS, use_temporal=False, use_spatial=False
        )
    # end with rejected empty model
# EOF


def test_mask_parameters_are_reported_only_when_spatial():
    with_spatial = JointSpatialTemporalDecoder(**JOINT_KWARGS)
    without_spatial = JointSpatialTemporalDecoder(
        **JOINT_KWARGS, use_spatial=False
    )
    # The masks need their own optimizer group, so they must be findable.
    assert with_spatial.mask_parameter_names() == ["spatial_branch.spatial_logits"]
    assert without_spatial.mask_parameter_names() == []
# EOF
