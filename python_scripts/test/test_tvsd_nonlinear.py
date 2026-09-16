import sys
from pathlib import Path

import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python_scripts" / "src"))

from IT_recap.tvsd_objectives import (  # noqa: E402
    WeightAverage,
    add_relative_noise,
    build_objective,
    correlation_loss,
    drop_feature_groups,
    masked_prediction_loss,
    mixup_batch,
    random_translate_maps,
)
from model_classes.spatial_readout_models import (  # noqa: E402
    FactorizedSpatialReadout,
)
from model_classes.tvsd_nonlinear_models import (  # noqa: E402
    MultiBranchTimebinDecoder,
    MultiMaskSpatialReadout,
)


N_TIME, N_SITES, N_CHANNELS, MAP_SIZE = 5, 8, 6, 7


def make_maps(batch=16):
    return torch.randn(batch, N_CHANNELS, MAP_SIZE, MAP_SIZE)
# EOF


def test_correlation_loss_is_zero_for_a_rescaled_prediction():
    torch.manual_seed(0)
    targets = torch.randn(64, N_TIME, N_SITES)
    assert float(correlation_loss(targets, targets)) == pytest.approx(0.0, abs=1e-5)
    # Correlation is scale-free, so shrinking cannot lower this loss.
    shrunk = 0.1 * targets + 3.0
    assert float(correlation_loss(shrunk, targets)) == pytest.approx(0.0, abs=1e-5)
    # An MSE, by contrast, is changed a lot by exactly that rescaling.
    mse = build_objective("mse")
    assert float(mse(shrunk, targets)) > float(mse(targets, targets))
# EOF


def test_correlation_loss_needs_more_than_one_image():
    with pytest.raises(ValueError):
        correlation_loss(torch.randn(1, N_TIME, N_SITES), torch.randn(1, N_TIME, N_SITES))
# EOF


def test_unknown_objective_is_rejected():
    with pytest.raises(KeyError):
        build_objective("poisson")
# EOF


def test_mixup_is_a_convex_combination_and_can_be_disabled():
    torch.manual_seed(0)
    features = torch.randn(32, 3, 4)
    targets = torch.randn(32, N_TIME, N_SITES)
    same, same_targets = mixup_batch([features], targets, alpha=0.0)
    assert torch.equal(same[0], features) and torch.equal(same_targets, targets)
    mixed, mixed_targets = mixup_batch([features], targets, alpha=0.4)
    # A convex mix cannot leave the range of the batch it was built from.
    assert mixed[0].max() <= features.max() + 1e-5
    assert mixed[0].min() >= features.min() - 1e-5
    assert mixed_targets.shape == targets.shape
# EOF


def test_translation_moves_the_map_but_keeps_its_shape():
    torch.manual_seed(0)
    maps = make_maps()
    assert torch.equal(random_translate_maps(maps, 0.0), maps)
    shifted = random_translate_maps(maps, 1.5)
    assert shifted.shape == maps.shape
    assert not torch.allclose(shifted, maps)
# EOF


def test_group_dropout_keeps_the_expected_scale():
    torch.manual_seed(0)
    features = torch.ones(4096, 3, 2)
    dropped = drop_feature_groups(features, 0.25)
    assert float(dropped.mean()) == pytest.approx(1.0, abs=0.05)
    # Whole depths are dropped, never single coordinates.
    per_group = dropped[0].abs().sum(dim=-1).numpy()
    kept_value = 2.0 / 0.75
    assert np.all(np.isclose(per_group, 0.0) | np.isclose(per_group, kept_value))
# EOF


def test_relative_noise_scales_with_the_input_spread():
    torch.manual_seed(0)
    small = torch.randn(4000, 10) * 0.1
    large = small * 100.0
    small_noise = (add_relative_noise(small, 0.5) - small).std()
    large_noise = (add_relative_noise(large, 0.5) - large).std()
    assert float(large_noise / small_noise) == pytest.approx(100.0, rel=0.1)
    assert torch.equal(add_relative_noise(small, 0.0), small)
# EOF


def test_masked_loss_ignores_auxiliary_sites_when_unweighted():
    torch.manual_seed(0)
    predictions = torch.randn(32, N_TIME, N_SITES + 4)
    targets = torch.randn(32, N_TIME, N_SITES + 4)
    objective = build_objective("mse")
    primary_only = masked_prediction_loss(
        predictions, targets, N_SITES, objective, auxiliary_weight=0.0
    )
    assert float(primary_only) == pytest.approx(
        float(objective(predictions[..., :N_SITES], targets[..., :N_SITES]))
    )
    # A positive weight must move the loss.
    assert float(
        masked_prediction_loss(predictions, targets, N_SITES, objective, 1.0)
    ) > float(primary_only)
# EOF


def test_weight_average_tracks_the_model_it_follows():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 3)
    averager = WeightAverage(model, decay=0.5)
    with torch.no_grad():
        model.weight.add_(1.0)
    averager.update(model)
    # One update at decay 0.5 sits halfway between the two snapshots.
    expected = model.weight - 0.5
    assert torch.allclose(averager.module.weight, expected, atol=1e-6)
    with pytest.raises(ValueError):
        WeightAverage(model, decay=1.0)
# EOF


def test_multimask_readout_reduces_to_the_published_factorized_head():
    common = dict(
        n_channels=N_CHANNELS,
        spatial_size=(MAP_SIZE, MAP_SIZE),
        n_timepoints=N_TIME,
        n_neurons=N_SITES,
        hidden_dim=16,
        dropout=0.0,
    )
    torch.manual_seed(0)
    published = FactorizedSpatialReadout(**common)
    torch.manual_seed(0)
    generalized = MultiMaskSpatialReadout(**common, n_masks=1, core_layers=0)
    published_parameters = sum(p.numel() for p in published.parameters())
    assert sum(p.numel() for p in generalized.parameters()) == published_parameters
    generalized.load_state_dict(
        {
            name: published.state_dict()[name].reshape(
                generalized.state_dict()[name].shape
            )
            for name in generalized.state_dict()
        }
    )
    published.eval()
    generalized.eval()
    maps = make_maps()
    assert torch.allclose(published(maps)[0], generalized(maps)[0], atol=1e-5)
# EOF


def test_masks_are_normalized_and_extra_masks_add_parameters():
    readout = MultiMaskSpatialReadout(
        n_channels=N_CHANNELS,
        spatial_size=(MAP_SIZE, MAP_SIZE),
        n_timepoints=N_TIME,
        n_neurons=N_SITES,
        hidden_dim=16,
        n_masks=3,
        core_layers=2,
        core_dim=12,
    )
    masks = readout.spatial_masks()
    assert masks.shape == (N_SITES, 3, MAP_SIZE, MAP_SIZE)
    assert torch.allclose(masks.sum(dim=(-2, -1)), torch.ones(N_SITES, 3), atol=1e-5)
    predictions, diagnostics = readout(make_maps(), return_diagnostics=True)
    assert predictions.shape == (16, N_TIME, N_SITES)
    assert diagnostics["core_code"].shape == (16, 16)
# EOF


def test_readout_rejects_a_map_of_the_wrong_geometry():
    readout = MultiMaskSpatialReadout(
        n_channels=N_CHANNELS,
        spatial_size=(MAP_SIZE, MAP_SIZE),
        n_timepoints=N_TIME,
        n_neurons=N_SITES,
        hidden_dim=8,
    )
    with pytest.raises(ValueError):
        readout(torch.randn(4, N_CHANNELS + 1, MAP_SIZE, MAP_SIZE))
# EOF


def test_multi_branch_decoder_sums_its_branches():
    spec = dict(
        n_channels=N_CHANNELS,
        spatial_size=(MAP_SIZE, MAP_SIZE),
        n_timepoints=N_TIME,
        n_neurons=N_SITES,
        hidden_dim=8,
    )
    torch.manual_seed(0)
    model = MultiBranchTimebinDecoder(
        "gru",
        dict(
            n_layers=2, feature_dim=6, n_timepoints=N_TIME, n_neurons=N_SITES,
            hidden_dim=16, time_embedding_dim=4, dropout=0.0,
        ),
        [spec, spec],
        reconstruction_dim=6,
    ).eval()
    features = torch.randn(4, 2, 6)
    maps = [make_maps(4), make_maps(4)]
    predictions, diagnostics = model(features, maps, return_diagnostics=True)
    branch_sum = (
        diagnostics["temporal"] + diagnostics["spatial_0"] + diagnostics["spatial_1"]
    )
    assert torch.allclose(predictions, branch_sum, atol=1e-6)
    assert diagnostics["reconstruction"].shape == (4, 6)
    assert len(model.mask_parameter_names()) == 2
# EOF


def test_multi_branch_decoder_needs_at_least_one_branch():
    with pytest.raises(ValueError):
        MultiBranchTimebinDecoder(None, {}, [])
# EOF
