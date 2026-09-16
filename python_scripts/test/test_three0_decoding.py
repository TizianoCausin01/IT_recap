import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python_scripts" / "src"))

from IT_recap.three0_decoding import (  # noqa: E402
    fit_reduced_rank_decoder,
    kernel_ridge_path,
    make_image_splits,
    normalized_linear_kernel,
    score_predictions,
)


N_IMAGES, N_TIME, N_UNITS = 200, 5, 4


"""
make_low_rank_problem
Build responses that are a rank-two linear function of the image features, so
a reduced-rank decoder can recover them and a noise floor is known exactly.

INPUT:
    - noise: float -> standard deviation of the additive response noise
    - rank: int -> number of latent components driving the responses

OUTPUT:
    - features: np.ndarray -> [images, dimensions] image features
    - targets: np.ndarray -> [images, time * units] flattened responses
"""
def make_low_rank_problem(noise=0.1, rank=2):
    rng = np.random.default_rng(0)
    features = rng.standard_normal((N_IMAGES, 12))
    encoder = rng.standard_normal((12, rank))
    decoder = rng.standard_normal((rank, N_TIME * N_UNITS))
    targets = features @ encoder @ decoder
    targets = targets + noise * rng.standard_normal(targets.shape)
    return features, targets
# EOF


def test_splits_are_disjoint_and_cover_every_image():
    ann_index = np.arange(N_IMAGES)
    splits = make_image_splits(ann_index, seed=0)
    joined = np.concatenate([splits[name] for name in ("train", "validation", "test")])
    assert len(np.unique(joined)) == N_IMAGES
    assert len(splits["test"]) == round(N_IMAGES * 0.2)
    # A different seed must move images between splits.
    other = make_image_splits(ann_index, seed=1)
    assert not np.array_equal(np.sort(other["test"]), np.sort(splits["test"]))
# EOF


def test_normalized_kernel_is_scale_invariant():
    rng = np.random.default_rng(1)
    features = rng.standard_normal((30, 7))
    kernel = normalized_linear_kernel(features)
    # Per-column rescaling is undone by the z-scoring inside the kernel.
    rescaled = normalized_linear_kernel(features * rng.uniform(2, 5, 7))
    assert np.allclose(kernel, rescaled)
    assert np.isclose(np.trace(kernel), len(kernel))
# EOF


def test_kernel_ridge_path_matches_sklearn_ridge():
    features, targets = make_low_rank_problem()
    fit_rows = np.arange(150)
    eval_rows = np.arange(150, N_IMAGES)
    # Dual kernel ridge and primal ridge on centered features must agree.
    centered = features - features[fit_rows].mean(0)
    kernel = centered @ centered.T
    alpha = 3.0
    predictions = kernel_ridge_path(
        kernel, fit_rows, targets[fit_rows], eval_rows, [alpha]
    )[0]
    reference = Ridge(alpha=alpha).fit(features[fit_rows], targets[fit_rows])
    assert np.allclose(predictions, reference.predict(features[eval_rows]), atol=1e-6)
# EOF


def test_reduced_rank_decoder_recovers_a_low_rank_mapping():
    features, targets = make_low_rank_problem(noise=0.05, rank=2)
    kernel = normalized_linear_kernel(features)
    fit_rows = np.arange(150)
    eval_rows = np.arange(150, N_IMAGES)
    predictions, alphas = fit_reduced_rank_decoder(
        kernel, targets, fit_rows, eval_rows, rank=2, n_folds=3
    )
    assert predictions.shape == (len(eval_rows), targets.shape[1])
    assert len(alphas) == 2
    residual = ((predictions - targets[eval_rows]) ** 2).mean()
    total = ((targets[eval_rows] - targets[fit_rows].mean(0)) ** 2).mean()
    assert 1.0 - residual / total > 0.9
# EOF


def test_reduced_rank_decoder_beats_full_rank_when_targets_are_low_rank():
    features, targets = make_low_rank_problem(noise=1.5, rank=2)
    kernel = normalized_linear_kernel(features)
    fit_rows = np.arange(80)
    eval_rows = np.arange(80, N_IMAGES)
    low_rank, _ = fit_reduced_rank_decoder(
        kernel, targets, fit_rows, eval_rows, rank=2, n_folds=4
    )
    full_rank, _ = fit_reduced_rank_decoder(
        kernel, targets, fit_rows, eval_rows, rank=targets.shape[1], n_folds=4
    )
    low_error = ((low_rank - targets[eval_rows]) ** 2).mean()
    full_error = ((full_rank - targets[eval_rows]) ** 2).mean()
    assert low_error < full_error
# EOF


def test_output_weights_do_not_change_an_unweighted_solution():
    features, targets = make_low_rank_problem()
    kernel = normalized_linear_kernel(features)
    fit_rows, eval_rows = np.arange(150), np.arange(150, N_IMAGES)
    unweighted, _ = fit_reduced_rank_decoder(
        kernel, targets, fit_rows, eval_rows, rank=3, n_folds=3
    )
    # A constant weight only rescales the space the SVD is taken in.
    weighted, _ = fit_reduced_rank_decoder(
        kernel, targets, fit_rows, eval_rows, rank=3, n_folds=3,
        output_weights=np.full(targets.shape[1], 4.0),
    )
    assert np.allclose(unweighted, weighted, atol=1e-8)
# EOF


def test_score_predictions_reports_the_ceiling_fractions():
    rng = np.random.default_rng(2)
    targets = rng.standard_normal((40, N_TIME, N_UNITS))
    fit_mean = targets.mean(0)
    perfect = score_predictions(targets, targets, fit_mean, noise_floor=0.0)
    assert perfect["mse"] == pytest.approx(0.0)
    assert perfect["r2"] == pytest.approx(1.0)
    assert perfect["fraction_of_ceiling"] == pytest.approx(1.0)
    # A prediction stuck at the fitting mean explains nothing.
    flat = score_predictions(
        np.broadcast_to(fit_mean, targets.shape), targets, fit_mean
    )
    assert flat["r2"] == pytest.approx(0.0)
    assert "fraction_of_ceiling" not in flat
# EOF
