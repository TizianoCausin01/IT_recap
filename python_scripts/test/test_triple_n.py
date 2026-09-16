from pathlib import Path

import h5py
import numpy as np
import pytest

from IT_recap.neural_prediction_training import split_half_reliability
from IT_recap.triple_n import (
    TRIPLE_N_N_NSD_IMAGES,
    build_triple_n_datasets,
    build_triple_n_image_metadata,
    load_triple_n_good_unit_metadata,
    load_triple_n_processed,
    preprocess_triple_n_firing_rate_targets,
    select_triple_n_units,
    triple_n_nsd_trial_indices,
)


"""
_write_synthetic_good_unit
Write a miniature GoodUnit file with the MATLAB v7.3 layout the loader expects:
GoodUnitStrc fields are object-reference arrays, and every MATLAB array appears
transposed under h5py, so a [trial, time] raster is stored as [time, trial].

INPUT:
    - path: Path -> destination .mat path
    - n_units: int -> units to synthesize
    - pre_onset_ms: int -> samples stored before stimulus onset
    - post_onset_ms: int -> samples stored after stimulus onset
    - trial_image_ids: np.ndarray -> one-based stimulus id per valid trial

OUTPUT:
    - rasters: np.ndarray -> [units, trials, time] ground-truth spike counts
"""
def _write_synthetic_good_unit(
    path, n_units, pre_onset_ms, post_onset_ms, trial_image_ids
):
    rng = np.random.default_rng(0)
    n_samples = pre_onset_ms + post_onset_ms + 1
    n_trials = len(trial_image_ids)
    rasters = rng.integers(0, 2, size=(n_units, n_trials, n_samples)).astype(
        np.float64
    )

    # Two failed trials are interleaved so the validity mask is actually
    # exercised rather than trivially all-true.
    trial_valid_idx = np.concatenate([[0], trial_image_ids[:2], [0],
                                      trial_image_ids[2:]])
    dataset_valid_idx = trial_valid_idx != 0

    with h5py.File(path, "w") as h5file:
        params = h5file.create_group("global_params")
        params.create_dataset("pre_onset", data=np.asarray([[pre_onset_ms]]))
        params.create_dataset("post_onset", data=np.asarray([[post_onset_ms]]))

        meta = h5file.create_group("meta_data")
        meta.create_dataset(
            "trial_valid_idx", data=trial_valid_idx.reshape(1, -1).astype(float)
        )
        meta.create_dataset(
            "dataset_valid_idx",
            data=dataset_valid_idx.reshape(1, -1).astype(np.uint8),
        )

        unit_group = h5file.create_group("GoodUnitStrc")
        storage = h5file.create_group("unit_storage")
        reference_dtype = h5py.special_dtype(ref=h5py.Reference)
        raster_references = np.empty((n_units, 1), dtype=object)
        for unit_idx in range(n_units):
            # Stored transposed, exactly as MATLAB v7.3 writes [trial, time].
            dataset = storage.create_dataset(
                f"raster_{unit_idx}", data=rasters[unit_idx].T
            )
            raster_references[unit_idx, 0] = dataset.ref
        # end for unit
        unit_group.create_dataset(
            "Raster", data=raster_references, dtype=reference_dtype
        )
    # end with synthetic GoodUnit file
    return rasters
# EOF


def test_good_unit_metadata_drops_failed_trials(tmp_path):
    trial_image_ids = np.asarray([3, 7, 3, 7, 5, 5], dtype=np.int64)
    path = tmp_path / "GoodUnit_240629_Synthetic_NSD1000_LOC_g0.mat"
    _write_synthetic_good_unit(path, 2, 50, 100, trial_image_ids)

    loaded_ids, pre_onset_ms, n_units, n_samples = (
        load_triple_n_good_unit_metadata(path)
    )
    np.testing.assert_array_equal(loaded_ids, trial_image_ids)
    assert pre_onset_ms == 50
    assert n_units == 2
    assert n_samples == 151
# EOF


def test_firing_rate_targets_match_manual_binning(tmp_path):
    trial_image_ids = np.asarray([1, 2, 1, 2, 3, 3], dtype=np.int64)
    path = tmp_path / "GoodUnit_240629_Synthetic_NSD1000_LOC_g0.mat"
    rasters = _write_synthetic_good_unit(path, 3, 50, 200, trial_image_ids)

    targets = preprocess_triple_n_firing_rate_targets(
        path,
        tmp_path / "rates.npy",
        time_start_ms=0.0,
        time_end_ms=200.0,
        target_fs=100,
    )
    assert targets.shape == (6, 20, 3)

    # Sample index of stimulus onset is pre_onset; 10 ms bins at 100 Hz, and a
    # 1 kHz spike count averaged over a bin and scaled by 1,000 is a rate in Hz.
    expected = rasters[:, :, 50:250].reshape(3, 6, 20, 10).mean(axis=3) * 1000.0
    np.testing.assert_allclose(
        np.asarray(targets), expected.transpose(1, 2, 0), rtol=1e-5
    )

    sidecar = np.load(tmp_path / "rates.meta.npz")
    np.testing.assert_array_equal(sidecar["trial_image_ids"], trial_image_ids)
# EOF


def test_baseline_subtraction_removes_pre_onset_rate(tmp_path):
    trial_image_ids = np.asarray([1, 2, 1, 2], dtype=np.int64)
    path = tmp_path / "GoodUnit_240629_Synthetic_NSD1000_LOC_g0.mat"
    rasters = _write_synthetic_good_unit(path, 2, 50, 100, trial_image_ids)

    targets = preprocess_triple_n_firing_rate_targets(
        path,
        tmp_path / "rates_bc.npy",
        time_start_ms=0.0,
        time_end_ms=100.0,
        target_fs=100,
        subtract_baseline=True,
        baseline_start_ms=-50.0,
        baseline_end_ms=0.0,
    )

    baseline = rasters[:, :, 0:50].mean(axis=2) * 1000.0
    response = rasters[:, :, 50:150].reshape(2, 4, 10, 10).mean(axis=3) * 1000.0
    expected = response - baseline[:, :, None]
    np.testing.assert_allclose(
        np.asarray(targets), expected.transpose(1, 2, 0), rtol=1e-5
    )
# EOF


def test_partial_bin_window_is_rejected(tmp_path):
    trial_image_ids = np.asarray([1, 2], dtype=np.int64)
    path = tmp_path / "GoodUnit_240629_Synthetic_NSD1000_LOC_g0.mat"
    _write_synthetic_good_unit(path, 1, 50, 100, trial_image_ids)

    with pytest.raises(ValueError, match="complete"):
        preprocess_triple_n_firing_rate_targets(
            path,
            tmp_path / "bad.npy",
            time_start_ms=0.0,
            time_end_ms=95.0,
            target_fs=100,
        )
    # end with expected binning failure
# EOF


def test_image_metadata_splits_by_image_not_presentation():
    # Every image is repeated, so a held-out image must take all of its
    # repetitions with it; that is the property this split has to guarantee.
    trial_image_ids = np.repeat(np.arange(1, 201, dtype=np.int64), 4)
    metadata, train_image_ids, test_image_ids = build_triple_n_image_metadata(
        trial_image_ids, n_test_images=20, random_seed=0
    )

    assert metadata.shape == (len(trial_image_ids), 6)
    assert len(test_image_ids) == 20

    is_test = metadata[:, 2] > 0
    is_train = metadata[:, 1] > 0
    assert np.all(is_test ^ is_train)
    np.testing.assert_array_equal(
        np.unique(trial_image_ids[is_test]), test_image_ids
    )
    # No image may appear on both sides of the split.
    assert not set(np.unique(trial_image_ids[is_test])) & set(
        np.unique(trial_image_ids[is_train])
    )
    # Test ids are a contiguous one-based range, as TVSDTrialDataset expects.
    np.testing.assert_array_equal(
        np.unique(metadata[is_test, 2]), np.arange(1, 21)
    )
# EOF


def test_localizer_stimuli_are_excluded():
    trial_image_ids = np.concatenate(
        [
            np.repeat(np.arange(1, 51, dtype=np.int64), 2),
            np.asarray([1001, 1005, 1072], dtype=np.int64),
        ]
    )
    metadata, _, _ = build_triple_n_image_metadata(
        trial_image_ids, n_test_images=5, random_seed=0
    )
    localizer_rows = trial_image_ids > TRIPLE_N_N_NSD_IMAGES
    assert np.all(metadata[localizer_rows, 1] == 0)
    assert np.all(metadata[localizer_rows, 2] == 0)
# EOF


def test_unit_selection_by_type_and_reliability():
    summary = {
        "UnitType": np.asarray([1, 1, 2, 3, 1]),
        "reliability_best": np.asarray([0.9, 0.1, 0.8, 0.9, 0.5]),
    }
    np.testing.assert_array_equal(
        select_triple_n_units(summary, unit_types=(1,), min_reliability=0.4),
        np.asarray([0, 4]),
    )
    np.testing.assert_array_equal(
        select_triple_n_units(summary, unit_types=(1, 2), min_reliability=None),
        np.asarray([0, 1, 2, 4]),
    )
# EOF


def test_processed_summary_reads_v73_and_labels_units(tmp_path):
    path = tmp_path / "Processed_ses01_240629_M1_2.mat"
    with h5py.File(path, "w") as h5file:
        # MATLAB writes [1 x unit_num] rows as [unit_num x 1] under h5py.
        h5file.create_dataset("UnitType", data=np.asarray([[1.0, 2.0, 3.0]]))
        h5file.create_dataset(
            "reliability_best", data=np.asarray([[0.7, 0.2, 0.1]])
        )
    # end with synthetic processed file

    summary = load_triple_n_processed(path)
    np.testing.assert_array_equal(
        summary["unit_type_labels"],
        np.asarray(["single_unit", "mua", "non_somatic"]),
    )
    assert summary["reliability_best"].shape == (3,)
# EOF


def test_reliability_helper_accepts_triple_n_target_axes(tmp_path):
    # The project's split-half reliability is shape-generic over the trailing
    # axes, so it must run unchanged on [trials, time, units] firing rates.
    trial_image_ids = np.repeat(np.arange(1, 6, dtype=np.int64), 6)
    path = tmp_path / "GoodUnit_240629_Synthetic_NSD1000_LOC_g0.mat"
    _write_synthetic_good_unit(path, 4, 50, 100, trial_image_ids)

    targets = preprocess_triple_n_firing_rate_targets(
        path, tmp_path / "rates.npy", 0.0, 100.0, 100
    )
    ceiling = split_half_reliability(
        np.asarray(targets),
        trial_image_ids - 1,
        n_images=5,
        n_resamples=4,
        seed=0,
    )
    assert ceiling.shape == (10, 4)
    assert np.all((ceiling >= 0.0) & (ceiling <= 1.0) | np.isnan(ceiling))
# EOF


def test_datasets_split_images_without_leakage():
    # 200 images x 4 repetitions, with a distinct feature vector per image so a
    # leaked stimulus would be detectable by identity alone.
    trial_image_ids = np.repeat(np.arange(1, 201, dtype=np.int64), 4)
    metadata, train_image_ids, test_image_ids = build_triple_n_image_metadata(
        trial_image_ids, n_test_images=20, random_seed=0
    )
    targets = np.zeros((len(trial_image_ids), 5, 3), dtype=np.float32)
    targets += trial_image_ids[:, None, None].astype(np.float32)
    stimulus_features = np.arange(200 * 2 * 4, dtype=np.float32).reshape(200, 2, 4)

    datasets, indices, (unit_mean, unit_scale), (train_inputs, test_inputs) = (
        build_triple_n_datasets(
            targets,
            metadata,
            stimulus_features,
            train_image_ids,
            test_image_ids,
            validation_fraction=0.2,
            random_seed=0,
        )
    )

    # Features must be reordered to the metadata's position columns, not left
    # in raw id order, or every decoder would train on mismatched stimuli.
    np.testing.assert_allclose(
        train_inputs, stimulus_features[train_image_ids - 1]
    )
    np.testing.assert_allclose(
        test_inputs, stimulus_features[test_image_ids - 1]
    )

    fit_images = set(np.unique(trial_image_ids[indices["train"]]).tolist())
    validation_images = set(
        np.unique(trial_image_ids[indices["validation"]]).tolist()
    )
    test_images = set(np.unique(trial_image_ids[indices["test"]]).tolist())
    assert not fit_images & validation_images
    assert not fit_images & test_images
    assert not validation_images & test_images
    assert test_images == set(test_image_ids.tolist())

    # Every repetition of an image lands in the same pool.
    assert len(fit_images) + len(validation_images) == len(train_image_ids)
    assert len(indices["train"]) + len(indices["validation"]) == 180 * 4

    # A dataset item must pair the right stimulus with the right trial.
    model_input, target = datasets["test"][0]
    first_test_trial = indices["test"][0]
    expected_image = trial_image_ids[first_test_trial]
    np.testing.assert_allclose(
        model_input.numpy(), stimulus_features[expected_image - 1]
    )
    expected_target = np.broadcast_to(
        (float(expected_image) - unit_mean[None, :]) / unit_scale[None, :],
        target.shape,
    )
    np.testing.assert_allclose(target.numpy(), expected_target, rtol=1e-5)
# EOF


def test_nsd_trial_indices_exclude_localizer_rows():
    trial_image_ids = np.concatenate(
        [
            np.repeat(np.arange(1, 51, dtype=np.int64), 2),
            np.asarray([1001, 1072], dtype=np.int64),
        ]
    )
    metadata, _, _ = build_triple_n_image_metadata(
        trial_image_ids, n_test_images=5, random_seed=0
    )
    kept = triple_n_nsd_trial_indices(metadata)
    assert len(kept) == 100
    assert np.all(trial_image_ids[kept] <= TRIPLE_N_N_NSD_IMAGES)
# EOF


def test_unit_subset_is_applied_to_targets():
    trial_image_ids = np.repeat(np.arange(1, 31, dtype=np.int64), 3)
    metadata, train_image_ids, test_image_ids = build_triple_n_image_metadata(
        trial_image_ids, n_test_images=5, random_seed=0
    )
    targets = np.random.default_rng(0).normal(
        size=(len(trial_image_ids), 4, 6)
    ).astype(np.float32)
    stimulus_features = np.zeros((30, 2, 3), dtype=np.float32)

    datasets, _, _, _ = build_triple_n_datasets(
        targets,
        metadata,
        stimulus_features,
        train_image_ids,
        test_image_ids,
        unit_indices=np.asarray([0, 3, 5]),
    )
    _, target = datasets["train"][0]
    assert target.shape == (4, 3)
# EOF
