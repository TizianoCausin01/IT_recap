from pathlib import Path

import h5py
import numpy as np
import torch

from IT_recap.tvsd import (
    TVSDTrialDataset,
    compute_tvsd_channel_standardization,
    extract_tvsd_ann_features,
    get_tvsd_monkey_f_area_channels,
    load_tvsd_metadata,
    make_tvsd_headstage_mapping,
    preprocess_tvsd_mua_targets,
)


class _SyntheticAnn:
    """Minimal imgANN-compatible wrapper for ordered extraction testing."""

    def __init__(self):
        self.device = torch.device("cpu")
        self.features = {}

    def get_device(self):
        return self.device

    def set_relevant_layers(self, layer_names):
        self.layer_names = layer_names

    def create_forward_hook(self):
        return None

    def extract_features(self, model_input):
        pixel_values = model_input["pixel_values"]
        self.features = {
            layer_name: pixel_values + layer_idx
            for layer_idx, layer_name in enumerate(self.layer_names)
        }

    def clear_hooks(self):
        return None
# EOC


def _write_synthetic_tvsd(path: Path):
    time_ms = np.arange(-10, 10, dtype=float)
    metadata = np.asarray(
        [
            [1, 1, 0, 1, 1, 1],
            [2, 0, 1, 1, 2, 1],
            [3, 2, 0, 1, 3, 1],
            [4, 0, 1, 2, 4, 1],
        ],
        dtype=float,
    )
    mua = np.zeros((len(time_ms), len(metadata), 1024), dtype=np.float32)
    mua += np.arange(1024, dtype=np.float32)[None, None, :]
    mua[time_ms >= 0] += np.arange(len(metadata), dtype=np.float32)[None, :, None]
    with h5py.File(path, "w") as h5file:
        h5file.create_dataset("ALLMAT", data=metadata.T)
        h5file.create_dataset("ALLMUA", data=mua)
        h5file.create_dataset("tb", data=time_ms[:, None])
    # end with synthetic TVSD file


def test_official_headstage_mapping_and_monkey_f_areas():
    mapping = make_tvsd_headstage_mapping()
    assert np.array_equal(mapping[:96], np.arange(32, 128))
    assert np.array_equal(mapping[96:128], np.arange(0, 32))
    assert np.array_equal(mapping[512:544], np.arange(576, 608))
    assert np.array_equal(np.sort(mapping), np.arange(1024))

    it_channels, it_arrays = get_tvsd_monkey_f_area_channels("IT")
    assert np.array_equal(it_channels, np.arange(512, 832))
    assert it_arrays == (9, 10, 11, 12, 13)


def test_metadata_preprocessing_and_trial_alignment(tmp_path):
    mat_path = tmp_path / "synthetic.mat"
    cache_path = tmp_path / "it_targets.npy"
    _write_synthetic_tvsd(mat_path)

    metadata, time_ms = load_tvsd_metadata(mat_path)
    targets = preprocess_tvsd_mua_targets(
        mat_path=mat_path,
        output_path=cache_path,
        area="IT",
        time_start_ms=0,
        time_end_ms=10,
        target_fs=100,
        baseline_start_ms=-10,
        baseline_end_ms=0,
    )

    assert metadata.shape == (4, 6)
    assert np.array_equal(time_ms, np.arange(-10, 10))
    assert targets.shape == (4, 1, 320)
    assert np.allclose(targets[:, 0, 0], np.arange(4))

    training_indices = np.flatnonzero(metadata[:, 1] > 0)
    channel_mean, channel_scale = compute_tvsd_channel_standardization(
        targets,
        training_indices,
    )
    train_features = np.arange(2 * 2 * 3).reshape(2, 2, 3)
    test_features = np.full((1, 2, 3), 100)
    dataset = TVSDTrialDataset(
        train_inputs=train_features,
        test_inputs=test_features,
        targets=targets,
        metadata=metadata,
        trial_indices=np.arange(4),
        input_mode="activations",
        channel_mean=channel_mean,
        channel_scale=channel_scale,
    )

    first_input, first_target = dataset[0]
    repeated_test_input, _ = dataset[3]
    assert np.array_equal(first_input.numpy(), train_features[0])
    assert np.array_equal(repeated_test_input.numpy(), test_features[0])
    assert first_target.shape == (1, 320)


def test_metadata_rejects_ambiguous_split(tmp_path):
    mat_path = tmp_path / "invalid.mat"
    _write_synthetic_tvsd(mat_path)
    with h5py.File(mat_path, "r+") as h5file:
        metadata = np.asarray(h5file["ALLMAT"])
        metadata[1, 0] = 1
        metadata[2, 0] = 1
        h5file["ALLMAT"][:] = metadata
    # end with invalid synthetic file

    try:
        load_tvsd_metadata(mat_path)
    except ValueError as error:
        assert "exactly one" in str(error)
    else:
        raise AssertionError("Ambiguous train/test metadata was accepted.")
    # end try invalid stimulus split


def test_ordered_ann_feature_extraction(tmp_path):
    output_path = tmp_path / "features.npz"
    train_images = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    test_images = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    layer_names = ["early", "late"]

    train_features, test_features = extract_tvsd_ann_features(
        ann=_SyntheticAnn(),
        train_dataset=train_images,
        test_dataset=test_images,
        layer_names=layer_names,
        output_path=output_path,
        batch_size=2,
        progress_interval=0,
    )

    assert train_features.shape == (3, 2, 4)
    assert test_features.shape == (2, 2, 4)
    assert np.array_equal(train_features[:, 0], train_images.numpy())
    assert np.array_equal(train_features[:, 1], train_images.numpy() + 1)
    with np.load(output_path) as archive:
        assert archive["layer_names"].tolist() == layer_names
        assert np.array_equal(archive["test_features"], test_features)
