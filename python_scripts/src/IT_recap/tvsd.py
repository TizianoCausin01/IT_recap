from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


TVSD_METADATA_COLUMNS = (
    "trial_idx",
    "train_idx",
    "test_idx",
    "rep",
    "count",
    "day",
)

# Zero-based ranges after applying the official headstage permutation.
TVSD_MONKEY_F_AREAS = {
    "V1": (0, 512),
    "IT": (512, 832),
    "V4": (832, 1024),
}

TVSD_MONKEY_F_ARRAYS = {
    "V1": tuple(range(1, 9)),
    "IT": tuple(range(9, 14)),
    "V4": tuple(range(14, 17)),
}


"""
make_tvsd_headstage_mapping
Reproduce the official TVSD permutation from physical channel position to the
raw ALLMUA channel axis. The returned indices are zero-based for NumPy/HDF5.

OUTPUT:
    - raw_channel_indices: np.ndarray -> 1,024 raw indices in physical order
"""
def make_tvsd_headstage_mapping() -> np.ndarray:
    mapped_banks = []

    # The first four 128-channel headstage groups use the same bank order.
    # The final four groups use the Gemini expansion-board bank order.
    for headstage_group in range(8):
        bank_order = (1, 2, 3, 0) if headstage_group < 4 else (2, 1, 0, 3)
        group_start = headstage_group * 128
        for bank_idx in bank_order:
            bank_start = group_start + bank_idx * 32
            mapped_banks.append(np.arange(bank_start, bank_start + 32))
        # end for bank in physical order
    # end for 128-channel headstage group

    raw_channel_indices = np.concatenate(mapped_banks).astype(int)
    if not np.array_equal(np.sort(raw_channel_indices), np.arange(1024)):
        raise RuntimeError("The TVSD headstage map is not a valid permutation.")
    # end if the mapping does not contain each channel exactly once
    return raw_channel_indices
# EOF


"""
load_tvsd_metadata
Load and validate the small metadata variables without reading the 54 GB MUA
array. MATLAB v7.3 axes are transposed into presentation-major Python order.

INPUT:
    - mat_path: Path | str -> f_THINGS_MUA_trials.mat path

OUTPUT:
    - metadata: np.ndarray -> integer rows [presentations, 6]
    - time_ms: np.ndarray -> time of every source MUA sample in milliseconds
"""
def load_tvsd_metadata(mat_path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    mat_path = Path(mat_path)
    with h5py.File(mat_path, "r") as h5file:
        required_keys = {"ALLMAT", "ALLMUA", "tb"}
        missing_keys = required_keys - set(h5file.keys())
        if missing_keys:
            raise KeyError(f"Missing TVSD variables: {sorted(missing_keys)}.")
        # end if a required MATLAB variable is missing

        metadata_float = np.asarray(h5file["ALLMAT"]).T
        time_ms = np.asarray(h5file["tb"]).squeeze().astype(float)
        mua_shape = h5file["ALLMUA"].shape
    # end with TVSD file

    expected_metadata_shape = (mua_shape[1], len(TVSD_METADATA_COLUMNS))
    if metadata_float.shape != expected_metadata_shape:
        raise ValueError(
            f"Expected ALLMAT shape {expected_metadata_shape}, got "
            f"{metadata_float.shape}."
        )
    # end if presentation metadata and MUA differ
    if time_ms.ndim != 1 or len(time_ms) != mua_shape[0]:
        raise ValueError(
            f"Expected {mua_shape[0]} time values, got {time_ms.shape}."
        )
    # end if the time base and MUA differ
    if not np.allclose(metadata_float, np.round(metadata_float)):
        raise ValueError("ALLMAT contains non-integer metadata values.")
    # end if MATLAB identifiers are not integers

    metadata = np.round(metadata_float).astype(np.int64)
    train_idx = metadata[:, 1]
    test_idx = metadata[:, 2]
    if not np.all((train_idx == 0) ^ (test_idx == 0)):
        raise ValueError(
            "Each presentation must contain exactly one train_idx or test_idx."
        )
    # end if the stimulus split is ambiguous
    return metadata, time_ms
# EOF


"""
get_tvsd_monkey_f_area_channels
Return physical channel indices and Utah-array numbers for one monkey F ROI.

INPUT:
    - area: str -> one of V1, V4, or IT

OUTPUT:
    - physical_channels: np.ndarray -> zero-based physical channel indices
    - array_numbers: tuple[int, ...] -> one-based array identifiers
"""
def get_tvsd_monkey_f_area_channels(
    area: str,
) -> tuple[np.ndarray, tuple[int, ...]]:
    area = area.upper()
    if area not in TVSD_MONKEY_F_AREAS:
        raise KeyError(
            f"Unknown monkey F area {area!r}; choose from "
            f"{tuple(TVSD_MONKEY_F_AREAS)}."
        )
    # end if the requested area is unavailable

    channel_start, channel_end = TVSD_MONKEY_F_AREAS[area]
    physical_channels = np.arange(channel_start, channel_end)
    return physical_channels, TVSD_MONKEY_F_ARRAYS[area]
# EOF


"""
decode_matlab_char_reference
Decode one MATLAB v7.3 character-array reference into a Python string.

INPUT:
    - h5file: h5py.File -> open MATLAB v7.3 file
    - reference: h5py.Reference -> reference to a uint16 MATLAB char array

OUTPUT:
    - value: str -> decoded string
"""
def decode_matlab_char_reference(h5file, reference) -> str:
    character_codes = np.asarray(h5file[reference]).reshape(-1)
    return "".join(chr(int(code)) for code in character_codes)
# EOF


"""
load_tvsd_stimulus_paths
Load THINGS paths in the exact one-based train_idx/test_idx order used by TVSD.

INPUT:
    - things_metadata_path: Path | str -> official things_imgs.mat path
    - split: str -> either "train" or "test"

OUTPUT:
    - relative_paths: list[str] -> platform-neutral THINGS-relative paths
"""
def load_tvsd_stimulus_paths(
    things_metadata_path: Path | str,
    split: str,
) -> list[str]:
    if split not in {"train", "test"}:
        raise ValueError("split must be either 'train' or 'test'.")
    # end if the split is invalid

    dataset_key = f"{split}_imgs/things_path"
    with h5py.File(things_metadata_path, "r") as h5file:
        if dataset_key not in h5file:
            raise KeyError(f"{dataset_key!r} is missing from things_imgs.mat.")
        # end if the stimulus-path field is missing
        references = np.asarray(h5file[dataset_key]).reshape(-1)
        relative_paths = [
            decode_matlab_char_reference(h5file, reference).replace("\\", "/")
            for reference in references
        ]
    # end with things_imgs metadata
    return relative_paths
# EOF


"""
preprocess_tvsd_mua_targets
Stream one physically ordered ROI from ALLMUA, subtract each presentation's
pre-stimulus baseline, bin time, and save a compact float32 NumPy cache.

INPUT:
    - mat_path: Path | str -> f_THINGS_MUA_trials.mat path
    - output_path: Path | str -> destination .npy cache
    - area: str -> monkey F ROI (V1, V4, or IT)
    - time_start_ms: float -> inclusive response-window start
    - time_end_ms: float -> exclusive response-window end
    - target_fs: int -> output sampling rate that divides 1,000 Hz
    - baseline_start_ms: float -> inclusive baseline-window start
    - baseline_end_ms: float -> exclusive baseline-window end
    - chunk_size: int -> presentations processed per HDF5 read
    - overwrite: bool -> whether an existing cache may be replaced

OUTPUT:
    - targets: np.memmap -> [presentations, time bins, physical channels]
"""
def preprocess_tvsd_mua_targets(
    mat_path: Path | str,
    output_path: Path | str,
    area: str,
    time_start_ms: float,
    time_end_ms: float,
    target_fs: int,
    baseline_start_ms: float = -100.0,
    baseline_end_ms: float = 0.0,
    chunk_size: int = 64,
    overwrite: bool = False,
) -> np.memmap:
    mat_path = Path(mat_path)
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return np.load(output_path, mmap_mode="r")
    # end if a prepared cache already exists
    if target_fs <= 0 or 1000 % target_fs != 0:
        raise ValueError("target_fs must be a positive divisor of 1,000 Hz.")
    # end if temporal binning is incompatible with the source data
    if time_end_ms <= time_start_ms:
        raise ValueError("time_end_ms must be greater than time_start_ms.")
    # end if the response window is empty
    if baseline_end_ms <= baseline_start_ms:
        raise ValueError(
            "baseline_end_ms must be greater than baseline_start_ms."
        )
    # end if the baseline window is empty
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    # end if the chunk size is invalid

    _, time_ms = load_tvsd_metadata(mat_path)
    source_step_ms = float(np.median(np.diff(time_ms)))
    if not np.isclose(source_step_ms, 1.0):
        raise ValueError(
            f"Expected 1 kHz MUA samples, found {source_step_ms:g} ms steps."
        )
    # end if the source time base is unexpected

    response_mask = (time_ms >= time_start_ms) & (time_ms < time_end_ms)
    baseline_mask = (
        (time_ms >= baseline_start_ms) & (time_ms < baseline_end_ms)
    )
    response_indices = np.flatnonzero(response_mask)
    baseline_indices = np.flatnonzero(baseline_mask)
    if len(response_indices) == 0 or len(baseline_indices) == 0:
        raise ValueError("The requested response or baseline window is absent.")
    # end if either time window lies outside the file
    if not np.all(np.diff(response_indices) == 1) or not np.all(
        np.diff(baseline_indices) == 1
    ):
        raise ValueError("Response and baseline windows must be contiguous.")
    # end if a requested time window has gaps

    bin_width = 1000 // target_fs
    if len(response_indices) % bin_width != 0:
        raise ValueError(
            f"The {len(response_indices)} response samples do not form complete "
            f"{bin_width}-sample bins."
        )
    # end if the response window leaves a partial bin

    physical_channels, _ = get_tvsd_monkey_f_area_channels(area)
    raw_channels = make_tvsd_headstage_mapping()[physical_channels]
    raw_sort_order = np.argsort(raw_channels)
    sorted_raw_channels = raw_channels[raw_sort_order]
    restore_physical_order = np.argsort(raw_sort_order)

    with h5py.File(mat_path, "r") as h5file:
        mua_dataset = h5file["ALLMUA"]
        n_presentations = mua_dataset.shape[1]
        n_time_bins = len(response_indices) // bin_width
        target_shape = (
            n_presentations,
            n_time_bins,
            len(physical_channels),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        targets = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype=np.float32,
            shape=target_shape,
        )

        response_slice = slice(response_indices[0], response_indices[-1] + 1)
        baseline_slice = slice(baseline_indices[0], baseline_indices[-1] + 1)
        for trial_start in range(0, n_presentations, chunk_size):
            trial_end = min(trial_start + chunk_size, n_presentations)
            trial_slice = slice(trial_start, trial_end)

            # HDF5 permits one ordered fancy-index axis. Sort raw channels for
            # reading, then restore official physical order in memory.
            baseline = np.asarray(
                mua_dataset[baseline_slice, trial_slice, sorted_raw_channels],
                dtype=np.float32,
            )[:, :, restore_physical_order]
            response = np.asarray(
                mua_dataset[response_slice, trial_slice, sorted_raw_channels],
                dtype=np.float32,
            )[:, :, restore_physical_order]

            # Baseline is presentation- and channel-specific. Non-overlapping
            # response bins retain [presentation, time, channel] organization.
            baseline_mean = baseline.mean(axis=0, dtype=np.float32)
            response = response.reshape(
                n_time_bins,
                bin_width,
                trial_end - trial_start,
                len(physical_channels),
            ).mean(axis=1, dtype=np.float32)
            targets[trial_start:trial_end] = (
                response.transpose(1, 0, 2) - baseline_mean[:, None, :]
            )
        # end for presentation chunk
        targets.flush()
    # end with TVSD file
    return np.load(output_path, mmap_mode="r")
# EOF


"""
compute_tvsd_channel_standardization
Estimate channel mean and scale from training targets only, streaming over a
memory-mapped target cache so validation data never enter normalization.

INPUT:
    - targets: np.ndarray -> [presentations, time, channels] target cache
    - training_indices: array-like -> presentation indices used for fitting
    - chunk_size: int -> training presentations accumulated at once
    - minimum_scale: float -> lower bound for numerically stable scales

OUTPUT:
    - channel_mean: np.ndarray -> [channels] training mean
    - channel_scale: np.ndarray -> [channels] training standard deviation
"""
def compute_tvsd_channel_standardization(
    targets: np.ndarray,
    training_indices,
    chunk_size: int = 256,
    minimum_scale: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    if targets.ndim != 3:
        raise ValueError("targets must have shape [presentations, time, channels].")
    # end if target axes are invalid
    training_indices = np.asarray(training_indices, dtype=int)
    if training_indices.ndim != 1 or training_indices.size == 0:
        raise ValueError("training_indices must be a non-empty vector.")
    # end if no training presentations were supplied
    if training_indices.min() < 0 or training_indices.max() >= len(targets):
        raise IndexError("training_indices exceed the target cache.")
    # end if a training presentation is out of range
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    # end if the accumulation chunk is invalid

    channel_sum = np.zeros(targets.shape[2], dtype=np.float64)
    channel_squared_sum = np.zeros(targets.shape[2], dtype=np.float64)
    observation_count = 0
    for chunk_start in range(0, len(training_indices), chunk_size):
        chunk_indices = training_indices[chunk_start:chunk_start + chunk_size]
        target_chunk = np.asarray(targets[chunk_indices], dtype=np.float64)
        channel_sum += target_chunk.sum(axis=(0, 1))
        channel_squared_sum += np.square(target_chunk).sum(axis=(0, 1))
        observation_count += target_chunk.shape[0] * target_chunk.shape[1]
    # end for training-target chunk

    channel_mean = channel_sum / observation_count
    channel_variance = channel_squared_sum / observation_count - np.square(
        channel_mean
    )
    channel_scale = np.sqrt(np.maximum(channel_variance, 0.0))
    channel_scale = np.maximum(channel_scale, minimum_scale)
    return channel_mean.astype(np.float32), channel_scale.astype(np.float32)
# EOF


class TVSDOrderedImageDataset(Dataset):
    """
    Load THINGS images in the exact train_idx or test_idx order from metadata.

    INPUT (__getitem__):
        - index: int -> zero-based position corresponding to MATLAB ID index + 1

    OUTPUT:
        - image: torch.Tensor -> transformed model input
    """

    """
    __init__
    Validate ordered relative paths against a THINGS image root.

    INPUT:
        - image_root: Path | str -> root containing THINGS category folders
        - relative_paths: list[str] -> ordered paths from things_imgs.mat
        - transform: callable -> image preprocessing function

    OUTPUT:
        - None
    """
    def __init__(self, image_root, relative_paths, transform):
        self.image_root = Path(image_root)
        self.relative_paths = list(relative_paths)
        self.transform = transform
        if not self.relative_paths:
            raise ValueError("relative_paths must not be empty.")
        # end if no ordered stimuli were supplied

    def __len__(self):
        return len(self.relative_paths)
    # EOF

    def __getitem__(self, index):
        image_path = self.image_root / self.relative_paths[index]
        if not image_path.is_file():
            raise FileNotFoundError(f"THINGS image was not found at {image_path}.")
        # end if the ordered stimulus is missing
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            # end if preprocessing is configured
        # end with source image
        return image
    # EOF
# EOC


class TVSDTrialDataset(Dataset):
    """
    Pair each TVSD presentation with the matching static image or ANN feature.

    INPUT (__getitem__):
        - index: int -> local subset index

    OUTPUT:
        - model_input: torch.Tensor -> image or [layers, embedding] features
        - target: torch.Tensor -> standardized MUA [time, physical channels]
    """

    """
    __init__
    Store canonical train/test stimulus sources and selected presentations.

    INPUT:
        - train_inputs: Dataset | np.ndarray -> 22,248 train stimuli
        - test_inputs: Dataset | np.ndarray -> 100 repeated-test stimuli
        - targets: np.ndarray -> [presentations, time, channels] MUA cache
        - metadata: np.ndarray -> ALLMAT rows [presentations, 6]
        - trial_indices: array-like -> presentation subset exposed by this dataset
        - input_mode: str -> either "images" or "activations"
        - channel_mean: np.ndarray -> train-only mean for each channel
        - channel_scale: np.ndarray -> train-only scale for each channel

    OUTPUT:
        - None
    """
    def __init__(
        self,
        train_inputs,
        test_inputs,
        targets,
        metadata,
        trial_indices,
        input_mode,
        channel_mean,
        channel_scale,
    ):
        if input_mode not in {"images", "activations"}:
            raise ValueError("input_mode must be either 'images' or 'activations'.")
        # end if model input mode is invalid
        self.input_mode = input_mode
        self.train_inputs = train_inputs
        self.test_inputs = test_inputs
        self.targets = targets
        self.metadata = np.asarray(metadata, dtype=np.int64)
        self.trial_indices = np.asarray(trial_indices, dtype=int)
        self.channel_mean = np.asarray(channel_mean, dtype=np.float32)
        self.channel_scale = np.asarray(channel_scale, dtype=np.float32)

        if self.metadata.shape != (len(targets), len(TVSD_METADATA_COLUMNS)):
            raise ValueError("metadata and targets have incompatible shapes.")
        # end if neural targets do not align to ALLMAT
        if self.trial_indices.ndim != 1 or self.trial_indices.size == 0:
            raise ValueError("trial_indices must be a non-empty vector.")
        # end if no subset presentations were selected
        if self.trial_indices.min() < 0 or self.trial_indices.max() >= len(targets):
            raise IndexError("trial_indices exceed the target cache.")
        # end if a subset index is out of range
        if self.channel_mean.shape != (targets.shape[2],) or (
            self.channel_scale.shape != (targets.shape[2],)
        ):
            raise ValueError("channel_mean and channel_scale must match channels.")
        # end if channel normalization is misaligned

        max_train_idx = int(self.metadata[:, 1].max())
        max_test_idx = int(self.metadata[:, 2].max())
        if len(train_inputs) < max_train_idx or len(test_inputs) < max_test_idx:
            raise ValueError("Stimulus inputs do not cover ALLMAT identifiers.")
        # end if an indexed stimulus is unavailable

        if self.input_mode == "activations":
            train_shape = np.shape(train_inputs)
            test_shape = np.shape(test_inputs)
            if len(train_shape) != 3 or len(test_shape) != 3:
                raise ValueError(
                    "Activation inputs must have [stimuli, layers, embedding] axes."
                )
            # end if cached feature axes are invalid
            if train_shape[1:] != test_shape[1:]:
                raise ValueError("Train and test activation dimensions differ.")
            # end if feature spaces differ across splits
        # end if cached features are used

    def __len__(self):
        return len(self.trial_indices)
    # EOF

    def __getitem__(self, index):
        trial_idx = self.trial_indices[index]
        train_idx = self.metadata[trial_idx, 1]
        test_idx = self.metadata[trial_idx, 2]
        if train_idx > 0:
            model_input = self.train_inputs[train_idx - 1]
        else:
            model_input = self.test_inputs[test_idx - 1]
        # end if this presentation belongs to the train or repeated-test pool

        if self.input_mode == "activations":
            model_input = torch.as_tensor(model_input, dtype=torch.float32)
        # end if cached ANN features require tensor conversion
        target = np.asarray(self.targets[trial_idx], dtype=np.float32)
        target = (target - self.channel_mean[None, :]) / self.channel_scale[None, :]
        return model_input, torch.from_numpy(target.copy())
    # EOF
# EOC


"""
extract_tvsd_ann_features
Extract selected pooled ANN-layer features once for the ordered train and test
stimuli, then save the arrays needed for efficient BaselineModel training.

INPUT:
    - ann: imgANN -> frozen image-model wrapper
    - train_dataset: Dataset -> 22,248 ordered training images
    - test_dataset: Dataset -> 100 ordered repeated-test images
    - layer_names: list[str] -> hooked ANN layers in model order
    - output_path: Path | str -> destination compressed .npz archive
    - batch_size: int -> images processed per frozen-backbone pass
    - num_workers: int -> image-loading worker count
    - progress_interval: int -> report every N batches; zero disables reports

OUTPUT:
    - train_features: np.ndarray -> [train images, layers, embedding]
    - test_features: np.ndarray -> [test images, layers, embedding]
"""
def extract_tvsd_ann_features(
    ann,
    train_dataset,
    test_dataset,
    layer_names,
    output_path,
    batch_size=64,
    num_workers=0,
    progress_interval=50,
) -> tuple[np.ndarray, np.ndarray]:
    if not layer_names:
        raise ValueError("layer_names must contain at least one layer.")
    # end if no ANN layers were selected
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative.")
    # end if loader configuration is invalid
    if progress_interval < 0:
        raise ValueError("progress_interval must be non-negative.")
    # end if progress reporting is invalid

    ann.set_relevant_layers(list(layer_names))
    ann.create_forward_hook()
    split_features = []
    try:
        for split_name, image_dataset in (
            ("train", train_dataset),
            ("test", test_dataset),
        ):
            loader = DataLoader(
                image_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.device(ann.get_device()).type == "cuda",
            )
            layer_batches = {layer_name: [] for layer_name in layer_names}
            with torch.inference_mode():
                for batch_idx, pixel_values in enumerate(loader, start=1):
                    ann.extract_features(
                        {"pixel_values": pixel_values.to(ann.get_device())}
                    )
                    for layer_name in layer_names:
                        layer_batches[layer_name].append(
                            ann.features[layer_name].detach().cpu().numpy()
                        )
                    # end for selected layer
                    should_report = progress_interval > 0 and (
                        batch_idx % progress_interval == 0
                        or batch_idx == len(loader)
                    )
                    if should_report:
                        processed_images = min(
                            batch_idx * batch_size,
                            len(image_dataset),
                        )
                        print(
                            f"{split_name}: extracted {processed_images:,}/"
                            f"{len(image_dataset):,} images"
                        )
                    # end if this batch should be reported
                # end for image batch
            # end with frozen feature extraction
            split_features.append(
                np.stack(
                    [
                        np.concatenate(layer_batches[layer_name], axis=0)
                        for layer_name in layer_names
                    ],
                    axis=1,
                ).astype(np.float32, copy=False)
            )
        # end for train and test image datasets
    finally:
        ann.clear_hooks()
    # end try feature hooks

    train_features, test_features = split_features
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        train_features=train_features,
        test_features=test_features,
        layer_names=np.asarray(layer_names),
    )
    return train_features, test_features
# EOF
