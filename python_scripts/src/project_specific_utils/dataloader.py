import os, sys, yaml
import numpy as np
from pathlib import Path
import h5py
import re
import torch
from torch.utils.data import DataLoader, Dataset
ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
with open(PROJECT_ROOT / "config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
sys.path.append(paths["useful_stuff_path"])
from useful_stuff.general_utils.utils import TimeSeries

"""
decode_matlab_strings
Decodes MATLAB strings stored in a v7.3 .mat file (HDF5 format) into Python strings.
1) Iterates over HDF5 object references pointing to MATLAB char arrays
2) Reads the corresponding uint16 character codes
3) Converts character codes to Python characters and joins them into strings

INPUT:
- h5file: h5py.File -> open HDF5 file corresponding to a MATLAB v7.3 .mat file
- ref_array: np.ndarray -> array of HDF5 object references to MATLAB char arrays

OUTPUT:
- strings: list of str -> decoded MATLAB strings
"""
def decode_matlab_strings(h5file, ref_array):
    strings = []
    for ref in ref_array.squeeze():
        chars = h5file[ref][:]
        s = ''.join(chr(c) for c in chars.flatten()) # MATLAB chars are usually stored as Nx1 uint16
        strings.append(s)
    return strings

"""
load_img_natraster
Loads and preprocesses natural image raster data for a given monkey/session.

1) Loads the MATLAB v7.3 natraster file (HDF5 format)
2) Casts data to float32 and reorders axes to (neurons, time, trials)
3) Crops the requested time window at the original sampling frequency
4) Optionally slices the signal to a specific brain area
5) Wraps the data in a TimeSeries object and optionally resamples it

INPUT:
- paths: dict[str, str] -> dictionary containing base data paths
- monkey_name: str -> monkey identifier used in the raster filename
- date: str -> recording date used in the raster filename
- new_fs: float | None -> optional target sampling frequency
- brain_area: str | None -> optional configured cortical-area name
- original_fs: float -> original raster sampling frequency in Hz
- time_start_ms: float -> beginning of the retained time window
- time_end_ms: float | None -> end of the retained time window

OUTPUT:
- rasters: TimeSeries -> preprocessed neural raster time series
"""
def load_img_natraster(
    paths: dict[str: str],
    monkey_name,
    date,
    new_fs=None,
    brain_area=None,
    original_fs=1000,
    time_start_ms=0.0,
    time_end_ms=None,
):
    if original_fs <= 0 or new_fs is not None and new_fs <= 0:
        raise ValueError("Sampling frequencies must be positive.")
    # end if a sampling frequency is invalid

    rasters_path = f"{paths['data_path']}/data/{monkey_name}_natraster{date}.mat"
    with h5py.File(rasters_path, "r") as f:
        raster_dataset = f["natraster"]
        n_source_time = raster_dataset.shape[1]
        time_start = round(time_start_ms * original_fs / 1000.0)
        if time_end_ms is None:
            time_end = n_source_time
        else:
            time_end = round(time_end_ms * original_fs / 1000.0)
        # end if no time-window end was supplied
        if time_start < 0 or time_end <= time_start or time_end > n_source_time:
            raise ValueError(
                f"Requested source samples [{time_start}, {time_end}) from "
                f"a natraster with {n_source_time} time samples."
            )
        # end if the requested time window is invalid
        rasters = raster_dataset[:, time_start:time_end, :]
    # end with natraster file
    rasters = rasters.astype(np.float32)
    rasters = rasters.transpose(2, 1, 0)
    rasters = TimeSeries(rasters, original_fs)
    if brain_area is not None:
        brain_areas_obj = BrainAreas(monkey_name)
        rasters = brain_areas_obj.slice_brain_area(rasters, brain_area)
    # end if brain_area is not None:
    if new_fs is not None:
        rasters.resample(new_fs)
    # end if new_fs is not None:
    return rasters
# EOF


"""
load_img_raster
Load repeated single-trial image responses, slice channels and time, and
downsample each chunk before retaining it in memory.

INPUT:
    - paths: dict -> project paths containing data_path
    - monkey_name: str -> monkey identifier used for brain-area slicing
    - raster_file: str -> raster filename under data_path/data
    - image_names_file: str -> per-trial image-name filename under data_path/data
    - raster_key: str -> HDF5 raster dataset key
    - image_names_key: str -> HDF5 image-name reference dataset key
    - original_fs: int -> raster sampling frequency in Hz
    - new_fs: int | None -> target sampling frequency; None keeps original_fs
    - time_start_ms: float -> beginning of retained time window
    - time_end_ms: float | None -> end of retained time window
    - brain_area: str | None -> optional configured cortical-area name
    - chunk_size: int -> trials read and processed at once

OUTPUT:
    - rasters: TimeSeries -> responses [neurons, time, trials]
    - trial_image_names: list[str] -> one source image name per raster trial
"""
def load_img_raster(
    paths: dict,
    monkey_name: str,
    raster_file: str,
    image_names_file: str,
    raster_key: str = "rasters",
    image_names_key: str = "allimages",
    original_fs: int = 1000,
    new_fs: int | None = None,
    time_start_ms: float = 0.0,
    time_end_ms: float | None = None,
    brain_area: str | None = None,
    chunk_size: int = 512,
) -> tuple[TimeSeries, list[str]]:
    data_dir = Path(paths["data_path"]) / "data"
    with h5py.File(data_dir / image_names_file, "r") as h5file:
        references = h5file[image_names_key][:]
        trial_image_names = decode_matlab_strings(h5file, references)
    # end with image-name file

    if original_fs <= 0 or new_fs is not None and new_fs <= 0:
        raise ValueError("Sampling frequencies must be positive.")
    # end if a sampling frequency is invalid
    if new_fs is not None and (
        new_fs > original_fs or original_fs % new_fs != 0
    ):
        raise ValueError("original_fs must be divisible by new_fs.")
    # end if downsampling frequencies are incompatible
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    # end if chunk size is invalid

    time_start = round(time_start_ms * original_fs / 1000.0)
    with h5py.File(data_dir / raster_file, "r") as h5file:
        raster_dataset = h5file[raster_key]
        n_trials, n_source_time, n_source_neurons = raster_dataset.shape
        if len(trial_image_names) != n_trials:
            raise ValueError(
                f"Found {n_trials} raster trials but "
                f"{len(trial_image_names)} image names."
            )
        # end if raster trials and image names differ

        if time_end_ms is None:
            time_end = n_source_time
        else:
            time_end = round(time_end_ms * original_fs / 1000.0)
        # end if no time-window end was supplied
        if time_start < 0 or time_end <= time_start or time_end > n_source_time:
            raise ValueError(
                f"Requested source samples [{time_start}, {time_end}) from "
                f"a raster with {n_source_time} time samples."
            )
        # end if the requested time window is invalid

        if brain_area is None:
            neuron_ranges = [[0, n_source_neurons]]
        else:
            brain_areas = BrainAreas(monkey_name).get_brain_areas_idx()
            if brain_area not in brain_areas:
                raise KeyError(
                    f"Brain area {brain_area!r} is not configured for "
                    f"{monkey_name!r}."
                )
            # end if the requested brain area is unavailable
            neuron_ranges = brain_areas[brain_area]
        # end if no brain-area slice was requested

        n_neurons = sum(end - start for start, end in neuron_ranges)
        bin_width = 1 if new_fs is None else original_fs // new_fs
        n_time = (time_end - time_start) // bin_width
        if n_time == 0:
            raise ValueError("The requested time window is shorter than one bin.")
        # end if the time window cannot form one output bin
        retained_time_end = time_start + n_time * bin_width
        raster_array = np.empty((n_neurons, n_time, n_trials), dtype=np.float32)

        for trial_start in range(0, n_trials, chunk_size):
            trial_end = min(trial_start + chunk_size, n_trials)
            area_parts = [
                raster_dataset[
                    trial_start:trial_end,
                    time_start:retained_time_end,
                    neuron_start:neuron_end,
                ]
                for neuron_start, neuron_end in neuron_ranges
            ]
            trial_chunk = np.concatenate(area_parts, axis=2).astype(
                np.float32, copy=False
            )

            # Mean non-overlapping source bins before storing the trial chunk.
            if bin_width > 1:
                chunk_shape = (
                    trial_chunk.shape[0],
                    n_time,
                    bin_width,
                    n_neurons,
                )
                trial_chunk = trial_chunk.reshape(chunk_shape).mean(
                    axis=2, dtype=np.float32
                )
            # end if temporal downsampling is requested
            raster_array[:, :, trial_start:trial_end] = trial_chunk.transpose(
                2, 1, 0
            )
        # end for trial chunk
    # end with raster file

    return TimeSeries(raster_array, new_fs or original_fs), trial_image_names
# EOF


"""
map_trial_image_order_to_ann
Map every repeated neural trial to the matching ImageFolder source index.

INPUT:
    - trial_image_names: list[str] -> one stimulus filename per neural trial
    - dataset: ImageFolder -> image dataset in model feature-extraction order

OUTPUT:
    - mapping_idx: np.ndarray -> source image index for every neural trial
"""
def map_trial_image_order_to_ann(trial_image_names, dataset) -> np.ndarray:
    ann_image_names = [os.path.basename(path) for path, _ in dataset.samples]
    if len(ann_image_names) != len(set(ann_image_names)):
        raise ValueError("ImageFolder stimulus basenames must be unique.")
    # end if ImageFolder basenames are duplicated

    normalized_trial_names = list(trial_image_names)
    if os.path.basename(Path(dataset.root)) == "talia_20each_tizi":
        normalized_trial_names = rename_talia_dataset(normalized_trial_names)
    # end if the Talia filenames require normalization

    ann_name_to_idx = {
        image_name: image_idx
        for image_idx, image_name in enumerate(ann_image_names)
    }
    missing_names = sorted(set(normalized_trial_names) - set(ann_name_to_idx))
    if missing_names:
        raise ValueError(
            f"ImageFolder is missing raster stimuli: {missing_names[:10]}."
        )
    # end if a raster stimulus is unavailable
    return np.asarray(
        [ann_name_to_idx[name] for name in normalized_trial_names], dtype=int
    )
# EOF


class NeuralInputDataset(Dataset):
    """
    Pair neural responses with images or cached ANN activations.

    `image_indices` maps each neural trial to the corresponding source image.
    Neural targets are returned in `[time, neurons]` order.

    INPUT (__getitem__):
        - index: int -> neural trial index

    OUTPUT:
        - model_input: torch.Tensor -> image or cached [layers, embedding] features
        - target: torch.Tensor -> neural response [time, neurons]
    """

    """
    __init__
    Validate and store aligned image inputs, ANN features, and neural targets.

    INPUT:
        - image_dataset: Dataset -> source image dataset
        - activations: np.ndarray | None -> cached [images, layers, embedding]
        - neural_activity: np.ndarray -> responses [neurons, time, trials]
        - image_indices: array-like -> source image index for every neural trial
        - input_mode: str -> either "images" or "activations"

    OUTPUT:
        - None
    """
    def __init__(
        self,
        image_dataset,
        activations,
        neural_activity,
        image_indices,
        input_mode,
    ):
        if input_mode not in {"images", "activations"}:
            raise ValueError(
                "input_mode must be either 'images' or 'activations'."
            )
        # end if input mode is invalid

        self.input_mode = input_mode
        self.image_dataset = image_dataset
        neural_activity = np.asarray(neural_activity)
        self.image_indices = np.asarray(image_indices, dtype=int)

        if neural_activity.ndim != 3:
            raise ValueError(
                "neural_activity must have shape [neurons, time, trials]."
            )
        # end if neural activity has the wrong shape
        if self.image_indices.ndim != 1:
            raise ValueError("image_indices must be one-dimensional.")
        # end if image indices have the wrong shape
        if len(self.image_indices) != neural_activity.shape[2]:
            raise ValueError(
                f"Found {len(self.image_indices)} trial-image mappings but "
                f"{neural_activity.shape[2]} neural trials."
            )
        # end if mapped stimuli and neural responses differ
        if self.image_indices.size == 0:
            raise ValueError("image_indices must not be empty.")
        # end if no image indices were provided

        indices_are_invalid = (
            self.image_indices.min() < 0
            or self.image_indices.max() >= len(image_dataset)
        )
        if indices_are_invalid:
            raise IndexError("image_indices exceed the image dataset.")
        # end if an image index is out of bounds

        self.activations = None
        if activations is not None:
            activations = np.asarray(activations)
            if activations.ndim != 3:
                raise ValueError(
                    "Expected activations with shape "
                    "[images, layers, embedding], got "
                    f"{activations.shape}."
                )
            # end if activations have the wrong shape
            if len(image_dataset) != activations.shape[0]:
                raise ValueError(
                    f"Found {len(image_dataset)} images but "
                    f"{activations.shape[0]} activation rows."
                )
            # end if images and activations have different source counts
            self.activations = torch.as_tensor(
                activations,
                dtype=torch.float32,
            )
        # end if activations were provided
        if self.input_mode == "activations" and self.activations is None:
            raise ValueError(
                "activations are required when input_mode='activations'."
            )
        # end if activation mode was requested without activations

        # Targets follow neural trial order: [trials, time, neurons].
        self.neural_activity = torch.as_tensor(
            neural_activity.transpose(2, 1, 0),
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.neural_activity)
    # EOF

    def __getitem__(self, index):
        source_index = self.image_indices[index]
        if self.input_mode == "images":
            model_input, _ = self.image_dataset[source_index]
        else:
            model_input = self.activations[source_index]
        # end if images are requested
        target = self.neural_activity[index]
        return model_input, target
    # EOF
# EOC


"""
make_neural_input_loader
Build an aligned loader that returns images or cached ANN activations.

INPUT:
    - input_mode: str -> either "images" or "activations"
    - image_dataset: Dataset -> image dataset in its original source order
    - activations: np.ndarray | None -> cached [images, layers, embedding]
    - neural_activity: np.ndarray -> responses [neurons, time, trials]
    - image_indices: array-like -> source image index for every trial
    - batch_size: int -> samples per batch
    - shuffle: bool -> whether to shuffle aligned input-target pairs
    - pin_memory: bool -> whether DataLoader pins CPU tensors

OUTPUT:
    - loader: DataLoader -> aligned model-input and neural-target batches
"""
def make_neural_input_loader(
    input_mode,
    image_dataset,
    activations,
    neural_activity,
    image_indices,
    batch_size,
    shuffle,
    pin_memory,
):
    paired_dataset = NeuralInputDataset(
        image_dataset=image_dataset,
        activations=activations,
        neural_activity=neural_activity,
        image_indices=image_indices,
        input_mode=input_mode,
    )
    return DataLoader(
        paired_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )
# EOF


"""
BrainAreas
Utility class for slicing neural data into predefined brain areas.
1) Loads brain-area channel indices from a YAML configuration file
2) Validates input rasters against the expected number of channels
3) Extracts and concatenates channel ranges corresponding to a given brain area

INPUT:
- monkey_name: str -> identifier used to select the correct brain-area mapping

OUTPUT (slice_brain_area):
- brain_area_response: np.ndarray -> subset of rasters corresponding to the selected brain area
"""
class BrainAreas:
    def __init__(self, monkey_name: str):
        self.monkey_name = monkey_name
        with open(PROJECT_ROOT / "brain_areas.yaml", "r") as f:
            config = yaml.safe_load(f)
        try:
            self.areas_idx = config[self.monkey_name]
            self.brain_areas = [k for k in self.areas_idx.keys() if k!='n_chan']
        except KeyError:
            raise KeyError(f"Monkey '{self.monkey_name}' not found.", f"Supported monkeys {list(config.keys())}") from None
        # end try:
    # EOF
    # --- GETTERS ---
    def get_brain_areas_idx(self):
        return self.areas_idx
    #EOF
    def get_brain_areas(self):
        return self.brain_areas
    #EOF
    # --- OTHER FUNCTIONS ---
    def slice_brain_area(self, rasters: "TimeSeries", brain_area_name: str):
        if rasters.get_array().shape[0] < self.areas_idx["n_chan"][0]:
            raise ValueError(
                f"Rasters of shape {rasters.get_array().shape} doesn't match "
                f"the original number of channels ({self.areas_idx['n_chan']})."
            )
        # end if rasters.shape[0] < self.areas_idx["n_chan"][0]:
        try:
            target_brain_area = self.areas_idx[brain_area_name]
        except KeyError:
            raise KeyError(f"Brain area '{brain_area_name}' not found for monkey '{self.monkey_name}'.", f"Supported brain areas: {list(self.areas_idx.keys())}") from None
            
        except TypeError:
            if isinstance(brain_area_name, list) and len(brain_area_name) == 2:
                for idx in brain_area_name:
                    if idx > self.areas_idx["n_chan"][0]:
                        raise ValueError(
                            f"Indices passed {brain_area_name} don't match "
                            f"the original number of channels ({self.areas_idx['n_chan']})."
                        )
                    # end if idx > self.areas_idx["n_chan"][0]:
                # end for idx in brain_area_name:
                target_brain_area = [brain_area_name] # it's setting the limits in terms of channels idx where we don't have precise info about a brain area, wrapping them in a list of lists
            else:
                raise TypeError(f"brain_area_name should be either a str or a list of len 2.")
            # end if isinstance(brain_area_name, list) and len(brain_area_name) == 2:
        # end try:
        brain_area_response = []
        for lims in target_brain_area:
            start, end = lims
            brain_area_response.append(rasters.get_array()[start:end, ...])
        # end for lims in target_brain_area:
        brain_area_response = np.concatenate(brain_area_response)
        brain_area_response = TimeSeries(brain_area_response, rasters.fs)
        return brain_area_response
    # EOF
# EOC


"""
map_image_order_from_ann_to_monkey
Creates an index mapping to align ANN image order with monkey presentation order.

What this function does:
1) Loads the list of images presented to the monkey from a MATLAB file
2) Decodes MATLAB string references into Python strings
3) Removes duplicate image names while preserving order
4) Extracts the ANN image presentation order from the dataset
5) Computes the index mapping from monkey order to ANN order

INPUT:
- paths: dict -> dictionary with base paths
- monkey_name: str -> monkey identifier
- date: str -> experiment date
- dataset: torchvision.datasets.ImageFolder -> ANN image dataset

OUTPUT:
- mapping_idx: list[int] -> indices to reorder ANN features to monkey order
"""
def map_image_order_from_ann_to_monkey(paths, monkey_name, date, dataset):
    allimgs_path = f"{paths['data_path']}/data/{monkey_name}_allimages{date}.mat"
    with h5py.File(allimgs_path, "r") as f:
        try:
            refs = f["allimages"][:]      # shape (N, 1) of object refs
        except KeyError:
            refs = f["uniqueImage"][:]
        # end try:
        monkey_presentation_order = decode_matlab_strings(f, refs)
        monkey_presentation_order = sorted(set(monkey_presentation_order))
    ann_presentation_order = [os.path.basename(path) for path, _ in dataset.samples] # creates the order with which images are presented to the ANN
    if os.path.basename(Path(dataset.root))=="talia_20each_tizi": # little detour because I have changed the filenames for talia_20each_tizi
        monkey_presentation_order = rename_talia_dataset(monkey_presentation_order)
    # end if dataset=="talia_20each_tizi":
    mapping_idx = [ann_presentation_order.index(x) for x in monkey_presentation_order] # Creates a mapping from the monkey to the ann presentation order
    newly_ordered_ann = [ann_presentation_order[i] for i in mapping_idx]
    assert newly_ordered_ann == monkey_presentation_order
    return mapping_idx # by applying this to the ann features we'll get the same order as the monkeys'
# EOF


"""
rename_talia_dataset
just renaming the names the same way I did in the folder also in the uniqueImages file, 
otherwise I wouldn't be able to do the correct mapping. 
We add an underscore between the image name and the number and we take off the spaces.
"""
def rename_talia_dataset(monkey_presentation_order):
    monkey_presentation_order_renamed = []
    for f in monkey_presentation_order:
        # Step 1: insert underscore before first number following a letter
        newname = re.sub(r'([a-zA-Z])([0-9])', r'\1_\2', f)
        # Step 2: remove spaces
        newname = newname.replace(' ', '')
        # Rename if changed
        monkey_presentation_order_renamed.append(newname)
    # end for f in monkey_presentation_order:
    return monkey_presentation_order_renamed
# EOF
