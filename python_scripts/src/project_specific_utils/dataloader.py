import os, sys, yaml
import numpy as np
from pathlib import Path
import h5py
import re
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
3) Optionally slices the signal to a specific brain area
4) Wraps the data in a TimeSeries object
5) Resamples the signal to the target sampling frequency

INPUT:
- paths: dict[str, str] -> dictionary containing base data paths
- cfg: Cfg -> configuration object with required attributes:
    * monkey_name: str
    * date: str
    * new_fs: float
    * brain_area (optional): str

OUTPUT:
- rasters: TimeSeries -> preprocessed neural raster time series
"""
def load_img_natraster(paths: dict[str: str], monkey_name, date, new_fs=None, brain_area=None):
    rasters_path = f"{paths['data_path']}/data/{monkey_name}_natraster{date}.mat"
    with h5py.File(rasters_path, "r") as f:
        rasters = f["natraster"][:]      
    rasters = rasters.astype(np.float32)
    rasters = rasters.transpose(2, 1, 0)
    rasters = TimeSeries(rasters, 1000)
    if brain_area is not None:
            brain_areas_obj = BrainAreas(monkey_name)
            rasters = brain_areas_obj.slice_brain_area(rasters, brain_area)
    # end if brain_area is not None:
    if new_fs is not None:
        rasters.resample(new_fs)
    # if new_fs is not None:
    return rasters
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
