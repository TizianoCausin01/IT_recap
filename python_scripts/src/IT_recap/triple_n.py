"""
Loading and preprocessing for the Triple-N dataset.

Triple-N (Li et al., Nat Neurosci 2026; doi:10.57760/sciencedb.33556) holds
Neuropixels recordings from macaque visual cortex while the animals viewed the
1,000 shared NSD images plus a 72-image localizer set. Unlike the TVSD monkey F
Utah-array recordings used elsewhere in this project, its units are spike sorted
and quality labelled by BombCell, so isolated single units can be separated from
multi-unit clusters. That is what makes the dataset a reference point for asking
whether the TVSD MUA noise floor, rather than the decoder, limits prediction.

Two file types matter here, both per session and both MATLAB files:

    GoodUnit_YYMMDD_Subject_NSD1000_LOC_gx.mat   spike rasters and PSTHs
    Processed_sesXX_YYMMDD_Mx_y.mat              per-unit summaries and labels

The raw H5FILES exports are not hosted (platform file-size limit), so rasters
are read straight from the GoodUnit files here. Layout reference:
https://liyipeng-moon.github.io/Triple-N-Docs/
"""

import ftplib
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat


# The 1,000 NSD shared images carry ids 1-1000; the localizer set uses
# 1001-1072. A failed trial is marked with a zero in trial_valid_idx.
TRIPLE_N_N_NSD_IMAGES = 1000
TRIPLE_N_N_IMAGES = 1072

# BombCell classification stored in the Processed files' UnitType field.
TRIPLE_N_UNIT_TYPES = {
    1: "single_unit",
    2: "mua",
    3: "non_somatic",
    4: "non_somatic",
}


"""
find_matching_key
Look up one HDF5 key while tolerating MATLAB's inconsistent capitalization.

INPUT:
    - group: h5py.Group -> group whose keys are searched
    - name: str -> requested key, compared case-insensitively

OUTPUT:
    - key: str -> the actual key present in the file
"""
def find_matching_key(group, name):
    for key in group.keys():
        if key.lower() == name.lower():
            return key
        # end if this key matches apart from capitalization
    # end for group key
    raise KeyError(f"{name!r} is missing; found {sorted(group.keys())}.")
# EOF


"""
list_triple_n_remote_dir
List one directory of the Triple-N FTP share with file sizes.

Science Data Bank serves the dataset over FTP; the host, port, and credentials
are shown in the "Data File Download" panel of the dataset page after logging
in. Listing before downloading is the cheap way to pick small sessions.

INPUT:
    - host: str -> FTP hostname from the Science Data Bank download panel
    - username: str -> FTP user name
    - password: str -> FTP password
    - remote_dir: str -> directory to list, e.g. "Raw/GoodStruct"
    - port: int -> FTP port

OUTPUT:
    - entries: list[tuple[str, int]] -> (file name, size in bytes) pairs
"""
def list_triple_n_remote_dir(host, username, password, remote_dir, port=21):
    entries = []
    with ftplib.FTP() as ftp_connection:
        ftp_connection.connect(host, port, timeout=60)
        ftp_connection.login(username, password)
        ftp_connection.cwd(remote_dir)
        for file_name in ftp_connection.nlst():
            # SIZE fails on directories; those are reported with size -1 so the
            # caller can still see them without the listing aborting.
            try:
                file_size = ftp_connection.size(file_name)
            except ftplib.error_perm:
                file_size = -1
            # end try remote file size
            entries.append((file_name, file_size if file_size is not None else -1))
        # end for remote entry
    # end with FTP connection
    return sorted(entries)
# EOF


"""
download_triple_n_files
Fetch selected Triple-N files over FTP, skipping anything already present.

INPUT:
    - host: str -> FTP hostname from the Science Data Bank download panel
    - username: str -> FTP user name
    - password: str -> FTP password
    - remote_paths: list[str] -> share-relative paths to download
    - output_dir: Path | str -> local destination directory
    - port: int -> FTP port
    - overwrite: bool -> whether existing local files may be replaced

OUTPUT:
    - local_paths: list[Path] -> downloaded (or already present) local files
"""
def download_triple_n_files(
    host,
    username,
    password,
    remote_paths,
    output_dir,
    port=21,
    overwrite=False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    local_paths = []

    with ftplib.FTP() as ftp_connection:
        ftp_connection.connect(host, port, timeout=60)
        ftp_connection.login(username, password)
        for remote_path in remote_paths:
            local_path = output_dir / Path(remote_path).name
            if local_path.exists() and not overwrite:
                print(f"skip (already local): {local_path.name}")
                local_paths.append(local_path)
                continue
            # end if the file was downloaded earlier

            remote_size = ftp_connection.size(remote_path)
            size_note = (
                f" ({remote_size / 1e9:.2f} GB)" if remote_size else ""
            )
            print(f"downloading {remote_path}{size_note}")

            # Write to a partial file first so an interrupted transfer is never
            # mistaken for a complete download on the next run.
            partial_path = local_path.with_suffix(local_path.suffix + ".part")
            with open(partial_path, "wb") as output_file:
                ftp_connection.retrbinary(
                    f"RETR {remote_path}", output_file.write, blocksize=1 << 20
                )
            # end with partial download
            partial_path.rename(local_path)
            local_paths.append(local_path)
        # end for requested remote file
    # end with FTP connection
    return local_paths
# EOF


"""
load_triple_n_good_unit_metadata
Read the small header variables of a GoodUnit file without touching rasters.

INPUT:
    - good_unit_path: Path | str -> GoodUnit_*.mat path

OUTPUT:
    - trial_image_ids: np.ndarray -> one-based stimulus id per raster row
    - pre_onset_ms: int -> milliseconds of the file stored before stimulus onset
    - n_units: int -> number of units in GoodUnitStrc
    - n_samples: int -> raster samples per trial, at 1 kHz
"""
def load_triple_n_good_unit_metadata(good_unit_path):
    with h5py.File(Path(good_unit_path), "r") as h5file:
        params_key = find_matching_key(h5file, "global_params")
        pre_onset_key = find_matching_key(h5file[params_key], "pre_onset")
        pre_onset_ms = int(
            np.ravel(np.asarray(h5file[params_key][pre_onset_key]))[0]
        )

        meta_group = h5file[find_matching_key(h5file, "meta_data")]
        trial_valid_idx = np.asarray(
            meta_group[find_matching_key(meta_group, "trial_valid_idx")]
        ).reshape(-1).astype(np.int64)

        # The official loader masks trials with dataset_valid_idx; a zero in
        # trial_valid_idx marks a failed trial and is the documented fallback.
        try:
            dataset_valid_key = find_matching_key(meta_group, "dataset_valid_idx")
            valid_mask = np.asarray(meta_group[dataset_valid_key]).reshape(-1)
            valid_mask = valid_mask.astype(bool)
        except KeyError:
            valid_mask = trial_valid_idx != 0
        # end try dataset validity mask

        trial_image_ids = trial_valid_idx[valid_mask]

        unit_group = h5file[find_matching_key(h5file, "GoodUnitStrc")]
        raster_references = unit_group[find_matching_key(unit_group, "Raster")]
        n_units = len(raster_references)
        # Rasters are [trial, time] in MATLAB, hence [time, trial] under h5py.
        n_samples = h5file[raster_references[0][0]].shape[0]
    # end with GoodUnit file

    if len(trial_image_ids) == 0:
        raise ValueError(f"{good_unit_path} contains no valid trials.")
    # end if every trial was rejected
    return trial_image_ids, pre_onset_ms, n_units, n_samples
# EOF


"""
preprocess_triple_n_firing_rate_targets
Stream one GoodUnit file's rasters into a compact firing-rate cache.

Rasters are 1 kHz spike counts, so a bin mean scaled by 1,000 is the firing
rate in spikes per second. Output axes deliberately match the TVSD target cache
([presentations, time bins, channels]) so the same decoder code trains on
either dataset.

INPUT:
    - good_unit_path: Path | str -> GoodUnit_*.mat path
    - output_path: Path | str -> destination .npy cache
    - time_start_ms: float -> inclusive response-window start, onset-relative
    - time_end_ms: float -> exclusive response-window end
    - target_fs: int -> output sampling rate that divides 1,000 Hz
    - subtract_baseline: bool -> whether to remove each trial's pre-onset rate
    - baseline_start_ms: float -> inclusive baseline-window start
    - baseline_end_ms: float -> exclusive baseline-window end
    - overwrite: bool -> whether an existing cache may be replaced

OUTPUT:
    - targets: np.memmap -> [trials, time bins, units] rates in spikes/s
"""
def preprocess_triple_n_firing_rate_targets(
    good_unit_path,
    output_path,
    time_start_ms=0.0,
    time_end_ms=200.0,
    target_fs=100,
    subtract_baseline=False,
    baseline_start_ms=-50.0,
    baseline_end_ms=0.0,
    overwrite=False,
):
    good_unit_path = Path(good_unit_path)
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return np.load(output_path, mmap_mode="r")
    # end if a prepared cache already exists
    if target_fs <= 0 or 1000 % target_fs != 0:
        raise ValueError("target_fs must be a positive divisor of 1,000 Hz.")
    # end if temporal binning is incompatible with the 1 kHz rasters
    if time_end_ms <= time_start_ms:
        raise ValueError("time_end_ms must be greater than time_start_ms.")
    # end if the response window is empty

    trial_image_ids, pre_onset_ms, n_units, n_samples = (
        load_triple_n_good_unit_metadata(good_unit_path)
    )

    # PsthRange runs -pre_onset:1:post_onset, so sample index = pre_onset + t.
    response_start = pre_onset_ms + int(round(time_start_ms))
    response_end = pre_onset_ms + int(round(time_end_ms))
    baseline_start = pre_onset_ms + int(round(baseline_start_ms))
    baseline_end = pre_onset_ms + int(round(baseline_end_ms))
    if response_start < 0 or response_end > n_samples:
        raise ValueError(
            f"Response window [{time_start_ms}, {time_end_ms}) ms falls outside "
            f"the stored {n_samples}-sample epoch."
        )
    # end if the requested response window is not in the file
    if subtract_baseline and (baseline_start < 0 or baseline_end > n_samples):
        raise ValueError("The baseline window falls outside the stored epoch.")
    # end if the requested baseline window is not in the file

    bin_width = 1000 // target_fs
    n_response_samples = response_end - response_start
    if n_response_samples % bin_width != 0:
        raise ValueError(
            f"The {n_response_samples} response samples do not form complete "
            f"{bin_width}-sample bins."
        )
    # end if the response window leaves a partial bin
    n_time_bins = n_response_samples // bin_width
    n_trials = len(trial_image_ids)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    targets = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_trials, n_time_bins, n_units),
    )

    with h5py.File(good_unit_path, "r") as h5file:
        unit_group = h5file[find_matching_key(h5file, "GoodUnitStrc")]
        raster_references = unit_group[find_matching_key(unit_group, "Raster")]

        # One unit at a time keeps peak memory at a single [time, trial] slab
        # instead of the full [trial, time, unit] array.
        for unit_idx in range(n_units):
            raster = h5file[raster_references[unit_idx][0]]
            response = np.asarray(
                raster[response_start:response_end], dtype=np.float32
            )
            if response.shape[1] != n_trials:
                raise ValueError(
                    f"Unit {unit_idx} has {response.shape[1]} raster rows but "
                    f"metadata lists {n_trials} valid trials."
                )
            # end if a unit's raster disagrees with the trial mask

            # [time, trial] -> [bins, bin_width, trial] -> rate per bin in Hz.
            binned = response.reshape(n_time_bins, bin_width, n_trials)
            unit_rate = binned.mean(axis=1) * 1000.0
            if subtract_baseline:
                baseline = np.asarray(
                    raster[baseline_start:baseline_end], dtype=np.float32
                )
                unit_rate = unit_rate - baseline.mean(axis=0) * 1000.0
            # end if the pre-onset rate is removed per trial

            targets[:, :, unit_idx] = unit_rate.T
        # end for unit
    # end with GoodUnit file
    targets.flush()

    # The stimulus ids and window definition travel with the cache; without
    # them the array cannot be split by image or compared across sessions.
    np.savez(
        output_path.with_suffix(".meta.npz"),
        trial_image_ids=trial_image_ids,
        time_start_ms=time_start_ms,
        time_end_ms=time_end_ms,
        target_fs=target_fs,
        subtract_baseline=subtract_baseline,
        source_file=str(good_unit_path.name),
    )
    return np.load(output_path, mmap_mode="r")
# EOF


"""
load_triple_n_processed
Read a Processed_ses*.mat summary, which carries the per-unit quality labels.

The Processed files are small and may be saved either as MATLAB v7.3 (HDF5) or
as the older v7 format, so both readers are attempted.

INPUT:
    - processed_path: Path | str -> Processed_ses*.mat path

OUTPUT:
    - summary: dict -> per-unit arrays, each [n_units]-shaped where applicable
"""
def load_triple_n_processed(processed_path):
    processed_path = Path(processed_path)
    wanted_fields = (
        "UnitType",
        "reliability_basic",
        "reliability_best",
        "response_basic",
        "response_best",
        "snr",
        "snr_max",
        "pos",
        "mean_psth",
    )

    summary = {}
    if h5py.is_hdf5(processed_path):
        with h5py.File(processed_path, "r") as h5file:
            for field in wanted_fields:
                try:
                    key = find_matching_key(h5file, field)
                except KeyError:
                    continue
                # end try optional field
                # h5py returns MATLAB arrays transposed.
                summary[field] = np.asarray(h5file[key]).T.squeeze()
            # end for requested field
        # end with processed file
    else:
        matlab_data = loadmat(processed_path, squeeze_me=True)
        for field in wanted_fields:
            if field in matlab_data:
                summary[field] = np.asarray(matlab_data[field])
            # end if the field is present
        # end for requested field
    # end if the file is MATLAB v7.3

    if "UnitType" not in summary:
        raise KeyError(f"{processed_path.name} has no UnitType field.")
    # end if quality labels are missing

    summary["unit_type_labels"] = np.array(
        [
            TRIPLE_N_UNIT_TYPES.get(int(code), "unknown")
            for code in np.ravel(summary["UnitType"])
        ]
    )
    return summary
# EOF


"""
select_triple_n_units
Pick unit indices by BombCell class and split-half reliability.

INPUT:
    - summary: dict -> output of load_triple_n_processed
    - unit_types: tuple[int, ...] -> BombCell codes to keep, 1 is single unit
    - min_reliability: float -> lower bound on reliability_best
    - reliability_field: str -> which reliability column to threshold

OUTPUT:
    - unit_indices: np.ndarray -> zero-based indices of the retained units
"""
def select_triple_n_units(
    summary,
    unit_types=(1,),
    min_reliability=0.4,
    reliability_field="reliability_best",
):
    unit_type_codes = np.ravel(summary["UnitType"]).astype(int)
    keep_mask = np.isin(unit_type_codes, np.asarray(unit_types, dtype=int))

    if min_reliability is not None:
        if reliability_field not in summary:
            raise KeyError(f"{reliability_field!r} is absent from the summary.")
        # end if the requested reliability column is missing
        reliability = np.ravel(summary[reliability_field]).astype(float)
        if len(reliability) != len(unit_type_codes):
            raise ValueError("UnitType and reliability lengths disagree.")
        # end if the summary columns are misaligned
        keep_mask &= reliability > min_reliability
    # end if a reliability threshold was requested

    return np.flatnonzero(keep_mask)
# EOF



"""
build_triple_n_image_metadata
Build an ALLMAT-shaped metadata table so TVSDTrialDataset can wrap Triple-N.

TVSD splits train and test by presentation because its 22,248 training images
are shown once. Every Triple-N image is repeated instead, so the split has to
be made over images: a held-out pool of NSD images plays the role of the TVSD
repeated-test set, and the rest supply the training presentations.

Localizer trials keep zeros in both id columns; filter them out with
triple_n_nsd_trial_indices before building a dataset.

INPUT:
    - trial_image_ids: np.ndarray -> one-based stimulus id per trial
    - n_test_images: int -> NSD images reserved for the repeated-test pool
    - random_seed: int -> reproducible choice of the held-out images

OUTPUT:
    - metadata: np.ndarray -> [trials, 6] table with train and test id columns
    - train_image_ids: np.ndarray -> one-based ids in train-column order
    - test_image_ids: np.ndarray -> one-based ids in test-column order
"""
def build_triple_n_image_metadata(
    trial_image_ids,
    n_test_images=100,
    random_seed=0,
):
    trial_image_ids = np.asarray(trial_image_ids, dtype=np.int64)

    # Only the 1,000 shared NSD images are modelled; the localizer stimuli
    # (ids above 1000) are a different experiment and are dropped.
    nsd_mask = (trial_image_ids >= 1) & (trial_image_ids <= TRIPLE_N_N_NSD_IMAGES)
    present_image_ids = np.unique(trial_image_ids[nsd_mask])
    if len(present_image_ids) <= n_test_images:
        raise ValueError(
            f"Only {len(present_image_ids)} NSD images are present; cannot hold "
            f"out {n_test_images}."
        )
    # end if the session cannot support the requested test pool

    split_rng = np.random.default_rng(random_seed)
    test_image_ids = np.sort(
        split_rng.choice(present_image_ids, size=n_test_images, replace=False)
    )
    train_image_ids = np.setdiff1d(present_image_ids, test_image_ids)
    is_test_trial = np.isin(trial_image_ids, test_image_ids)

    # Columns follow TVSD's ALLMAT convention: exactly one of the train and
    # test id columns is non-zero on every NSD trial. The stored value is a
    # one-based position into train_image_ids or test_image_ids, so a caller
    # can order stimulus features to match either column.
    metadata = np.zeros((len(trial_image_ids), 6), dtype=np.int64)
    metadata[:, 0] = np.arange(len(trial_image_ids))

    train_position = {
        image_id: position + 1
        for position, image_id in enumerate(train_image_ids)
    }
    test_position = {
        image_id: position + 1
        for position, image_id in enumerate(test_image_ids)
    }
    for trial_idx, image_id in enumerate(trial_image_ids):
        if not nsd_mask[trial_idx]:
            continue
        # end if this trial showed a localizer stimulus
        if is_test_trial[trial_idx]:
            metadata[trial_idx, 2] = test_position[image_id]
        else:
            metadata[trial_idx, 1] = train_position[image_id]
        # end if the trial belongs to the held-out pool
    # end for trial
    return metadata, train_image_ids, test_image_ids
# EOF


"""
triple_n_nsd_trial_indices
Select the trials that showed one of the 1,000 shared NSD images.

Localizer trials keep zeros in both id columns of the metadata table. Passing
them to TVSDTrialDataset would silently index the last test stimulus, so every
caller must filter with this helper before building a dataset.

INPUT:
    - metadata: np.ndarray -> [trials, 6] table from build_triple_n_image_metadata

OUTPUT:
    - trial_indices: np.ndarray -> rows carrying a train or test stimulus id
"""
def triple_n_nsd_trial_indices(metadata):
    metadata = np.asarray(metadata, dtype=np.int64)
    return np.flatnonzero((metadata[:, 1] > 0) | (metadata[:, 2] > 0))
# EOF


"""
build_triple_n_datasets
Wrap one Triple-N session in the TVSD dataset objects used by the decoders.

The fit/validation division also has to be made over images, not presentations.
Every Triple-N image is repeated several times, so holding out random trials
would leave copies of the same stimulus on both sides and inflate validation
scores. Images are divided first, then trials follow their image.

INPUT:
    - targets: np.ndarray -> [trials, time, units] firing-rate cache
    - metadata: np.ndarray -> [trials, 6] table from build_triple_n_image_metadata
    - stimulus_features: np.ndarray -> [stimuli, layers, embedding] in id order
    - train_image_ids: np.ndarray -> one-based ids in train-column order
    - test_image_ids: np.ndarray -> one-based ids in test-column order
    - validation_fraction: float -> share of training images used for validation
    - random_seed: int -> reproducible fit/validation division
    - unit_indices: np.ndarray | None -> units to keep, or None for all

OUTPUT:
    - datasets: dict -> train, validation, and test TVSDTrialDataset objects
    - indices: dict -> trial indices behind each subset
    - standardization: tuple -> fit-only per-unit mean and scale
    - features: tuple -> ordered train and test stimulus feature arrays
"""
def build_triple_n_datasets(
    targets,
    metadata,
    stimulus_features,
    train_image_ids,
    test_image_ids,
    validation_fraction=0.1,
    random_seed=0,
    unit_indices=None,
):
    from IT_recap.tvsd import (
        TVSDTrialDataset,
        compute_tvsd_channel_standardization,
    )

    metadata = np.asarray(metadata, dtype=np.int64)
    stimulus_features = np.asarray(stimulus_features, dtype=np.float32)
    train_image_ids = np.asarray(train_image_ids, dtype=np.int64)
    test_image_ids = np.asarray(test_image_ids, dtype=np.int64)

    max_image_id = max(int(train_image_ids.max()), int(test_image_ids.max()))
    if len(stimulus_features) < max_image_id:
        raise ValueError(
            f"stimulus_features has {len(stimulus_features)} rows but stimulus "
            f"id {max_image_id} is used; rows must follow one-based ids."
        )
    # end if the feature table does not cover every shown stimulus

    if unit_indices is not None:
        unit_indices = np.asarray(unit_indices, dtype=int)
        if unit_indices.size == 0:
            raise ValueError("unit_indices selected no units.")
        # end if the unit selection is empty
        targets = np.asarray(targets, dtype=np.float32)[:, :, unit_indices]
    # end if a unit subset was requested

    # Feature rows are reordered to match the metadata's position columns, so
    # train_idx - 1 and test_idx - 1 index straight into these arrays.
    train_inputs = stimulus_features[train_image_ids - 1]
    test_inputs = stimulus_features[test_image_ids - 1]

    # Divide training images, never training trials, between fit and validation.
    split_rng = np.random.default_rng(random_seed)
    n_train_images = len(train_image_ids)
    shuffled_positions = split_rng.permutation(n_train_images) + 1
    n_validation_images = max(1, round(n_train_images * validation_fraction))
    validation_positions = np.zeros(n_train_images + 1, dtype=bool)
    validation_positions[shuffled_positions[:n_validation_images]] = True

    train_trials = np.flatnonzero(metadata[:, 1] > 0)
    is_validation_trial = validation_positions[metadata[train_trials, 1]]
    indices = {
        "train": train_trials[~is_validation_trial],
        "validation": train_trials[is_validation_trial],
        "test": np.flatnonzero(metadata[:, 2] > 0),
    }

    unit_mean, unit_scale = compute_tvsd_channel_standardization(
        targets, indices["train"]
    )
    dataset_inputs = {
        "train_inputs": train_inputs,
        "test_inputs": test_inputs,
        "targets": targets,
        "metadata": metadata,
        "input_mode": "activations",
        "channel_mean": unit_mean,
        "channel_scale": unit_scale,
    }
    datasets = {
        subset_name: TVSDTrialDataset(
            **dataset_inputs, trial_indices=subset_indices
        )
        for subset_name, subset_indices in indices.items()
    }
    return (
        datasets,
        indices,
        (unit_mean, unit_scale),
        (train_inputs, test_inputs),
    )
# EOF
