import argparse
import csv
import json
import os
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_SRC = PROJECT_ROOT / "python_scripts" / "src"
sys.path.insert(0, str(PROJECT_SRC))

from model_classes.temporal_models import TemporalNaiveLayerAttention  # noqa: E402


@dataclass
class Cfg:
    # Environment and output paths.
    env: str | None = None
    data_root: str | None = None
    stimuli_root: str | None = None
    output_dir: str | None = None

    # Frozen image backbone and hooked transformer layers.
    folder_name: str = "talia_20each_tizi"
    model_name: str = "dino_v3_l"
    pkg: str = "hf"
    repo_url: str | None = None
    model_img_size: int = 224
    pooling: str = "mean"
    layer_names: str | None = None  # Comma-separated; None uses imgANN defaults.
    use_fast_processor: bool = True

    # three0 neural target data.
    monkey_name: str = "three0"
    brain_area: str = "AIT"
    raster_file: str = "rasters_three0_250313to21.mat"
    raster_key: str = "rasters"
    image_names_file: str = "allimages_three0_250313to21.mat"
    image_names_key: str = "allimages"
    original_fs: int = 1000
    target_fs: int = 100
    time_start_ms: float = 0.0
    time_end_ms: float = 300.0

    # Data split and target scaling.
    val_fraction: float = 0.2
    normalize_targets: bool = True

    # Temporal model and optimizer.
    position_embedding_dim: int = 128
    layer_projection_dim: int = 128
    latent_dim: int = 384
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    seed: int = 0
    device: str = "auto"
    save_attention: bool = False
    smoke_test: bool = False


@dataclass
class PreparedData:
    targets: np.ndarray
    image_names: list[str]
    time_values_ms: np.ndarray


class NeuralImageDataset(Dataset):
    """Pair model-preprocessed ImageFolder samples with aligned neural targets."""

    def __init__(self, image_dataset: Dataset, targets: np.ndarray):
        if len(image_dataset) != len(targets):
            raise ValueError(
                f"Image and target counts differ: {len(image_dataset)} and "
                f"{len(targets)}."
            )
        # end if image and target counts differ
        self.image_dataset = image_dataset
        self.targets = torch.from_numpy(targets.astype(np.float32))

    def __len__(self) -> int:
        return len(self.image_dataset)
    # EOF

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # ImageFolder returns (processed_image, class_id); class_id is not needed.
        processed_image, _ = self.image_dataset[index]
        return processed_image, self.targets[index]
    # EOF
# EOC


"""
parse_args
Parse command-line overrides into the training configuration.

OUTPUT:
    - cfg: Cfg -> data, backbone, temporal-model, and optimization parameters
"""
def parse_args() -> Cfg:
    parser = argparse.ArgumentParser(
        description=(
            "Train a position-conditioned layer-attention model with online frozen "
            "DINO feature extraction."
        )
    )
    parser.add_argument("--env", default=Cfg.env)
    parser.add_argument("--data_root", default=Cfg.data_root)
    parser.add_argument("--stimuli_root", default=Cfg.stimuli_root)
    parser.add_argument("--output_dir", default=Cfg.output_dir)
    parser.add_argument("--folder_name", default=Cfg.folder_name)
    parser.add_argument("--model_name", default=Cfg.model_name)
    parser.add_argument("--pkg", default=Cfg.pkg)
    parser.add_argument("--repo_url", default=Cfg.repo_url)
    parser.add_argument("--model_img_size", type=int, default=Cfg.model_img_size)
    parser.add_argument("--pooling", default=Cfg.pooling)
    parser.add_argument("--layer_names", default=Cfg.layer_names)
    parser.add_argument(
        "--use_fast_processor",
        action=argparse.BooleanOptionalAction,
        default=Cfg.use_fast_processor,
    )
    parser.add_argument("--monkey_name", default=Cfg.monkey_name)
    parser.add_argument("--brain_area", default=Cfg.brain_area)
    parser.add_argument("--raster_file", default=Cfg.raster_file)
    parser.add_argument("--raster_key", default=Cfg.raster_key)
    parser.add_argument("--image_names_file", default=Cfg.image_names_file)
    parser.add_argument("--image_names_key", default=Cfg.image_names_key)
    parser.add_argument("--original_fs", type=int, default=Cfg.original_fs)
    parser.add_argument("--target_fs", type=int, default=Cfg.target_fs)
    parser.add_argument("--time_start_ms", type=float, default=Cfg.time_start_ms)
    parser.add_argument("--time_end_ms", type=float, default=Cfg.time_end_ms)
    parser.add_argument("--val_fraction", type=float, default=Cfg.val_fraction)
    parser.add_argument(
        "--normalize_targets",
        action=argparse.BooleanOptionalAction,
        default=Cfg.normalize_targets,
    )
    parser.add_argument(
        "--position_embedding_dim", type=int, default=Cfg.position_embedding_dim
    )
    parser.add_argument(
        "--layer_projection_dim", type=int, default=Cfg.layer_projection_dim
    )
    parser.add_argument("--latent_dim", type=int, default=Cfg.latent_dim)
    parser.add_argument("--batch_size", type=int, default=Cfg.batch_size)
    parser.add_argument("--epochs", type=int, default=Cfg.epochs)
    parser.add_argument("--learning_rate", type=float, default=Cfg.learning_rate)
    parser.add_argument("--weight_decay", type=float, default=Cfg.weight_decay)
    parser.add_argument("--num_workers", type=int, default=Cfg.num_workers)
    parser.add_argument("--seed", type=int, default=Cfg.seed)
    parser.add_argument("--device", default=Cfg.device)
    parser.add_argument("--save_attention", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    return Cfg(**vars(parser.parse_args()))
# EOF


"""
load_project_paths
Load environment-specific paths and expose the configured useful_stuff package.

INPUT:
    - cfg: Cfg -> configuration containing an optional environment override

OUTPUT:
    - paths: dict -> environment-specific project paths
"""
def load_project_paths(cfg: Cfg) -> dict:
    env = cfg.env or os.getenv("MY_ENV", "tiziano_mac_mini")
    with open(PROJECT_ROOT / "config.yaml", "r") as f:
        config = yaml.safe_load(f)
    # end with open config
    if env not in config:
        raise KeyError(f"Environment {env!r} is not present in config.yaml.")
    # end if environment is not configured
    paths = config[env]["paths"]
    sys.path.append(paths["useful_stuff_path"])
    return paths
# EOF


"""
resolve_paths
Resolve neural data, stimulus ImageFolder, and experiment output directories.

INPUT:
    - cfg: Cfg -> training configuration
    - paths: dict -> environment-specific path configuration

OUTPUT:
    - data_root: Path -> IT_recap_local-style root
    - stimuli_root: Path -> ordered ImageFolder stimulus root
    - output_dir: Path -> checkpoint and log directory
"""
def resolve_paths(cfg: Cfg, paths: dict) -> tuple[Path, Path, Path]:
    configured_data_root = paths.get("it_recap_data_path", paths.get("data_path"))
    if cfg.data_root is None and configured_data_root is None:
        raise KeyError("Set --data_root or add data_path to the active environment.")
    # end if data root is not configured
    data_root = Path(cfg.data_root or configured_data_root).expanduser()

    stimuli_root = Path(
        cfg.stimuli_root
        or Path(paths["livingstone_lab"]) / "Stimuli" / cfg.folder_name
    ).expanduser()
    output_dir = Path(
        cfg.output_dir
        or data_root / "results" / "temporally_naive_layer_attention"
    ).expanduser()
    return data_root, stimuli_root, output_dir
# EOF


"""
build_img_ann_and_dataset
Load the frozen imgANN, its official image processor, and the ordered ImageFolder.

INPUT:
    - cfg: Cfg -> image-backbone configuration
    - paths: dict -> configured paths including useful_stuff_path
    - stimuli_root: Path -> stimulus ImageFolder root
    - device: torch.device -> backbone and temporal-model device

OUTPUT:
    - img_ann: imgANN -> frozen backbone wrapper
    - image_dataset: ImageFolder -> processor-transformed image dataset
    - layer_names: list[str] -> ordered layers registered by the temporal model
"""
def build_img_ann_and_dataset(cfg: Cfg, paths: dict, stimuli_root: Path, device):
    if cfg.pkg != "hf":
        raise ValueError("The online training pipeline currently supports pkg='hf'.")
    # end if the package does not use the configured HF processor

    # Import after useful_stuff_path has been added by load_project_paths.
    from transformers import AutoImageProcessor
    from torchvision.datasets import ImageFolder

    from IT_recap.hf_feature_extraction import (
        HF_MODEL_REPOS,
        ProcessorTransform,
        is_valid_image_file,
    )
    from useful_stuff.image_processing.computational_models import imgANN

    repo_url = cfg.repo_url or HF_MODEL_REPOS[cfg.model_name]
    requested_layers = None
    if cfg.layer_names is not None:
        # A comma-separated CLI value provides an explicit ordered layer subset.
        requested_layers = [
            name.strip() for name in cfg.layer_names.split(",") if name.strip()
        ]
        if not requested_layers:
            raise ValueError("--layer_names did not contain any valid layer names.")
        # end if the explicit layer list is empty
    # end if explicit layer names were requested

    # imgANN loads the backbone in evaluation mode; its hooks are created by the
    # TemporalNaiveLayerAttention constructor rather than here.
    img_ann = imgANN(
        model_name=cfg.model_name,
        pkg=cfg.pkg,
        img_size=cfg.model_img_size,
        relevant_layers=requested_layers,
        pooling=cfg.pooling,
        dtype=torch.float32,
        repo_url=repo_url,
        device=device,
    )
    layer_names = list(img_ann.get_relevant_layers())

    # Apply the checkpoint author's preprocessing to every image.
    processor = AutoImageProcessor.from_pretrained(
        repo_url, use_fast=cfg.use_fast_processor
    )
    image_dataset = ImageFolder(
        root=stimuli_root,
        transform=ProcessorTransform(processor),
        is_valid_file=is_valid_image_file,
        allow_empty=True,
    )
    if len(image_dataset) == 0:
        raise ValueError(f"No images were found in {stimuli_root}.")
    # end if the stimulus dataset is empty

    # Catch processor/checkpoint size mismatches before starting a long training run.
    sample_shape = tuple(image_dataset[0][0].shape[-2:])
    if sample_shape != (cfg.model_img_size, cfg.model_img_size):
        raise ValueError(
            f"Processor returned {sample_shape}, expected "
            f"{(cfg.model_img_size, cfg.model_img_size)}."
        )
    # end if the processor returns an unexpected image size
    return img_ann, image_dataset, layer_names
# EOF


def imagefolder_names(image_dataset) -> list[str]:
    """Return unique stimulus basenames in the exact ImageFolder feature order."""
    image_names = [Path(path).name for path, _ in image_dataset.samples]
    if len(image_names) != len(set(image_names)):
        raise ValueError("Stimulus basenames must be unique to align neural targets.")
    # end if stimulus basenames are duplicated
    return image_names
# EOF


"""
decode_matlab_strings
Decode MATLAB char arrays referenced by a v7.3 HDF5 dataset.

INPUT:
    - h5file: h5py.File -> open MATLAB v7.3 file
    - references: np.ndarray -> HDF5 references to MATLAB char arrays

OUTPUT:
    - strings: list[str] -> decoded strings
"""
def decode_matlab_strings(h5file: h5py.File, references: np.ndarray) -> list[str]:
    strings = []
    for reference in references.squeeze():
        character_codes = h5file[reference][:].flatten()
        strings.append("".join(chr(code) for code in character_codes))
    # end for reference
    return strings
# EOF


def normalize_talia_name(name: str) -> str:
    """Match neural filenames to the renamed talia ImageFolder files."""
    name = re.sub(r"([a-zA-Z])(\d)", r"\1_\2", name)
    return name.replace(" ", "")
# EOF


"""
load_brain_area_ranges
Load the configured neuron ranges for one monkey and brain area.

INPUT:
    - monkey_name: str -> monkey key in brain_areas.yaml
    - brain_area: str -> requested cortical area

OUTPUT:
    - ranges: list[list[int]] -> half-open neuron index ranges
"""
def load_brain_area_ranges(monkey_name: str, brain_area: str) -> list[list[int]]:
    with open(PROJECT_ROOT / "brain_areas.yaml", "r") as f:
        brain_areas = yaml.safe_load(f)
    # end with open brain areas
    try:
        return brain_areas[monkey_name][brain_area]
    except KeyError:
        raise KeyError(
            f"Brain area {brain_area!r} is not configured for {monkey_name!r}."
        ) from None
    # end try
# EOF


"""
temporal_bin_mean
Downsample image x time x neuron responses with non-overlapping mean bins.

INPUT:
    - responses: np.ndarray -> responses [images, original_time, neurons]
    - original_fs: int -> source sampling rate in Hz
    - target_fs: int -> target sampling rate in Hz

OUTPUT:
    - binned: np.ndarray -> responses [images, target_time, neurons]
"""
def temporal_bin_mean(
    responses: np.ndarray, original_fs: int, target_fs: int
) -> np.ndarray:
    if target_fs <= 0 or target_fs > original_fs or original_fs % target_fs != 0:
        raise ValueError("original_fs must be divisible by a positive target_fs.")
    # end if sampling rates are incompatible
    bin_width = original_fs // target_fs
    n_complete_bins = responses.shape[1] // bin_width
    if n_complete_bins == 0:
        raise ValueError("The neural window is shorter than one output bin.")
    # end if no complete bins exist
    responses = responses[:, : n_complete_bins * bin_width]
    new_shape = (responses.shape[0], n_complete_bins, bin_width, responses.shape[2])
    return responses.reshape(new_shape).mean(axis=2, dtype=np.float32)
# EOF


"""
load_neural_targets
Load AIT rasters, average repeated presentations, and align them to ImageFolder.

INPUT:
    - data_root: Path -> root containing the neural data directory
    - cfg: Cfg -> neural dataset and temporal preprocessing parameters
    - image_names: list[str] -> target ImageFolder order

OUTPUT:
    - targets: np.ndarray -> image-averaged responses [images, time, neurons]
    - time_values_ms: np.ndarray -> output-bin centers in milliseconds
"""
def load_neural_targets(
    data_root: Path, cfg: Cfg, image_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    # Decode one stimulus name for every recorded presentation.
    image_names_path = data_root / "data" / cfg.image_names_file
    with h5py.File(image_names_path, "r") as h5file:
        references = h5file[cfg.image_names_key][:]
        trial_names = [
            normalize_talia_name(name)
            for name in decode_matlab_strings(h5file, references)
        ]
    # end with image names file

    # Convert the requested millisecond interval into source-sample indices.
    time_start = round(cfg.time_start_ms * cfg.original_fs / 1000.0)
    time_end = round(cfg.time_end_ms * cfg.original_fs / 1000.0)
    if time_start < 0 or time_end <= time_start:
        raise ValueError("time_start_ms and time_end_ms define an invalid window.")
    # end if neural time window is invalid

    # Read only the requested time window and cortical channels from the large file.
    neuron_ranges = load_brain_area_ranges(cfg.monkey_name, cfg.brain_area)
    raster_path = data_root / "data" / cfg.raster_file
    with h5py.File(raster_path, "r") as h5file:
        raster_dataset = h5file[cfg.raster_key]
        if raster_dataset.shape[0] != len(trial_names):
            raise ValueError(
                "Raster trials and image names differ: "
                f"{raster_dataset.shape[0]} and {len(trial_names)}."
            )
        # end if raster trials and image names differ
        if time_end > raster_dataset.shape[1]:
            raise ValueError(
                f"Requested sample {time_end}, but rasters have "
                f"{raster_dataset.shape[1]} time samples."
            )
        # end if requested time exceeds the raster
        area_parts = [
            raster_dataset[:, time_start:time_end, start:end]
            for start, end in neuron_ranges
        ]
        trial_responses = np.concatenate(area_parts, axis=2).astype(np.float32)
    # end with raster file

    # Collect every repeated presentation index for each unique image.
    name_to_indices = {}
    for trial_idx, name in enumerate(trial_names):
        name_to_indices.setdefault(name, []).append(trial_idx)
    # end for trial_idx, name
    missing_names = sorted(set(image_names) - set(name_to_indices))
    if missing_names:
        raise ValueError(f"Neural rasters are missing stimuli: {missing_names[:10]}.")
    # end if stimuli are missing

    # Average repetitions and reorder neural responses to match ImageFolder exactly.
    image_responses = np.stack(
        [trial_responses[name_to_indices[name]].mean(axis=0) for name in image_names],
        axis=0,
    ).astype(np.float32)
    targets = temporal_bin_mean(image_responses, cfg.original_fs, cfg.target_fs)

    # Label each downsampled response by the center of its temporal bin.
    bin_width_ms = 1000.0 / cfg.target_fs
    time_values_ms = (
        cfg.time_start_ms
        + 0.5 * bin_width_ms
        + np.arange(targets.shape[1]) * bin_width_ms
    ).astype(np.float32)
    return targets, time_values_ms
# EOF


"""
split_and_normalize_targets
Split images and fit per-neuron target normalization on training images only.

INPUT:
    - targets: np.ndarray -> neural targets [images, time, neurons]
    - cfg: Cfg -> validation fraction, normalization, and seed

OUTPUT:
    - arrays: dict -> normalized targets, split indices, mean, and standard deviation
"""
def split_and_normalize_targets(targets: np.ndarray, cfg: Cfg) -> dict:
    if not 0.0 < cfg.val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")
    # end if validation fraction is invalid

    # Split unique images, preventing repeated neural trials from leaking across sets.
    rng = np.random.default_rng(cfg.seed)
    shuffled_indices = rng.permutation(targets.shape[0])
    n_val = max(1, round(len(shuffled_indices) * cfg.val_fraction))
    val_indices = np.sort(shuffled_indices[:n_val])
    train_indices = np.sort(shuffled_indices[n_val:])
    if len(train_indices) == 0:
        raise ValueError("The validation split leaves no training images.")
    # end if training split is empty

    normalized_targets = targets.copy()
    target_mean = np.zeros((1, 1, targets.shape[2]), dtype=np.float32)
    target_std = np.ones_like(target_mean)
    if cfg.normalize_targets:
        # One global mean/std per neuron preserves temporal response structure.
        target_mean = targets[train_indices].mean(axis=(0, 1), keepdims=True)
        target_std = targets[train_indices].std(axis=(0, 1), keepdims=True)
        target_std[target_std < 1e-6] = 1.0
        normalized_targets = (targets - target_mean) / target_std
    # end if normalize targets

    return {
        "targets": normalized_targets.astype(np.float32),
        "train_indices": train_indices,
        "val_indices": val_indices,
        "target_mean": target_mean.astype(np.float32),
        "target_std": target_std.astype(np.float32),
    }
# EOF


def choose_device(device_name: str) -> torch.device:
    """Select CUDA, MPS, or CPU when device_name is auto."""
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
# EOF


def set_random_seed(seed: int):
    """Set Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # end if CUDA is available
# EOF


"""
run_epoch
Run one training or validation epoch and return image-averaged MSE.

INPUT:
    - model: nn.Module -> integrated imgANN plus temporal layer-attention model
    - loader: DataLoader -> processed image and neural-target batches
    - loss_function: nn.Module -> prediction loss
    - device: torch.device -> shared backbone and temporal-model device
    - optimizer: torch.optim.Optimizer | None -> None selects validation mode

OUTPUT:
    - mean_loss: float -> sample-weighted epoch loss
"""
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    n_samples = 0
    for images, targets in loader:
        # Both backbone inputs and neural targets live on the shared device.
        images = images.to(device)
        targets = targets.to(device)
        with torch.set_grad_enabled(is_training):
            pred, _, _ = model(images, return_attention=False)
            if pred is None:
                raise RuntimeError("Training requires a prediction head.")
            # end if prediction head is missing
            loss = loss_function(pred, targets)
            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            # end if training
        # end with grad enabled
        total_loss += loss.item() * images.shape[0]
        n_samples += images.shape[0]
    # end for batch
    return total_loss / n_samples
# EOF


"""
save_checkpoint
Save temporal parameters, optimizer state, normalization, split, and metadata.

INPUT:
    - checkpoint_path: Path -> output checkpoint path
    - model: TemporalNaiveLayerAttention -> trained temporal model
    - optimizer: torch.optim.Optimizer -> optimizer state
    - epoch: int -> completed epoch index
    - best_val_loss: float -> best validation loss so far
    - cfg: Cfg -> experiment configuration
    - model_kwargs: dict -> temporal constructor arguments excluding imgANN
    - arrays: dict -> split indices and target normalization
    - data: PreparedData -> image and temporal labels

OUTPUT:
    - None
"""
def save_checkpoint(
    checkpoint_path: Path,
    model: TemporalNaiveLayerAttention,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    cfg: Cfg,
    model_kwargs: dict,
    arrays: dict,
    data: PreparedData,
):
    # imgANN/backbone weights are intentionally excluded from model.state_dict().
    checkpoint = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "cfg": asdict(cfg),
        "model_kwargs": model_kwargs,
        "transformer_feature_dim": model.feature_dim,
        "train_indices": arrays["train_indices"],
        "val_indices": arrays["val_indices"],
        "target_mean": arrays["target_mean"],
        "target_std": arrays["target_std"],
        "image_names": data.image_names,
        "layer_names": model.layer_names,
        "time_values_ms": data.time_values_ms,
    }
    torch.save(checkpoint, checkpoint_path)
# EOF


"""
save_validation_attention
Run the best model on validation images and save its time x layer weights.

INPUT:
    - model: nn.Module -> model loaded with its best temporal checkpoint
    - loader: DataLoader -> non-shuffled validation batches
    - device: torch.device -> compute device
    - output_path: Path -> compressed NPZ destination
    - val_indices: np.ndarray -> source image indices
    - data: PreparedData -> image and temporal labels

OUTPUT:
    - None
"""
def save_validation_attention(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_path: Path,
    val_indices: np.ndarray,
    data: PreparedData,
):
    model.eval()
    all_attention = []
    with torch.inference_mode():
        for images, _ in loader:
            _, _, attention = model(images.to(device), return_attention=True)
            all_attention.append(attention.cpu().numpy())
        # end for images
    # end with inference mode
    np.savez_compressed(
        output_path,
        attention=np.concatenate(all_attention, axis=0),
        image_indices=val_indices,
        image_names=np.asarray([data.image_names[idx] for idx in val_indices]),
        layer_names=np.asarray(model.layer_names),
        time_values_ms=data.time_values_ms,
    )
# EOF


class SmokeTestImgANN:
    """Small imgANN-compatible object used to test hooks without loading DINO."""

    def __init__(self, feature_dim: int = 32):
        self.model = nn.Identity()
        self.feature_dim = feature_dim
        self.layer_names = []

    def get_model(self):
        return self.model
    # EOF

    def get_pkg(self) -> str:
        return "hf"
    # EOF

    def create_forward_hook(self, layer_names: list[str]):
        # Record the same ordered names that real imgANN hooks would register.
        self.layer_names = list(layer_names)
        return {}, {}
    # EOF

    def extract_features(self, x: dict) -> dict[str, torch.Tensor]:
        # Create deterministic pooled layer vectors from the fake image batch.
        images = x["pixel_values"]
        flattened_images = images.flatten(start_dim=1)
        base_features = flattened_images[:, : self.feature_dim]
        return {
            layer_name: base_features + layer_idx
            for layer_idx, layer_name in enumerate(self.layer_names)
        }
    # EOF
# EOC


"""
smoke_test
Verify internal feature extraction, lazy projection, output shapes, and independence.

OUTPUT:
    - None
"""
def smoke_test():
    batch_size, n_layers, feature_dim = 3, 6, 32
    n_time_bins, latent_dim, output_dim = 5, 24, 11
    layer_names = [f"layer.{idx}" for idx in range(n_layers)]
    img_ann = SmokeTestImgANN(feature_dim=feature_dim)
    model = TemporalNaiveLayerAttention(
        img_ann=img_ann,
        layer_names=layer_names,
        n_time_bins=n_time_bins,
        position_embedding_dim=16,
        layer_projection_dim=20,
        latent_dim=latent_dim,
        output_dim=output_dim,
    )

    # Raw images are converted to hooked fake transformer features inside forward.
    images = torch.randn(batch_size, 3, 4, 4)
    pred, latent, attention = model(images)
    assert img_ann.layer_names == layer_names
    assert model.feature_dim == feature_dim
    assert pred.shape == (batch_size, n_time_bins, output_dim)
    assert latent.shape == (batch_size, n_time_bins, latent_dim)
    assert attention.shape == (batch_size, n_time_bins, n_layers)
    torch.testing.assert_close(
        attention.sum(dim=-1), torch.ones(batch_size, n_time_bins)
    )
    torch.testing.assert_close(attention[0], attention[1])

    # Reordering time positions must only reorder the independent outputs.
    time_permutation = torch.tensor([3, 0, 4, 1, 2])
    permuted_pred, permuted_latent, permuted_attention = model(
        images, time_idx=time_permutation
    )
    torch.testing.assert_close(permuted_pred, pred[:, time_permutation])
    torch.testing.assert_close(permuted_latent, latent[:, time_permutation])
    torch.testing.assert_close(permuted_attention, attention[:, time_permutation])
    print(
        f"smoke test passed: images={tuple(images.shape)}, feature_dim={model.feature_dim}, "
        f"latent={tuple(latent.shape)}, pred={tuple(pred.shape)}, "
        f"attention={tuple(attention.shape)}"
    )
# EOF


"""
train
Load images and neural targets, build the integrated model, train, and save outputs.

INPUT:
    - cfg: Cfg -> complete experiment configuration

OUTPUT:
    - None
"""
def train(cfg: Cfg):
    set_random_seed(cfg.seed)
    paths = load_project_paths(cfg)
    data_root, stimuli_root, output_dir = resolve_paths(cfg, paths)
    device = choose_device(cfg.device)

    # Build the frozen image backbone and its correctly preprocessed ImageFolder.
    img_ann, image_dataset, layer_names = build_img_ann_and_dataset(
        cfg, paths, stimuli_root, device
    )
    image_names = imagefolder_names(image_dataset)

    # Load neural responses in exactly the same image order as the backbone inputs.
    targets, time_values_ms = load_neural_targets(data_root, cfg, image_names)
    if not np.isfinite(targets).all():
        raise ValueError("Neural targets must contain only finite values.")
    # end if targets contain non-finite values
    data = PreparedData(targets, image_names, time_values_ms)
    arrays = split_and_normalize_targets(targets, cfg)

    # Pair each processed image with its normalized dynamic neural target.
    paired_dataset = NeuralImageDataset(image_dataset, arrays["targets"])
    train_dataset = Subset(paired_dataset, arrays["train_indices"].tolist())
    val_dataset = Subset(paired_dataset, arrays["val_indices"].tolist())
    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    # imgANN is supplied separately because frozen backbone weights are not saved.
    model_kwargs = {
        "layer_names": layer_names,
        "n_time_bins": targets.shape[1],
        "position_embedding_dim": cfg.position_embedding_dim,
        "layer_projection_dim": cfg.layer_projection_dim,
        "latent_dim": cfg.latent_dim,
        "output_dim": targets.shape[2],
    }
    model = TemporalNaiveLayerAttention(img_ann=img_ann, **model_kwargs).to(device)

    # Materialize each LazyLinear input width before constructing the optimizer.
    first_images, _ = next(iter(train_loader))
    with torch.inference_mode():
        model(first_images.to(device), return_attention=False)
    # end with inference mode
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    loss_function = nn.MSELoss()

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}")
    print(
        f"images: {len(image_dataset)}; hooked layers: {len(layer_names)}; "
        f"transformer feature_dim: {model.feature_dim}"
    )
    print(f"neural targets: {targets.shape}")
    print(
        f"split: {len(arrays['train_indices'])} train / "
        f"{len(arrays['val_indices'])} validation images"
    )
    print(f"outputs: {output_dir}")

    best_val_loss = float("inf")
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    log_path = output_dir / "losses.csv"
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    # end with config output

    with open(log_path, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for epoch in range(1, cfg.epochs + 1):
            train_loss = run_epoch(
                model, train_loader, loss_function, device, optimizer
            )
            val_loss = run_epoch(model, val_loader, loss_function, device)
            is_best = val_loss < best_val_loss
            best_val_loss = min(best_val_loss, val_loss)
            writer.writerow([epoch, train_loss, val_loss])
            log_file.flush()
            print(
                f"epoch {epoch:03d}/{cfg.epochs:03d} | "
                f"train {train_loss:.6f} | val {val_loss:.6f}"
            )

            # last.pt supports resuming; best.pt supports final evaluation.
            save_checkpoint(
                last_path,
                model,
                optimizer,
                epoch,
                best_val_loss,
                cfg,
                model_kwargs,
                arrays,
                data,
            )
            if is_best:
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    epoch,
                    best_val_loss,
                    cfg,
                    model_kwargs,
                    arrays,
                    data,
                )
            # end if is best
        # end for epoch
    # end with log file

    if cfg.save_attention:
        # The current model already owns the frozen imgANN and registered hooks.
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        save_validation_attention(
            model,
            val_loader,
            device,
            output_dir / "validation_attention.npz",
            arrays["val_indices"],
            data,
        )
        print(f"saved {output_dir / 'validation_attention.npz'}")
    # end if save attention
    print(f"best validation loss: {best_val_loss:.6f}; checkpoint: {best_path}")
# EOF


if __name__ == "__main__":
    cfg = parse_args()
    if cfg.smoke_test:
        smoke_test()
    else:
        train(cfg)
    # end if smoke test
# EOF
