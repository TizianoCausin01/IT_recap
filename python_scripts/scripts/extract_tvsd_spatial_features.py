"""
Cache the last convolutional feature map of a torchvision backbone for TVSD.

The archives used so far keep one pooled vector per layer, which discards where
in the image a feature was found. This script keeps the map itself -- AlexNet
conv5 is 256 x 13 x 13, ConvNeXt-T stage 4 is 768 x 7 x 7 -- so a readout can
give every IT site its own spatial weighting.

Rows follow the one-based train_idx and test_idx identifiers of
f_THINGS_MUA_trials.mat, exactly like the pooled archives, and each split is
written as its own float16 .npy for memory-mapped reading.

Weights come from torchvision's own enums, so the official preprocessing
transform of each checkpoint is the one applied to the THINGS pixels.
"""

import argparse
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml
from torchvision.models import get_model, get_model_weights


ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python_scripts" / "src"))

from IT_recap.tvsd import (  # noqa: E402
    TVSDOrderedImageDataset,
    extract_tvsd_spatial_features,
    load_tvsd_stimulus_paths,
)
from IT_recap.tvsd_experiments import resolve_device  # noqa: E402


# The last convolutional stage of each supported backbone. AlexNet is taken
# after its conv5 ReLU and before the final max-pool, so the 13 x 13 grid is
# preserved; ConvNeXt is taken at the output of its fourth stage.
DEFAULT_SPATIAL_LAYERS = {
    "alexnet": "features.11",
    "convnext_tiny": "features.7",
    "convnext_small": "features.7",
    "convnext_base": "features.7",
}


@dataclass
class Cfg:
    env: str = ENV
    things_metadata_path: str | None = None
    things_image_root: str | None = None
    output_dir: str | None = None
    output_stem: str | None = None

    model_name: str = "alexnet"
    layer_name: str | None = None

    batch_size: int = 64
    num_workers: int = 4
    progress_interval: int = 25
    device: str = "auto"
    overwrite: bool = False


"""
parse_args
Parse command-line overrides into the spatial extraction configuration.

OUTPUT:
    - cfg: Cfg -> paths, backbone, layer, and loader parameters
"""
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for field_name, field_definition in Cfg.__dataclass_fields__.items():
        default = field_definition.default
        argument_name = f"--{field_name}"
        if isinstance(default, bool):
            parser.add_argument(
                argument_name,
                action=argparse.BooleanOptionalAction,
                default=default,
            )
        elif default is None:
            parser.add_argument(argument_name, default=default)
        else:
            parser.add_argument(
                argument_name, type=type(default), default=default
            )
        # end if boolean, optional string, or typed argument
    # end for configuration field
    return Cfg(**vars(parser.parse_args()))
# EOF


"""
resolve_paths
Resolve the THINGS metadata, the THINGS pixels, and the cache directory.

INPUT:
    - cfg: Cfg -> extraction configuration

OUTPUT:
    - metadata_path: Path -> things_imgs.mat
    - image_root: Path -> directory holding the THINGS category folders
    - output_dir: Path -> destination directory for the caches
"""
def resolve_paths(cfg):
    with open(PROJECT_ROOT / "config.yaml", "r") as config_file:
        environment_paths = yaml.safe_load(config_file)[cfg.env]["paths"]
    # end with project configuration

    data_root = Path(
        environment_paths.get("it_recap_data_path")
        or environment_paths["data_path"]
    ).expanduser()
    metadata_path = Path(
        cfg.things_metadata_path or data_root / "things_imgs.mat"
    ).expanduser()

    # The THINGS pixels are a plain image tree; where it lives differs per
    # machine, so an explicit config entry wins over the lab-drive default.
    configured_root = cfg.things_image_root or environment_paths.get(
        "things_images"
    )
    if configured_root is not None:
        image_root = Path(configured_root).expanduser()
    else:
        image_root = (
            Path(environment_paths["livingstone_lab"]) / "Stimuli" / "THINGS"
        )
    # end if the THINGS root needs the lab-drive fallback
    if not image_root.is_dir():
        raise FileNotFoundError(
            f"THINGS images were not found at {image_root}. Set "
            "paths.things_images in config.yaml or pass --things_image_root."
        )
    # end if the stimulus tree is unavailable

    output_dir = Path(cfg.output_dir or data_root / "models").expanduser()
    return metadata_path, image_root, output_dir
# EOF


def main():
    cfg = parse_args()
    metadata_path, image_root, output_dir = resolve_paths(cfg)
    layer_name = cfg.layer_name or DEFAULT_SPATIAL_LAYERS.get(cfg.model_name)
    if layer_name is None:
        raise KeyError(
            f"No default spatial layer for {cfg.model_name!r}; pass "
            "--layer_name."
        )
    # end if the backbone has no configured layer
    output_stem = cfg.output_stem or (
        f"tvsd_monkeyF_{cfg.model_name}_{layer_name.replace('.', '_')}_spatial"
    )

    device = resolve_device(cfg.device)
    weights = get_model_weights(cfg.model_name).DEFAULT
    model = get_model(cfg.model_name, weights=weights).eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    # end for frozen backbone parameter

    # The checkpoint's own transform decides resize, crop, and normalization.
    transform = weights.transforms()
    image_datasets = {
        split_name: TVSDOrderedImageDataset(
            image_root,
            load_tvsd_stimulus_paths(metadata_path, split_name),
            transform,
        )
        for split_name in ("train", "test")
    }
    print(
        f"{cfg.model_name} · {layer_name} · device {device} | "
        f"{len(image_datasets['train']):,} train and "
        f"{len(image_datasets['test']):,} test stimuli from {image_root}"
    )

    cache_paths, feature_shape = extract_tvsd_spatial_features(
        model,
        layer_name,
        image_datasets,
        output_dir,
        output_stem,
        device,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        progress_interval=cfg.progress_interval,
        overwrite=cfg.overwrite,
    )
    metadata = {
        **asdict(cfg),
        "layer_name": layer_name,
        "output_stem": output_stem,
        "feature_shape": list(feature_shape),
        "cache_paths": {
            split_name: str(path) for split_name, path in cache_paths.items()
        },
    }
    metadata_file = output_dir / f"{output_stem}_metadata.yaml"
    with open(metadata_file, "w") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)
    # end with saved cache metadata
    for split_name, path in cache_paths.items():
        size_gb = path.stat().st_size / 1024 ** 3
        print(f"{split_name}: {path.name} {feature_shape} ({size_gb:.2f} GB)")
    # end for cached split
    print(f"metadata written to {metadata_file}")
# EOF


if __name__ == "__main__":
    main()
# EOC
