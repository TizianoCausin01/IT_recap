import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoImageProcessor


ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_SRC = PROJECT_ROOT / "python_scripts" / "src"
sys.path.insert(0, str(PROJECT_SRC))

with open(PROJECT_ROOT / "config.yaml", "r") as f:
    config = yaml.safe_load(f)

default_paths = config[ENV]["paths"]
sys.path.insert(0, default_paths["useful_stuff_path"])

from IT_recap.hf_feature_extraction import (  # noqa: E402
    HF_MODEL_REPOS,
    HF_MODEL_USE_FAST,
    ProcessorTransform,
    prepare_hf_checkpoint,
)
from IT_recap.tvsd import (  # noqa: E402
    TVSDOrderedImageDataset,
    extract_tvsd_ann_features,
    load_tvsd_stimulus_paths,
)


DEFAULT_LAYERS = (
    "layer.3.mlp.down_proj",
    "layer.13.mlp.down_proj",
    "layer.16.mlp.down_proj",
    "layer.20.mlp.down_proj",
)


@dataclass
class Cfg:
    # Environment and data locations. None resolves through config.yaml.
    env: str = ENV
    data_root: str | None = None
    things_metadata_path: str | None = None
    things_image_root: str | None = None
    output_path: str | None = None

    # Frozen target ANN and selected representation layers.
    model_name: str = "dino_v3_l"
    repo_url: str | None = None
    img_size: int = 224
    pooling: str = "mean"
    layer_names: str = ",".join(DEFAULT_LAYERS)
    use_fast_processor: bool | None = None
    trust_remote_code: bool = True
    attn_implementation: str | None = "sdpa"

    # Data-loading and execution parameters.
    batch_size: int = 64
    num_workers: int = 0
    progress_interval: int = 50
    overwrite: bool = False
    prepare_only: bool = False


"""
parse_args
Parse command-line overrides into the TVSD ANN extraction configuration.

OUTPUT:
    - cfg: Cfg -> paths, ANN definition, layers, and loader parameters
"""
def parse_args() -> Cfg:
    parser = argparse.ArgumentParser(
        description=(
            "Extract ordered ANN features for the TVSD/THINGS stimuli. Output "
            "rows follow the one-based train_idx and test_idx identifiers in "
            "f_THINGS_MUA_trials.mat."
        )
    )
    parser.add_argument("--env", default=Cfg.env, choices=config)
    parser.add_argument("--data_root", default=Cfg.data_root)
    parser.add_argument("--things_metadata_path", default=Cfg.things_metadata_path)
    parser.add_argument("--things_image_root", default=Cfg.things_image_root)
    parser.add_argument("--output_path", default=Cfg.output_path)
    parser.add_argument(
        "--model_name",
        default=Cfg.model_name,
        choices=HF_MODEL_REPOS,
    )
    parser.add_argument("--repo_url", default=Cfg.repo_url)
    parser.add_argument("--img_size", type=int, default=Cfg.img_size)
    parser.add_argument("--pooling", default=Cfg.pooling)
    parser.add_argument(
        "--layer_names",
        default=Cfg.layer_names,
        help="Comma-separated hooked module names in model order.",
    )
    parser.add_argument(
        "--use_fast_processor",
        action=argparse.BooleanOptionalAction,
        default=Cfg.use_fast_processor,
    )
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=Cfg.trust_remote_code,
    )
    parser.add_argument(
        "--attn_implementation",
        default=Cfg.attn_implementation,
        help="Attention backend passed to imgANN; use 'none' to leave unset.",
    )
    parser.add_argument("--batch_size", type=int, default=Cfg.batch_size)
    parser.add_argument("--num_workers", type=int, default=Cfg.num_workers)
    parser.add_argument(
        "--progress_interval",
        type=int,
        default=Cfg.progress_interval,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prepare_only", action="store_true")
    return Cfg(**vars(parser.parse_args()))
# EOF


"""
resolve_paths
Resolve dataset metadata, THINGS pixels, and output through config.yaml unless
the corresponding command-line path was supplied explicitly.

INPUT:
    - cfg: Cfg -> extraction configuration

OUTPUT:
    - metadata_path: Path -> official things_imgs.mat path
    - image_root: Path -> root containing THINGS category directories
    - output_path: Path -> destination TVSD feature archive
"""
def resolve_paths(cfg: Cfg) -> tuple[Path, Path, Path]:
    environment_paths = config[cfg.env]["paths"]
    configured_data_root = (
        cfg.data_root
        or environment_paths.get("it_recap_data_path")
        or environment_paths.get("data_path")
    )
    if configured_data_root is None:
        raise KeyError(
            f"Environment {cfg.env!r} has no data_path. Supply --data_root."
        )
    # end if the environment does not define project-local data

    data_root = Path(configured_data_root).expanduser()
    metadata_path = Path(
        cfg.things_metadata_path or data_root / "things_imgs.mat"
    ).expanduser()

    if cfg.things_image_root is not None:
        image_root = Path(cfg.things_image_root).expanduser()
    elif "livingstone_lab" in environment_paths:
        image_root = (
            Path(environment_paths["livingstone_lab"]) / "Stimuli" / "THINGS"
        )
    else:
        raise KeyError(
            f"Environment {cfg.env!r} has no livingstone_lab path. Supply "
            "--things_image_root."
        )
    # end if the THINGS root needs an explicit override

    default_output_name = (
        f"tvsd_monkeyF_{cfg.model_name}_{cfg.img_size}_features.npz"
    )
    output_path = Path(
        cfg.output_path or data_root / "models" / default_output_name
    ).expanduser()
    return metadata_path, image_root, output_path
# EOF


"""
validate_ordered_images
Check that all paths referenced by things_imgs.mat exist below the THINGS root.

INPUT:
    - image_root: Path -> THINGS dataset root
    - split_paths: dict[str, list[str]] -> ordered train and test paths

OUTPUT:
    - None
"""
def validate_ordered_images(
    image_root: Path,
    split_paths: dict[str, list[str]],
) -> None:
    missing_examples = []
    missing_count = 0
    for split_name, relative_paths in split_paths.items():
        for relative_path in relative_paths:
            image_path = image_root / relative_path
            if not image_path.is_file():
                missing_count += 1
                if len(missing_examples) < 5:
                    missing_examples.append(f"{split_name}: {image_path}")
                # end if another example should be retained
            # end if one ordered image is absent
        # end for ordered image path
    # end for TVSD split

    if missing_count:
        examples = "\n  ".join(missing_examples)
        raise FileNotFoundError(
            f"{missing_count:,} ordered THINGS images are missing. Examples:\n"
            f"  {examples}\nSupply the dataset root with --things_image_root."
        )
    # end if the image dataset is incomplete
# EOF


"""
main
Load the target ANN, extract ordered TVSD stimulus features, and save one
archive that the BaselineModel marimo notebook can consume directly.

INPUT:
    - cfg: Cfg -> extraction configuration

OUTPUT:
    - None
"""
def main(cfg: Cfg) -> None:
    if cfg.img_size <= 0:
        raise ValueError("img_size must be positive.")
    # end if the image resolution is invalid

    layer_names = [
        layer_name.strip()
        for layer_name in cfg.layer_names.split(",")
        if layer_name.strip()
    ]
    if not layer_names:
        raise ValueError("layer_names must contain at least one module name.")
    # end if no target ANN layers were selected

    repo_url = cfg.repo_url or HF_MODEL_REPOS[cfg.model_name]
    use_fast_processor = (
        HF_MODEL_USE_FAST[cfg.model_name]
        if cfg.use_fast_processor is None
        else cfg.use_fast_processor
    )
    if cfg.prepare_only:
        prepare_hf_checkpoint(repo_url, use_fast_processor)
        print(f"Prepared model and image processor: {repo_url}")
        return
    # end if only the Hugging Face checkpoint was requested

    metadata_path, image_root, output_path = resolve_paths(cfg)
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"things_imgs.mat was not found at {metadata_path}. Supply "
            "--things_metadata_path."
        )
    # end if the official stimulus ordering is unavailable
    if not image_root.is_dir():
        raise FileNotFoundError(
            f"The THINGS image root was not found at {image_root}. Supply "
            "--things_image_root."
        )
    # end if THINGS pixels are unavailable
    if output_path.exists() and not cfg.overwrite:
        raise FileExistsError(
            f"Feature archive already exists at {output_path}. Pass --overwrite "
            "to replace it."
        )
    # end if replacing features was not authorized

    # things_imgs.mat defines the exact row-to-stimulus mapping used by ALLMAT.
    split_paths = {
        split_name: load_tvsd_stimulus_paths(metadata_path, split_name)
        for split_name in ("train", "test")
    }
    expected_counts = {"train": 22_248, "test": 100}
    actual_counts = {
        split_name: len(relative_paths)
        for split_name, relative_paths in split_paths.items()
    }
    if actual_counts != expected_counts:
        raise ValueError(
            f"Expected TVSD stimulus counts {expected_counts}, got {actual_counts}."
        )
    # end if the metadata does not describe the published monkey F dataset
    validate_ordered_images(image_root, split_paths)

    # Apply the checkpoint-author's normalization and resize/crop procedure.
    processor = AutoImageProcessor.from_pretrained(
        repo_url,
        use_fast=use_fast_processor,
    )
    transform = ProcessorTransform(processor)
    train_dataset = TVSDOrderedImageDataset(
        image_root,
        split_paths["train"],
        transform,
    )
    test_dataset = TVSDOrderedImageDataset(
        image_root,
        split_paths["test"],
        transform,
    )

    # imgANN pools every hooked transformer representation to one feature vector.
    from useful_stuff.image_processing.computational_models import imgANN

    attention_backend = cfg.attn_implementation
    if attention_backend is not None and attention_backend.lower() == "none":
        attention_backend = None
    # end if the attention backend should be selected automatically
    ann = imgANN(
        model_name=cfg.model_name,
        pkg="hf",
        img_size=cfg.img_size,
        pooling=cfg.pooling,
        dtype=torch.float32,
        attn_implementation=attention_backend,
        repo_url=repo_url,
        trust_remote_code=cfg.trust_remote_code,
    )
    print(f"Extracting {len(layer_names)} layers with {ann}")
    train_features, test_features = extract_tvsd_ann_features(
        ann=ann,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        layer_names=layer_names,
        output_path=output_path,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        progress_interval=cfg.progress_interval,
    )

    print(f"Saved feature archive: {output_path}")
    print(f"  train_features: {train_features.shape}")
    print(f"  test_features:  {test_features.shape}")
    print(f"  layer_names:    {np.asarray(layer_names).tolist()}")
    print("Rows map to train_idx - 1 and test_idx - 1, respectively.")
# EOF


if __name__ == "__main__":
    main(parse_args())
# EOF
