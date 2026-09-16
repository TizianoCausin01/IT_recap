"""
Cache the mean-pooled backbone features the three0 decoder regresses from.

Two sources are merged into one archive, both stored in the monkey's
presentation order so they line up row by row with the natraster:

    - the per-layer HuggingFace archives already under data_path/models
      (dino_v3_l, ijepa_vith14_1k, vit_l_16), written by the existing feature
      extraction scripts and stored as [embedding, images] in ANN order;
    - torchvision backbones with locally cached ImageNet weights, hooked at a
      few layers and mean-pooled over space or tokens.

The point of the zoo is diversity: averaging their kernels is what buys the
last part of the gain over the published DINOv3-only ridge. Extraction runs
once and takes a couple of minutes for 776 images.
"""

import argparse
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision.models as tv_models
import torchvision.transforms as tv_transforms
import yaml
from PIL import Image


ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_SRC = PROJECT_ROOT / "python_scripts" / "src"
sys.path.insert(0, str(PROJECT_SRC))

with open(PROJECT_ROOT / "config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)
# end with project configuration

from IT_recap.three0_decoding import THREE0_STIMULI, load_three0_targets  # noqa: E402


# Cached per-layer archives: name -> (file stem template, layer keys).
HF_ARCHIVES = {
    "dino_v3_l": (
        "{dataset}_dino_v3_l_224_layer.{layer}.mlp.down_proj_features_meanpool.npz",
        [str(layer) for layer in range(24)],
    ),
    "ijepa_vith14_1k": (
        "{dataset}_ijepa_vith14_1k_224_encoder.layer.{layer}.output.dense_"
        "features_meanpool.npz",
        [str(layer) for layer in range(32)],
    ),
    "vit_l_16": (
        "{dataset}_vit_l_16_384_blocks.{layer}.mlp.fc2_features_meanpool.npz",
        [str(layer) for layer in range(24)],
    ),
}

# torchvision backbones: name -> (constructor, weights enum, hooked layers).
TORCHVISION_BACKBONES = {
    "alexnet": (
        tv_models.alexnet,
        tv_models.AlexNet_Weights.IMAGENET1K_V1,
        ["features.2", "features.5", "features.7", "features.9", "features.12",
         "classifier.2", "classifier.5"],
    ),
    "vgg16": (
        tv_models.vgg16,
        tv_models.VGG16_Weights.IMAGENET1K_V1,
        ["features.8", "features.15", "features.22", "features.29",
         "classifier.1", "classifier.4"],
    ),
    "resnet50": (
        tv_models.resnet50,
        tv_models.ResNet50_Weights.IMAGENET1K_V1,
        ["layer1", "layer2", "layer3", "layer4"],
    ),
    "resnext50": (
        tv_models.resnext50_32x4d,
        tv_models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1,
        ["layer2", "layer3", "layer4"],
    ),
    "convnext_tiny": (
        tv_models.convnext_tiny,
        tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
        ["features.1", "features.3", "features.5", "features.7"],
    ),
    "densenet201": (
        tv_models.densenet201,
        tv_models.DenseNet201_Weights.IMAGENET1K_V1,
        ["features.denseblock2", "features.denseblock3", "features.denseblock4"],
    ),
    "vit_b_16": (
        tv_models.vit_b_16,
        tv_models.ViT_B_16_Weights.IMAGENET1K_V1,
        ["encoder.layers.encoder_layer_5", "encoder.layers.encoder_layer_8",
         "encoder.layers.encoder_layer_10", "encoder.layers.encoder_layer_11"],
    ),
}


@dataclass
class Cfg:
    env: str | None = None
    stimuli_folder: str = THREE0_STIMULI
    brain_area: str = "AIT"
    img_size: int = 224
    batch_size: int = 32
    device: str = "auto"
    output_name: str = "three0_backbone_features.npz"
    overwrite: bool = False


"""
parse_args
Parse command-line overrides into the feature-extraction configuration.

OUTPUT:
    - cfg: Cfg -> environment, stimulus set, and extraction settings
"""
def parse_args():
    parser = argparse.ArgumentParser(
        description="Cache mean-pooled backbone features for the three0 decoder."
    )
    for field_name, field in Cfg.__dataclass_fields__.items():
        default = field.default
        if isinstance(default, bool):
            parser.add_argument(f"--{field_name}", action="store_true")
        elif default is None:
            parser.add_argument(f"--{field_name}", default=default)
        else:
            parser.add_argument(f"--{field_name}", type=type(default), default=default)
        # end if boolean, optional, or typed argument
    # end for configuration field
    return Cfg(**vars(parser.parse_args()))
# EOF


"""
resolve_device
Resolve an explicit device or select CUDA, then MPS, then CPU.

INPUT:
    - device_name: str -> explicit device or "auto"

OUTPUT:
    - device: torch.device -> compute device
"""
def resolve_device(device_name):
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
# EOF


"""
load_cached_hf_block
Read one per-layer HuggingFace archive family and stack it in monkey order.

INPUT:
    - models_dir: Path -> data_path/models
    - dataset_name: str -> stimulus folder name used in the file stems
    - template: str -> file-name template with {dataset} and {layer}
    - layers: list[str] -> layer identifiers to concatenate
    - ann_index: np.ndarray -> ANN index of every monkey presentation row

OUTPUT:
    - block: np.ndarray | None -> [images, layers * embedding], None if absent
"""
def load_cached_hf_block(models_dir, dataset_name, template, layers, ann_index):
    columns = []
    for layer in layers:
        archive = models_dir / template.format(dataset=dataset_name, layer=layer)
        if not archive.is_file():
            return None
        # end if a layer of this family is missing
        with np.load(archive) as cached:
            # Stored as [embedding, images] in ANN order.
            columns.append(cached["arr_0"].T[ann_index].astype(np.float32))
        # end with cached layer archive
    # end for layer
    return np.concatenate(columns, axis=1)
# EOF


"""
extract_torchvision_block
Mean-pool the hooked activations of one torchvision backbone.

INPUT:
    - constructor: callable -> torchvision model constructor
    - weights: enum -> pretrained weights to load
    - layers: list[str] -> module names to hook
    - images: torch.Tensor -> [images, 3, size, size] preprocessed stimuli
    - device: torch.device -> compute device
    - batch_size: int -> images per forward pass

OUTPUT:
    - block: np.ndarray -> [images, sum of layer widths] pooled features
"""
def extract_torchvision_block(constructor, weights, layers, images, device, batch_size):
    model = constructor(weights=weights).eval().to(device)
    modules = dict(model.named_modules())
    activations = {}
    handles = [
        modules[layer].register_forward_hook(
            lambda module, inputs, output, layer=layer: activations.__setitem__(
                layer, output.detach()
            )
        )
        for layer in layers
    ]
    pooled = {layer: [] for layer in layers}
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            model(images[start : start + batch_size].to(device))
            for layer in layers:
                activation = activations[layer]
                # Conv maps pool over space, token maps over the sequence.
                if activation.dim() == 4:
                    activation = activation.mean((2, 3))
                elif activation.dim() == 3:
                    activation = activation.mean(1)
                # end if the activation is spatial or token shaped
                pooled[layer].append(activation.float().cpu().numpy())
            # end for hooked layer
        # end for image batch
    # end with inference mode
    for handle in handles:
        handle.remove()
    # end for hook handle
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    # end if the MPS cache should be released
    return np.concatenate(
        [np.concatenate(pooled[layer]) for layer in layers], axis=1
    )
# EOF


def main():
    cfg = parse_args()
    env = cfg.env or ENV
    paths = dict(config[env]["paths"])
    sys.path.append(paths["useful_stuff_path"])
    device = resolve_device(cfg.device)

    models_dir = Path(paths["data_path"]) / "models"
    output_path = models_dir / cfg.output_name
    if output_path.is_file() and not cfg.overwrite:
        raise SystemExit(f"{output_path} already exists; pass --overwrite.")
    # end if the archive is already cached

    _, ann_index, image_paths = load_three0_targets(
        paths, brain_area=cfg.brain_area, stimuli_folder=cfg.stimuli_folder
    )
    print(f"{len(image_paths)} stimuli in monkey presentation order")

    blocks = {}
    for name, (template, layers) in HF_ARCHIVES.items():
        block = load_cached_hf_block(
            models_dir, cfg.stimuli_folder, template, layers, ann_index
        )
        if block is None:
            print(f"  {name}: cached layers missing, skipped")
            continue
        # end if this family was never extracted
        blocks[name] = block
        print(f"  {name}: {block.shape}")
    # end for cached HuggingFace family

    transform = tv_transforms.Compose(
        [
            tv_transforms.Resize(int(cfg.img_size * 256 / 224)),
            tv_transforms.CenterCrop(cfg.img_size),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )
    images = torch.stack(
        [transform(Image.open(path).convert("RGB")) for path in image_paths]
    )
    for name, (constructor, weights, layers) in TORCHVISION_BACKBONES.items():
        blocks[name] = extract_torchvision_block(
            constructor, weights, layers, images, device, cfg.batch_size
        )
        print(f"  {name}: {blocks[name].shape}")
    # end for torchvision backbone

    np.savez_compressed(
        output_path, ann_index=ann_index, image_paths=np.array(image_paths), **blocks
    )
    with open(models_dir / (output_path.stem + "_config.yaml"), "w") as meta_file:
        yaml.safe_dump(asdict(cfg) | {"blocks": sorted(blocks)}, meta_file)
    # end with extraction metadata
    print(f"\nwrote {output_path}")
# EOF


if __name__ == "__main__":
    main()
