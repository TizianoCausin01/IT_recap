from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from transformers import AutoImageProcessor

from useful_stuff.general_utils import print_wise
from useful_stuff.image_processing.computational_models import imgANN


HF_MODEL_REPOS = {
    "dino_v3_l": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "ijepa_vith14_1k": "facebook/ijepa_vith14_1k",
}
HF_MODEL_USE_FAST = {
    "dino_v3_l": True,
    "ijepa_vith14_1k": False,
}


"""
ProcessorTransform
Apply a saved Hugging Face image processor inside an ImageFolder dataset.

INPUT:
    - processor: AutoImageProcessor -> checkpoint-specific image processor

OUTPUT:
    - transform: ProcessorTransform -> callable returning one pixel-value tensor
"""
class ProcessorTransform:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, image):
        return self.processor(images=image, return_tensors="pt")["pixel_values"][0]
    # EOF
# EOC


"""
is_valid_image_file
Exclude finder metadata files from ImageFolder indexing.

INPUT:
    - path: str -> candidate image path

OUTPUT:
    - is_valid: bool -> whether ImageFolder should index the file
"""
def is_valid_image_file(path: str) -> bool:
    return not path.endswith("Thumbs.db")
# EOF


"""
prepare_hf_checkpoint
Download a Hugging Face checkpoint and its author-provided image processor.

INPUT:
    - repo_url: str -> Hugging Face checkpoint identifier
    - use_fast: bool -> whether to load the checkpoint's fast processor

OUTPUT:
    - None
"""
def prepare_hf_checkpoint(repo_url: str, use_fast: bool):
    snapshot_download(repo_id=repo_url)
    AutoImageProcessor.from_pretrained(repo_url, use_fast=use_fast)
# EOF


"""
feature_save_path
Build the established ViT activation filename for one model layer.

INPUT:
    - output_dir: Path -> model activation directory
    - dataset_name: str -> stimulus dataset name
    - model_name: str -> saved model alias
    - img_size: int -> processor output resolution
    - layer_name: str -> hooked module path
    - pooling: str -> feature pooling method

OUTPUT:
    - save_path: Path -> output NPZ path
"""
def feature_save_path(
    output_dir: Path,
    dataset_name: str,
    model_name: str,
    img_size: int,
    layer_name: str,
    pooling: str,
) -> Path:
    file_name = (
        f"{dataset_name}_{model_name}_{img_size}_{layer_name}"
        f"_features_{pooling}pool.npz"
    )
    return output_dir / file_name
# EOF


"""
split_layers
Split model layers into at most one extraction group per MPI worker.

INPUT:
    - layer_names: list[str] -> ordered model layer names
    - n_workers: int -> number of available worker ranks

OUTPUT:
    - layer_groups: list[list[str]] -> non-empty contiguous layer groups
"""
def split_layers(layer_names: list[str], n_workers: int) -> list[list[str]]:
    groups = np.array_split(np.asarray(layer_names, dtype=object), n_workers)
    return [group.tolist() for group in groups if len(group) > 0]
# EOF


"""
extract_hf_layer_group
Extract and save pooled activations for a group of layers in one dataset pass.

INPUT:
    - paths: dict -> environment-specific project paths
    - rank: int -> MPI worker rank
    - layer_names: list[str] -> layers assigned to this worker
    - ann: imgANN -> loaded model wrapper
    - loader: DataLoader -> model-preprocessed image batches
    - dataset_name: str -> stimulus dataset name
    - model_name: str -> saved model alias
    - img_size: int -> processor output resolution
    - pooling: str -> feature pooling method

OUTPUT:
    - None -> one compressed NPZ file is saved per missing layer
"""
def extract_hf_layer_group(
    paths,
    rank,
    layer_names,
    ann,
    loader,
    dataset_name,
    model_name,
    img_size,
    pooling,
):
    output_dir = Path(paths["data_path"]) / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    missing_layers = [
        layer_name
        for layer_name in layer_names
        if not feature_save_path(
            output_dir,
            dataset_name,
            model_name,
            img_size,
            layer_name,
            pooling,
        ).exists()
    ]
    if not missing_layers:
        print_wise(f"all assigned {model_name} layers already exist", rank=rank)
        return
    # end if not missing_layers:

    ann.create_forward_hook(missing_layers)
    layer_features = {layer_name: [] for layer_name in missing_layers}

    with torch.inference_mode():
        for batch_idx, (pixel_values, _) in enumerate(loader):
            ann.extract_features({"pixel_values": pixel_values.to(ann.device)})
            for layer_name in missing_layers:
                features = ann.features[layer_name].detach().cpu().numpy()
                layer_features[layer_name].append(features)
            # end for layer_name in missing_layers:
            print_wise(f"processed batch {batch_idx + 1}/{len(loader)}", rank=rank)
        # end for batch_idx, (pixel_values, _) in enumerate(loader):
    # end with torch.inference_mode():

    for layer_name in missing_layers:
        features = np.concatenate(layer_features[layer_name], axis=0).T
        save_path = feature_save_path(
            output_dir,
            dataset_name,
            model_name,
            img_size,
            layer_name,
            pooling,
        )
        np.savez_compressed(save_path, features)
        print_wise(f"saved {features.shape} features at {save_path}", rank=rank)
    # end for layer_name in missing_layers:
    ann.clear_hooks()
# EOF


"""
load_hf_worker_inputs
Load imgANN, the official processor, and the ordered ImageFolder dataset.

INPUT:
    - paths: dict -> environment-specific project paths
    - model_name: str -> model alias registered in useful_stuff
    - repo_url: str -> Hugging Face checkpoint identifier
    - folder_name: str -> ImageFolder name under the configured Stimuli path
    - img_size: int -> expected processor output resolution
    - batch_size: int -> image batch size
    - pooling: str -> activation pooling method
    - num_workers: int -> DataLoader worker count
    - device: torch.device -> worker inference device
    - use_fast: bool -> whether to load the checkpoint's fast processor

OUTPUT:
    - ann: imgANN -> model wrapper on the worker device
    - loader: DataLoader -> ordered, model-preprocessed image batches
"""
def load_hf_worker_inputs(
    paths,
    model_name,
    repo_url,
    folder_name,
    img_size,
    batch_size,
    pooling,
    num_workers,
    device,
    use_fast,
):
    ann = imgANN(
        model_name=model_name,
        pkg="hf",
        img_size=img_size,
        pooling=pooling,
        dtype=torch.float32,
        repo_url=repo_url,
        device=device,
    )
    processor = AutoImageProcessor.from_pretrained(repo_url, use_fast=use_fast)
    dataset_path = Path(paths["livingstone_lab"]) / "Stimuli" / folder_name
    dataset = ImageFolder(
        root=dataset_path,
        transform=ProcessorTransform(processor),
        is_valid_file=is_valid_image_file,
        allow_empty=True,
    )

    sample_shape = dataset[0][0].shape[-2:]
    if sample_shape != (img_size, img_size):
        raise ValueError(
            f"{repo_url} processor returned {sample_shape}, expected "
            f"{(img_size, img_size)}"
        )
    # end if sample_shape != (img_size, img_size):

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return ann, loader
# EOF
