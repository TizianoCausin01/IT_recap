import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml


ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "config.yaml", "r") as f:
    config = yaml.safe_load(f)

paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
sys.path.append(paths["useful_stuff_path"])

from IT_recap.hf_feature_extraction import (
    HF_MODEL_REPOS,
    HF_MODEL_USE_FAST,
    extract_hf_layer_group,
    load_hf_worker_inputs,
    prepare_hf_checkpoint,
    split_layers,
)
from useful_stuff.general_utils import get_device, print_wise
from useful_stuff.image_processing.computational_models import get_relevant_output_layers
from useful_stuff.parallel.parallel_funcs import master_workers_queue, parallel_setup


@dataclass
class Cfg:
    model_name: str = "dino_v3_l"
    folder_name: str = "talia_20each_tizi"
    img_size: int = 224
    batch_size: int = 8
    pooling: str = "mean"
    num_workers: int = 0
    repo_url: str | None = None
    prepare_only: bool = False


"""
parse_args
Parse command-line overrides into the extraction configuration.

OUTPUT:
    - cfg: Cfg -> extraction parameters
"""
def parse_args() -> Cfg:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=Cfg.model_name, choices=HF_MODEL_REPOS)
    parser.add_argument("--folder_name", default=Cfg.folder_name)
    parser.add_argument("--img_size", type=int, default=Cfg.img_size)
    parser.add_argument("--batch_size", type=int, default=Cfg.batch_size)
    parser.add_argument("--pooling", default=Cfg.pooling)
    parser.add_argument("--num_workers", type=int, default=Cfg.num_workers)
    parser.add_argument("--repo_url", default=Cfg.repo_url)
    parser.add_argument("--prepare_only", action="store_true")
    return Cfg(**vars(parser.parse_args()))
# EOF


"""
main
Prepare a checkpoint or distribute grouped layer extraction across MPI workers.

INPUT:
    - cfg: Cfg -> extraction parameters

OUTPUT:
    - None
"""
def main(cfg: Cfg):
    repo_url = cfg.repo_url or HF_MODEL_REPOS[cfg.model_name]
    use_fast = HF_MODEL_USE_FAST[cfg.model_name]
    if cfg.prepare_only:
        prepare_hf_checkpoint(repo_url, use_fast)
        print(f"prepared {repo_url}")
        return
    # end if cfg.prepare_only:

    _, rank, size = parallel_setup()
    if size < 2:
        raise RuntimeError("Run extraction with at least two MPI processes.")
    # end if size < 2:

    layer_names = get_relevant_output_layers(cfg.model_name, pkg="hf")
    layer_groups = split_layers(layer_names, size - 1)
    if rank == 0:
        ann, loader = None, None
    else:
        n_threads = max(1, (os.cpu_count() or 1) // (size - 1))
        torch.set_num_threads(n_threads)
        device = get_device()
        ann, loader = load_hf_worker_inputs(
            paths,
            cfg.model_name,
            repo_url,
            cfg.folder_name,
            cfg.img_size,
            cfg.batch_size,
            cfg.pooling,
            cfg.num_workers,
            device,
            use_fast,
        )
        print_wise(
            f"loaded {cfg.model_name} and {len(loader.dataset)} images on {device}",
            rank=rank,
        )
    # end if rank == 0:

    master_workers_queue(
        layer_groups,
        paths,
        extract_hf_layer_group,
        ann,
        loader,
        cfg.folder_name,
        cfg.model_name,
        cfg.img_size,
        cfg.pooling,
    )
# EOF


if __name__ == "__main__":
    main(parse_args())
# EOF
