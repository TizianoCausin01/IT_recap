import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "python_scripts" / "src"))

from IT_recap.convrnn_analysis import load_paths
from IT_recap.static_drsa import (
    compute_static_drsa,
    load_static_model_features,
    save_static_drsa_result,
    static_drsa_result_name,
)


@dataclass
class Cfg:
    folder_name: str = "talia_20each_tizi"
    target_model_name: str = "rgc_intermediate"
    target_layer: str = "conv10"
    target_img_size: int = 224
    target_pooling: str = "mean"
    model_name: str = "vit_l_16"
    pkg: str = "timm"
    img_size: int = 384
    pooling: str = "mean"
    signal_RDM_metric: str = "correlation"
    model_RDM_metric: str = "correlation"
    RSA_metric: str = "spearman"
    env: Optional[str] = None
    out_name: Optional[str] = None


"""
parse_args
Parse command-line arguments into a Cfg object.

OUTPUT:
    - cfg: Cfg -> script configuration
"""
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare registered static model layers with a dynamic ConvRNN target."
    )
    parser.add_argument("--folder_name", default=Cfg.folder_name)
    parser.add_argument("--target_model_name", default=Cfg.target_model_name)
    parser.add_argument("--target_layer", default=Cfg.target_layer)
    parser.add_argument("--target_img_size", type=int, default=Cfg.target_img_size)
    parser.add_argument("--target_pooling", default=Cfg.target_pooling)
    parser.add_argument("--model_name", default=Cfg.model_name)
    parser.add_argument("--pkg", default=Cfg.pkg)
    parser.add_argument("--img_size", type=int, default=Cfg.img_size)
    parser.add_argument("--pooling", default=Cfg.pooling)
    parser.add_argument("--signal_RDM_metric", default=Cfg.signal_RDM_metric)
    parser.add_argument("--model_RDM_metric", default=Cfg.model_RDM_metric)
    parser.add_argument("--RSA_metric", choices=("spearman", "pearson"), default=Cfg.RSA_metric)
    parser.add_argument("--env", default=Cfg.env)
    parser.add_argument("--out_name", default=Cfg.out_name)
    return Cfg(**vars(parser.parse_args()))
# EOF


"""
main
Load registered static layers, compute dRSA against ConvRNN time, and save the result.

INPUT:
    - cfg: Cfg -> script configuration

OUTPUT:
    - None
"""
def main(cfg):
    paths = load_paths(PROJECT_ROOT, cfg.env)
    sys.path.append(paths["useful_stuff_path"])
    from useful_stuff.image_processing.computational_models import (
        get_relevant_output_layers,
    )

    model_dir = Path(paths["data_path"]) / "models"
    results_dir = Path(paths["data_path"]) / "results"
    target_path = model_dir / (
        f"{cfg.folder_name}_{cfg.target_model_name}_{cfg.target_img_size}_"
        f"{cfg.target_layer}_features_timeseries_{cfg.target_pooling}pool.npz"
    )
    if not target_path.exists():
        raise FileNotFoundError(
            f"Missing ConvRNN target features: {target_path}. "
            "Run extract_convrnn_features.py first."
        )
    # end if not target_path.exists():

    with np.load(target_path) as target_file:
        target_features = target_file["arr_0"]
        times = target_file["times"]
    # end with np.load(target_path) as target_file:

    layer_names = get_relevant_output_layers(cfg.model_name, pkg=cfg.pkg)
    features_by_layer = load_static_model_features(
        model_dir,
        cfg.folder_name,
        cfg.model_name,
        cfg.img_size,
        layer_names,
        cfg.pooling,
    )
    drsa, best_layer_idx, best_layer_name = compute_static_drsa(
        target_features,
        times,
        features_by_layer,
        cfg.signal_RDM_metric,
        cfg.model_RDM_metric,
        cfg.RSA_metric,
    )

    out_name = cfg.out_name or static_drsa_result_name(
        cfg.signal_RDM_metric,
        cfg.model_RDM_metric,
        cfg.folder_name,
        cfg.target_model_name,
        cfg.target_layer,
        cfg.model_name,
        cfg.RSA_metric,
    )
    result_path = results_dir / out_name
    save_static_drsa_result(
        result_path,
        drsa,
        times,
        layer_names,
        best_layer_idx,
        best_layer_name,
        asdict(cfg),
    )
    print(f"saved {result_path}")
# EOF


if __name__ == "__main__":
    main(parse_args())
# EOF
