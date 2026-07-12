import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "python_scripts" / "src"))

from IT_recap.convrnn_analysis import (  # noqa: E402
    VIT_L_16_TIMM_LAYERS,
    load_paths,
    rdm_vector,
    rsa_similarity,
)


@dataclass
class Cfg:
    folder_name: str = "talia_20each_tizi"
    target_model_name: str = "rgc_intermediate"
    target_layer: str = "conv10"
    target_img_size: int = 224
    target_pooling: str = "mean"
    model_name: str = "vit_l_16"
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
        description="Run static dRSA from ViT layers to a dynamic ConvRNN IT-like target layer."
    )
    parser.add_argument("--folder_name", default=Cfg.folder_name)
    parser.add_argument("--target_model_name", default=Cfg.target_model_name)
    parser.add_argument("--target_layer", default=Cfg.target_layer)
    parser.add_argument("--target_img_size", type=int, default=Cfg.target_img_size)
    parser.add_argument("--target_pooling", default=Cfg.target_pooling)
    parser.add_argument("--model_name", default=Cfg.model_name)
    parser.add_argument("--img_size", type=int, default=Cfg.img_size)
    parser.add_argument("--pooling", default=Cfg.pooling)
    parser.add_argument("--signal_RDM_metric", default=Cfg.signal_RDM_metric)
    parser.add_argument("--model_RDM_metric", default=Cfg.model_RDM_metric)
    parser.add_argument("--RSA_metric", choices=("spearman", "pearson"), default=Cfg.RSA_metric)
    parser.add_argument("--env", default=Cfg.env)
    parser.add_argument("--out_name", default=Cfg.out_name)
    args = parser.parse_args()
    return Cfg(**vars(args))
# EOF


"""
load_static_model_features
Load static model features using the metrics_II save-name convention.

INPUT:
    - paths: dict -> project path config
    - cfg: Cfg -> script configuration

OUTPUT:
    - features_by_layer: dict[str, np.ndarray] -> features x images matrices
"""
def load_static_model_features(paths, cfg):
    model_dir = Path(paths["data_path"]) / "models"
    features_by_layer = {}
    for layer in VIT_L_16_TIMM_LAYERS:
        filename = (
            f"{cfg.folder_name}_{cfg.model_name}_{cfg.img_size}_"
            f"{layer}_features_{cfg.pooling}pool.npz"
        )
        path = model_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        # end if not path.exists():
        features_by_layer[layer] = np.load(path)["arr_0"]
    # end for layer in VIT_L_16_TIMM_LAYERS:
    return features_by_layer
# EOF


if __name__ == "__main__":
    cfg = parse_args()
    paths = load_paths(PROJECT_ROOT, cfg.env)
    model_dir = Path(paths["data_path"]) / "models"
    results_dir = Path(paths["data_path"]) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

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

    target_npz = np.load(target_path)
    target_features = target_npz["arr_0"]
    times = target_npz["times"]
    model_features_by_layer = load_static_model_features(paths, cfg)
    model_rdms = {
        layer: rdm_vector(features, metric=cfg.model_RDM_metric)
        for layer, features in model_features_by_layer.items()
    }

    drsa = np.zeros((len(times), len(VIT_L_16_TIMM_LAYERS)), dtype=np.float32)
    for time_idx, time in enumerate(times):
        signal_rdm = rdm_vector(
            target_features[:, time_idx, :], metric=cfg.signal_RDM_metric
        )
        for layer_idx, layer in enumerate(VIT_L_16_TIMM_LAYERS):
            drsa[time_idx, layer_idx] = rsa_similarity(
                signal_rdm, model_rdms[layer], metric=cfg.RSA_metric
            )
        # end for layer_idx, layer in enumerate(VIT_L_16_TIMM_LAYERS):
        best_idx = int(np.nanargmax(drsa[time_idx]))
        print(
            f"time={int(time):02d} best={VIT_L_16_TIMM_LAYERS[best_idx]} "
            f"dRSA={drsa[time_idx, best_idx]:.4f}"
        )
    # end for time_idx, time in enumerate(times):

    best_layer_idx = np.nanargmax(drsa, axis=1)
    best_layer_name = np.array([VIT_L_16_TIMM_LAYERS[idx] for idx in best_layer_idx])
    out_name = cfg.out_name
    if out_name is None:
        out_name = (
            f"static_dRSA_{cfg.signal_RDM_metric}-{cfg.model_RDM_metric}_"
            f"{cfg.folder_name}_{cfg.target_model_name}_{cfg.target_layer}_"
            f"target_vs_{cfg.model_name}_{cfg.RSA_metric}.npz"
        )
    # end if out_name is None:
    np.savez_compressed(
        results_dir / out_name,
        drsa=drsa,
        rsa=drsa,
        times=times,
        model_layers=np.array(VIT_L_16_TIMM_LAYERS),
        vit_layers=np.array(VIT_L_16_TIMM_LAYERS),
        best_layer_idx=best_layer_idx,
        best_layer_name=best_layer_name,
        cfg=cfg.__dict__,
    )
    print(f"saved {results_dir / out_name}")
# EOF
