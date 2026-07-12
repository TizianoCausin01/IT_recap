import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "python_scripts" / "src"))

from IT_recap.convrnn_analysis import (  # noqa: E402
    list_imagefolder_files,
    load_paths,
    rankdata_average,
    rdm_vector,
)
from project_specific_utils.dataloader import (  # noqa: E402
    load_img_natraster,
    map_image_order_from_ann_to_monkey,
)


@dataclass
class Cfg:
    monkey_name: str = "three0"
    date: str = "250313"
    brain_area: str = "AIT"
    folder_name: str = "talia_20each_tizi"
    target_model_name: str = "rgc_intermediate"
    target_layer: str = "conv10"
    target_img_size: int = 224
    target_pooling: str = "mean"
    new_fs: int = 100
    neural_time_start_ms: float = 0.0
    model_timestep_ms: float = 10.0
    signal_RDM_metric: str = "cosine_cnt"
    model_RDM_metric: str = "cosine_cnt"
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
        description="Run dynamic dRSA between real neural rasters and ConvRNN activations."
    )
    parser.add_argument("--monkey_name", default=Cfg.monkey_name)
    parser.add_argument("--date", default=Cfg.date)
    parser.add_argument("--brain_area", default=Cfg.brain_area)
    parser.add_argument("--folder_name", default=Cfg.folder_name)
    parser.add_argument("--target_model_name", default=Cfg.target_model_name)
    parser.add_argument("--target_layer", default=Cfg.target_layer)
    parser.add_argument("--target_img_size", type=int, default=Cfg.target_img_size)
    parser.add_argument("--target_pooling", default=Cfg.target_pooling)
    parser.add_argument("--new_fs", type=int, default=Cfg.new_fs)
    parser.add_argument("--neural_time_start_ms", type=float, default=Cfg.neural_time_start_ms)
    parser.add_argument("--model_timestep_ms", type=float, default=Cfg.model_timestep_ms)
    parser.add_argument("--signal_RDM_metric", default=Cfg.signal_RDM_metric)
    parser.add_argument("--model_RDM_metric", default=Cfg.model_RDM_metric)
    parser.add_argument("--RSA_metric", choices=("spearman", "pearson"), default=Cfg.RSA_metric)
    parser.add_argument("--env", default=Cfg.env)
    parser.add_argument("--out_name", default=Cfg.out_name)
    args = parser.parse_args()
    return Cfg(**vars(args))
# EOF


class SimpleImageFolder:
    """
    Minimal ImageFolder-like object needed by map_image_order_from_ann_to_monkey.
    """

    def __init__(self, root):
        self.root = str(root)
        self.samples = [(str(path), 0) for path in list_imagefolder_files(root)]
    # EOF
# EOC


"""
load_convrnn_timeseries
Load ConvRNN activations saved by extract_convrnn_features.py.

INPUT:
    - paths: dict -> project path config
    - cfg: Cfg -> script configuration

OUTPUT:
    - features: np.ndarray -> features x model_time x images activation tensor
    - model_times: np.ndarray -> ConvRNN timestep indices
    - feature_path: Path -> loaded feature file
"""
def load_convrnn_timeseries(paths, cfg):
    feature_path = Path(paths["data_path"]) / "models" / (
        f"{cfg.folder_name}_{cfg.target_model_name}_{cfg.target_img_size}_"
        f"{cfg.target_layer}_features_timeseries_{cfg.target_pooling}pool.npz"
    )
    if not feature_path.exists():
        raise FileNotFoundError(feature_path)
    # end if not feature_path.exists():
    with np.load(feature_path, allow_pickle=True) as data:
        features = data["arr_0"]
        model_times = data["times"]
    # end with np.load(...)
    if features.ndim != 3:
        raise ValueError(f"Expected features x time x images tensor, got {features.shape}")
    # end if features.ndim != 3:
    return features, model_times, feature_path
# EOF


"""
load_neural_raster
Load and slice the monkey neural raster.

INPUT:
    - paths: dict -> project path config
    - cfg: Cfg -> script configuration

OUTPUT:
    - neural_features: np.ndarray -> neurons x neural_time x images tensor
"""
def load_neural_raster(paths, cfg):
    raster = load_img_natraster(
        paths,
        cfg.monkey_name,
        cfg.date,
        new_fs=cfg.new_fs,
        brain_area=cfg.brain_area,
    )
    neural_features = raster.get_array()
    if neural_features.ndim != 3:
        raise ValueError(
            f"Expected neurons x time x images neural tensor, got {neural_features.shape}"
        )
    # end if neural_features.ndim != 3:
    return neural_features
# EOF


"""
get_monkey_image_order
Compute the index mapping from ANN image order to monkey image order.

INPUT:
    - paths: dict -> project path config
    - cfg: Cfg -> script configuration

OUTPUT:
    - idx_ord: np.ndarray -> image indices selecting ANN features in monkey order
"""
def get_monkey_image_order(paths, cfg):
    stimuli_root = Path(paths["livingstone_lab"]) / "Stimuli" / cfg.folder_name
    dataset = SimpleImageFolder(stimuli_root)
    idx_ord = map_image_order_from_ann_to_monkey(
        paths, cfg.monkey_name, cfg.date, dataset
    )
    return np.asarray(idx_ord, dtype=int)
# EOF


"""
rdm_timeseries
Compute one RDM vector for each timepoint in a features x time x images tensor.

INPUT:
    - features: np.ndarray -> features x time x images tensor
    - metric: str -> RDM metric passed to rdm_vector

OUTPUT:
    - rdms: np.ndarray -> time x image-pair RDM matrix
"""
def rdm_timeseries(features, metric):
    rdms = []
    for time_idx in range(features.shape[1]):
        curr_rdm = rdm_vector(features[:, time_idx, :], metric=metric)
        rdms.append(curr_rdm.astype(np.float32))
    # end for time_idx in range(features.shape[1]):
    return np.stack(rdms, axis=0)
# EOF


"""
prepare_rdm_matrix
Rank and normalize RDM time series for fast RSA matrix computation.

INPUT:
    - rdms: np.ndarray -> time x image-pair RDM matrix
    - RSA_metric: str -> "spearman" or "pearson"

OUTPUT:
    - prepared: np.ndarray -> centered unit-norm RDM matrix
"""
def prepare_rdm_matrix(rdms, RSA_metric):
    prepared = np.asarray(rdms, dtype=np.float64)
    if RSA_metric == "spearman":
        prepared = np.stack([rankdata_average(row) for row in prepared], axis=0)
    elif RSA_metric != "pearson":
        raise ValueError(f"Unsupported RSA metric: {RSA_metric}")
    # end if RSA_metric == "spearman":
    prepared = prepared - prepared.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(prepared, axis=1, keepdims=True)
    norms[norms == 0] = np.nan
    return prepared / norms
# EOF


"""
compute_dynamic_drsa
Compute dynamic RSA between neural and model RDM time series.

INPUT:
    - neural_rdms: np.ndarray -> neural_time x image-pair RDM matrix
    - model_rdms: np.ndarray -> model_time x image-pair RDM matrix
    - RSA_metric: str -> "spearman" or "pearson"

OUTPUT:
    - drsa: np.ndarray -> neural_time x model_time dynamic RSA matrix
"""
def compute_dynamic_drsa(neural_rdms, model_rdms, RSA_metric):
    neural_prepared = prepare_rdm_matrix(neural_rdms, RSA_metric)
    model_prepared = prepare_rdm_matrix(model_rdms, RSA_metric)
    return (neural_prepared @ model_prepared.T).astype(np.float32)
# EOF


"""
time_axis_ms
Create a millisecond time axis from a sampling frequency.

INPUT:
    - n_time: int -> number of timepoints
    - fs: float -> sampling frequency in Hz
    - start_ms: float -> first timepoint in ms

OUTPUT:
    - times_ms: np.ndarray -> time axis in ms
"""
def time_axis_ms(n_time, fs, start_ms=0.0):
    return start_ms + np.arange(n_time, dtype=float) * (1000.0 / fs)
# EOF


if __name__ == "__main__":
    cfg = parse_args()
    paths = load_paths(PROJECT_ROOT, cfg.env)
    results_dir = Path(paths["data_path"]) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    neural_features = load_neural_raster(paths, cfg)
    model_features, model_times, model_feature_path = load_convrnn_timeseries(paths, cfg)
    idx_ord = get_monkey_image_order(paths, cfg)
    model_features = model_features[:, :, idx_ord]

    if neural_features.shape[2] != model_features.shape[2]:
        raise ValueError(
            "Neural and ConvRNN image counts differ after reordering: "
            f"{neural_features.shape[2]} and {model_features.shape[2]}."
        )
    # end if neural_features.shape[2] != model_features.shape[2]:

    print(f"neural features: {neural_features.shape}")
    print(f"model features: {model_features.shape}")
    print(f"model feature file: {model_feature_path}")

    neural_rdms = rdm_timeseries(neural_features, cfg.signal_RDM_metric)
    model_rdms = rdm_timeseries(model_features, cfg.model_RDM_metric)
    drsa = compute_dynamic_drsa(neural_rdms, model_rdms, cfg.RSA_metric)

    neural_times_ms = time_axis_ms(
        neural_features.shape[1], cfg.new_fs, cfg.neural_time_start_ms
    )
    model_times_ms = model_times.astype(float) * cfg.model_timestep_ms
    best_model_time_idx = np.nanargmax(drsa, axis=1)
    best_model_time = model_times[best_model_time_idx]
    best_model_time_ms = model_times_ms[best_model_time_idx]

    out_name = cfg.out_name
    if out_name is None:
        out_name = (
            f"dynamic_dRSA_{cfg.signal_RDM_metric}-{cfg.model_RDM_metric}_"
            f"{cfg.monkey_name}_{cfg.date}_{cfg.brain_area}_"
            f"{cfg.target_model_name}_{cfg.target_layer}_"
            f"{cfg.new_fs}Hz_{cfg.RSA_metric}.npz"
        )
    # end if out_name is None:

    np.savez_compressed(
        results_dir / out_name,
        drsa=drsa,
        neural_rdms=neural_rdms,
        model_rdms=model_rdms,
        neural_times_ms=neural_times_ms,
        model_times=model_times,
        model_times_ms=model_times_ms,
        best_model_time_idx=best_model_time_idx,
        best_model_time=best_model_time,
        best_model_time_ms=best_model_time_ms,
        image_order_idx=idx_ord,
        cfg=cfg.__dict__,
    )
    print(f"saved {results_dir / out_name}")
    print(f"dRSA shape: {drsa.shape}")
# EOF
