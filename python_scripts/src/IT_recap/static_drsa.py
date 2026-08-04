import csv
from pathlib import Path

import numpy as np

from IT_recap.convrnn_analysis import rankdata_average, rdm_vector


"""
static_feature_path
Build the established feature filename for one static model layer.

INPUT:
    - model_dir: Path -> directory containing model activations
    - dataset_name: str -> stimulus dataset name
    - model_name: str -> static model name
    - img_size: int -> model input resolution
    - layer_name: str -> static model layer name
    - pooling: str -> activation pooling method

OUTPUT:
    - feature_path: Path -> layer activation file path
"""
def static_feature_path(
    model_dir,
    dataset_name,
    model_name,
    img_size,
    layer_name,
    pooling,
):
    file_name = (
        f"{dataset_name}_{model_name}_{img_size}_{layer_name}"
        f"_features_{pooling}pool.npz"
    )
    return Path(model_dir) / file_name
# EOF


"""
load_static_model_features
Load ordered feature matrices for the requested static model layers.

INPUT:
    - model_dir: Path -> directory containing model activations
    - dataset_name: str -> stimulus dataset name
    - model_name: str -> static model name
    - img_size: int -> model input resolution
    - layer_names: list[str] -> ordered static model layers
    - pooling: str -> activation pooling method

OUTPUT:
    - features_by_layer: dict[str, np.ndarray] -> features x images matrices
"""
def load_static_model_features(
    model_dir,
    dataset_name,
    model_name,
    img_size,
    layer_names,
    pooling,
):
    features_by_layer = {}
    for layer_name in layer_names:
        feature_path = static_feature_path(
            model_dir,
            dataset_name,
            model_name,
            img_size,
            layer_name,
            pooling,
        )
        if not feature_path.exists():
            raise FileNotFoundError(feature_path)
        # end if not feature_path.exists():
        with np.load(feature_path) as feature_file:
            features_by_layer[layer_name] = feature_file["arr_0"]
        # end with np.load(feature_path) as feature_file:
    # end for layer_name in layer_names:
    return features_by_layer
# EOF


"""
prepare_rdm_for_rsa
Rank when requested, center, and unit-normalize one RDM vector for correlation.

INPUT:
    - rdm: np.ndarray -> condensed RDM vector
    - rsa_metric: str -> "spearman" or "pearson"

OUTPUT:
    - prepared_rdm: np.ndarray -> centered unit-length RDM vector
"""
def prepare_rdm_for_rsa(rdm, rsa_metric):
    prepared_rdm = np.asarray(rdm, dtype=np.float64)
    if rsa_metric == "spearman":
        prepared_rdm = rankdata_average(prepared_rdm)
    elif rsa_metric != "pearson":
        raise ValueError(f"Unsupported RSA metric: {rsa_metric}")
    # end if rsa_metric == "spearman":

    prepared_rdm = prepared_rdm - prepared_rdm.mean()
    norm = np.linalg.norm(prepared_rdm)
    if norm == 0:
        return np.full(prepared_rdm.shape, np.nan)
    # end if norm == 0:
    return prepared_rdm / norm
# EOF


"""
compute_static_drsa
Compare every static layer RDM with every dynamic target timepoint RDM.

INPUT:
    - target_features: np.ndarray -> target features x time x images
    - times: np.ndarray -> target model timestep indices
    - features_by_layer: dict[str, np.ndarray] -> static features x images matrices
    - signal_rdm_metric: str -> target RDM distance metric
    - model_rdm_metric: str -> static model RDM distance metric
    - rsa_metric: str -> RDM comparison metric

OUTPUT:
    - drsa: np.ndarray -> target time x static layer similarity matrix
    - best_layer_idx: np.ndarray -> best static layer index per target time
    - best_layer_name: np.ndarray -> best static layer name per target time
"""
def compute_static_drsa(
    target_features,
    times,
    features_by_layer,
    signal_rdm_metric,
    model_rdm_metric,
    rsa_metric,
):
    layer_names = list(features_by_layer)
    prepared_model_rdms = np.stack(
        [
            prepare_rdm_for_rsa(
                rdm_vector(features_by_layer[layer_name], metric=model_rdm_metric),
                rsa_metric,
            )
            for layer_name in layer_names
        ]
    )
    drsa = np.zeros((len(times), len(layer_names)), dtype=np.float32)

    for time_idx, time in enumerate(times):
        prepared_signal_rdm = prepare_rdm_for_rsa(
            rdm_vector(
                target_features[:, time_idx, :],
                metric=signal_rdm_metric,
            ),
            rsa_metric,
        )
        drsa[time_idx] = prepared_model_rdms @ prepared_signal_rdm
        best_idx = int(np.nanargmax(drsa[time_idx]))
        print(
            f"time={int(time):02d} best={layer_names[best_idx]} "
            f"dRSA={drsa[time_idx, best_idx]:.4f}"
        )
    # end for time_idx, time in enumerate(times):

    best_layer_idx = np.nanargmax(drsa, axis=1)
    best_layer_name = np.array([layer_names[idx] for idx in best_layer_idx])
    return drsa, best_layer_idx, best_layer_name
# EOF


"""
static_drsa_result_name
Build a general static-dRSA result filename.

INPUT:
    - signal_rdm_metric: str -> target RDM distance metric
    - model_rdm_metric: str -> static model RDM distance metric
    - dataset_name: str -> stimulus dataset name
    - target_model_name: str -> dynamic target model name
    - target_layer: str -> dynamic target layer name
    - model_name: str -> static model name
    - rsa_metric: str -> RDM comparison metric

OUTPUT:
    - file_name: str -> static-dRSA result filename
"""
def static_drsa_result_name(
    signal_rdm_metric,
    model_rdm_metric,
    dataset_name,
    target_model_name,
    target_layer,
    model_name,
    rsa_metric,
):
    return (
        f"static_dRSA_{signal_rdm_metric}-{model_rdm_metric}_"
        f"{dataset_name}_{target_model_name}_{target_layer}_"
        f"target_vs_{model_name}_{rsa_metric}.npz"
    )
# EOF


"""
save_static_drsa_result
Save a general static-dRSA result with ordered model metadata.

INPUT:
    - result_path: Path -> output NPZ path
    - drsa: np.ndarray -> target time x static layer similarity matrix
    - times: np.ndarray -> target model timestep indices
    - layer_names: list[str] -> ordered static model layers
    - best_layer_idx: np.ndarray -> best static layer index per timepoint
    - best_layer_name: np.ndarray -> best static layer name per timepoint
    - cfg: dict -> analysis configuration

OUTPUT:
    - None
"""
def save_static_drsa_result(
    result_path,
    drsa,
    times,
    layer_names,
    best_layer_idx,
    best_layer_name,
    cfg,
):
    result_path = Path(result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        result_path,
        drsa=drsa,
        rsa=drsa,
        times=times,
        model_layers=np.array(layer_names),
        best_layer_idx=best_layer_idx,
        best_layer_name=best_layer_name,
        cfg=cfg,
    )
# EOF


"""
load_static_drsa_result
Load old or general static-dRSA results into a consistent dictionary.

INPUT:
    - result_path: Path -> static-dRSA NPZ file
    - timestep_ms: float -> milliseconds represented by one target timestep

OUTPUT:
    - result: dict -> dRSA matrix, times, layers, best layers, config, and path
"""
def load_static_drsa_result(result_path, timestep_ms=10.0):
    result_path = Path(result_path)
    with np.load(result_path, allow_pickle=True) as result_file:
        drsa = (
            result_file["drsa"]
            if "drsa" in result_file.files
            else result_file["rsa"]
        )
        layers = (
            result_file["model_layers"]
            if "model_layers" in result_file.files
            else result_file["vit_layers"]
        )
        cfg = result_file["cfg"].item() if "cfg" in result_file.files else {}
        times = result_file["times"]
        best_layer_idx = result_file["best_layer_idx"]
        best_layer_name = result_file["best_layer_name"]
    # end with np.load(result_path, allow_pickle=True) as result_file:

    if drsa.shape != (len(times), len(layers)):
        raise ValueError(f"dRSA shape {drsa.shape} does not match times x layers")
    # end if drsa.shape != (len(times), len(layers)):

    return {
        "path": result_path,
        "drsa": drsa,
        "times": times,
        "time_ms": times * timestep_ms,
        "layers": layers,
        "best_layer_idx": best_layer_idx,
        "best_layer_name": best_layer_name,
        "cfg": cfg,
        "model_name": cfg.get("model_name", "unknown_model"),
    }
# EOF


"""
find_static_drsa_results
Discover and load one matching static-dRSA result per requested model.

INPUT:
    - results_dir: Path -> directory containing result NPZ files
    - pattern: str -> glob pattern selecting comparable static-dRSA files
    - timestep_ms: float -> milliseconds represented by one target timestep
    - model_names: list[str] | None -> optional ordered model selection

OUTPUT:
    - results_by_model: dict[str, dict] -> loaded results keyed by model name
"""
def find_static_drsa_results(
    results_dir,
    pattern,
    timestep_ms=10.0,
    model_names=None,
):
    loaded_results = [
        load_static_drsa_result(path, timestep_ms=timestep_ms)
        for path in sorted(Path(results_dir).glob(pattern))
    ]
    if model_names is None:
        return {result["model_name"]: result for result in loaded_results}
    # end if model_names is None:

    loaded_by_model = {result["model_name"]: result for result in loaded_results}
    missing_models = [name for name in model_names if name not in loaded_by_model]
    if missing_models:
        raise FileNotFoundError(
            f"Missing static-dRSA results for: {', '.join(missing_models)}"
        )
    # end if missing_models:
    return {name: loaded_by_model[name] for name in model_names}
# EOF


"""
layer_depth_labels
Create compact ordinal depth labels independent of model layer naming syntax.

INPUT:
    - layers: np.ndarray -> ordered static model layer names

OUTPUT:
    - labels: list[str] -> zero-based layer-depth labels
"""
def layer_depth_labels(layers):
    return [str(layer_idx) for layer_idx in range(len(layers))]
# EOF


"""
temporal_centroids
Compute the temporal centroid of every static-layer similarity curve.

INPUT:
    - time_ms: np.ndarray -> target time from image onset in milliseconds
    - drsa: np.ndarray -> target time x static layer similarity matrix
    - weight_floor: float -> minimum similarity retained as centroid weight

OUTPUT:
    - centroids_ms: np.ndarray -> centroid time for every static layer
"""
def temporal_centroids(time_ms, drsa, weight_floor=0.0):
    weights = np.array(drsa, dtype=float, copy=True)
    weights[weights < weight_floor] = 0.0
    weight_sums = weights.sum(axis=0)
    centroids_ms = np.full(weights.shape[1], np.nan, dtype=float)
    valid = weight_sums > 0
    centroids_ms[valid] = (
        time_ms[:, None] * weights[:, valid]
    ).sum(axis=0) / weight_sums[valid]
    return centroids_ms
# EOF


"""
save_static_drsa_figure
Save one static-dRSA Matplotlib figure using the result stem.

INPUT:
    - fig: matplotlib.figure.Figure -> figure to save
    - figures_dir: Path -> output directory
    - result_path: Path -> source static-dRSA result path
    - suffix: str -> figure-specific filename suffix
    - dpi: int -> saved figure resolution

OUTPUT:
    - figure_path: Path -> saved figure path
"""
def save_static_drsa_figure(fig, figures_dir, result_path, suffix, dpi=180):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / f"{Path(result_path).stem}_{suffix}.png"
    fig.savefig(figure_path, dpi=dpi, bbox_inches="tight")
    return figure_path
# EOF


"""
save_centroid_table
Save static-model temporal centroids with layer names and depths.

INPUT:
    - table_path: Path -> output CSV path
    - layers: np.ndarray -> ordered static model layer names
    - centroids_ms: np.ndarray -> temporal centroid per static layer

OUTPUT:
    - None
"""
def save_centroid_table(table_path, layers, centroids_ms):
    table_path = Path(table_path)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with open(table_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer_depth", "layer_name", "centroid_ms_from_image_onset"])
        for layer_idx, (layer_name, centroid_ms) in enumerate(
            zip(layers, centroids_ms)
        ):
            writer.writerow([layer_idx, str(layer_name), float(centroid_ms)])
        # end for layer_idx, (layer_name, centroid_ms) in enumerate(...):
    # end with open(table_path, "w", newline="") as f:
# EOF
