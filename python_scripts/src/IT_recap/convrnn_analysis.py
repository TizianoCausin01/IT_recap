import os
from pathlib import Path
from typing import Optional

import numpy as np
import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIT_L_16_TIMM_LAYERS = [f"blocks.{idx}.mlp.fc2" for idx in range(24)]


"""
load_paths
Load the project config paths for the active environment.

INPUT:
    - project_root: Path -> root folder containing config.yaml
    - env: str | None -> optional environment name

OUTPUT:
    - paths: dict -> environment-specific path mapping
"""
def load_paths(project_root: Path, env: Optional[str] = None):
    env = os.getenv("MY_ENV", "tiziano_mac_mini") if env is None else env
    with open(project_root / "config.yaml", "r") as f:
        config = yaml.safe_load(f)
    # end with open(...)
    return config[env]["paths"]
# EOF


"""
list_imagefolder_files
Replicate torchvision.datasets.ImageFolder sample ordering without requiring PyTorch.

INPUT:
    - root: str | Path -> image-folder root with one class directory per category

OUTPUT:
    - image_files: list[Path] -> image paths sorted by class and filename
"""
def list_imagefolder_files(root):
    root = Path(root)
    classes = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)
    image_files = []
    for class_dir in classes:
        curr_files = [
            p
            for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        image_files.extend(sorted(curr_files, key=lambda p: p.name))
    # end for class_dir in classes:
    return image_files
# EOF


"""
pool_features
Pool or flatten ConvRNN activations.

INPUT:
    - features: np.ndarray -> activations with images on axis 0
    - pooling: str -> "mean", "all", or "none"

OUTPUT:
    - pooled_features: np.ndarray -> images x features matrix
"""
def pool_features(features, pooling):
    if pooling == "none":
        return features
    if features.ndim == 4 and pooling == "mean":
        return features.mean(axis=(1, 2))
    if features.ndim == 3 and pooling == "mean":
        return features.mean(axis=1)
    if pooling == "all":
        return features.reshape(features.shape[0], -1)
    if features.ndim == 2:
        return features
    raise ValueError(f"Unsupported pooling={pooling} for activation shape {features.shape}")
# EOF


"""
srp_project
Apply a deterministic dense signed random projection to reduce feature dimension.

INPUT:
    - features: np.ndarray -> images x features matrix
    - n_components: int | None -> target feature count
    - random_seed: int -> projection seed

OUTPUT:
    - projected_features: np.ndarray -> images x n_components matrix
"""
def srp_project(features, n_components=None, random_seed=0):
    if n_components is None or features.shape[1] <= n_components:
        return features
    # end if n_components is None or features.shape[1] <= n_components:
    rng = np.random.default_rng(random_seed)
    projection = rng.choice(
        np.array([-1.0, 1.0], dtype=np.float32),
        size=(features.shape[1], n_components),
    )
    projection /= np.sqrt(n_components)
    return features @ projection
# EOF


"""
rdm_vector
Compute the upper-triangle vector of a representational dissimilarity matrix.

INPUT:
    - features: np.ndarray -> features x images matrix
    - metric: str -> "correlation", "cosine", "cosine_cnt", or "euclidean"

OUTPUT:
    - rdm_vec: np.ndarray -> condensed RDM vector
"""
def rdm_vector(features, metric="correlation"):
    features = np.asarray(features, dtype=np.float64)
    if metric == "cosine_cnt":
        features = features - features.mean(axis=1, keepdims=True)
    # end if metric == "cosine_cnt":
    x = features.T
    if metric == "correlation":
        x = x - x.mean(axis=1, keepdims=True)
    # end if metric == "correlation":
    if metric in ("correlation", "cosine", "cosine_cnt"):
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        x = x / norms
        sim = x @ x.T
        rdm = 1.0 - sim
    elif metric == "euclidean":
        diffs = x[:, None, :] - x[None, :, :]
        rdm = np.sqrt(np.sum(diffs * diffs, axis=-1))
    else:
        raise ValueError(f"Unsupported RDM metric: {metric}")
    # end if metric in (...)
    triu = np.triu_indices(rdm.shape[0], k=1)
    return rdm[triu]
# EOF


"""
rsa_similarity
Compare two condensed RDM vectors.

INPUT:
    - rdm_a: np.ndarray -> first RDM vector
    - rdm_b: np.ndarray -> second RDM vector
    - metric: str -> "pearson" or "spearman"

OUTPUT:
    - similarity: float -> correlation between RDM vectors
"""
def rsa_similarity(rdm_a, rdm_b, metric="spearman"):
    x = np.asarray(rdm_a, dtype=np.float64)
    y = np.asarray(rdm_b, dtype=np.float64)
    if metric == "spearman":
        x = rankdata_average(x)
        y = rankdata_average(y)
    elif metric != "pearson":
        raise ValueError(f"Unsupported RSA metric: {metric}")
    # end if metric == "spearman":
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return np.nan
    return float((x @ y) / denom)
# EOF


"""
rankdata_average
Compute average ranks for a 1D array, matching scipy.stats.rankdata(method="average").

INPUT:
    - values: np.ndarray -> vector to rank

OUTPUT:
    - ranks: np.ndarray -> average ranks
"""
def rankdata_average(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        # end while tied values
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    # end while start < len(values):
    return ranks
# EOF
