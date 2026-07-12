import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "python_scripts" / "src"))

from IT_recap.convrnn_analysis import (  # noqa: E402
    list_imagefolder_files,
    load_paths,
    pool_features,
    srp_project,
)


@dataclass
class Cfg:
    folder_name: str = "talia_20each_tizi"
    model_name: str = "rgc_intermediate"
    img_size: int = 224
    pooling: str = "mean"
    layers: str = "conv9,conv10"
    batch_size: int = 16
    image_pres: str = "neural"
    times: Optional[int] = None
    image_off: Optional[int] = None
    include_all_times: bool = False
    ckpt_dir: str = "third_party/convrnns/ckpts"
    output_model_name: Optional[str] = None
    srp_dim: Optional[int] = None
    random_seed: int = 0
    env: Optional[str] = None


"""
parse_args
Parse command-line arguments into a Cfg object.

OUTPUT:
    - cfg: Cfg -> script configuration
"""
def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract Nayebi et al. ConvRNN features for ImageFolder stimuli."
    )
    parser.add_argument("--folder_name", default=Cfg.folder_name)
    parser.add_argument("--model_name", default=Cfg.model_name)
    parser.add_argument("--img_size", type=int, default=Cfg.img_size)
    parser.add_argument("--pooling", choices=("mean", "all", "none"), default=Cfg.pooling)
    parser.add_argument("--layers", default=Cfg.layers)
    parser.add_argument("--batch_size", type=int, default=Cfg.batch_size)
    parser.add_argument("--image_pres", choices=("default", "constant", "neural"), default=Cfg.image_pres)
    parser.add_argument("--times", type=int, default=Cfg.times)
    parser.add_argument("--image_off", type=int, default=Cfg.image_off)
    parser.add_argument("--include_all_times", action="store_true")
    parser.add_argument("--ckpt_dir", default=Cfg.ckpt_dir)
    parser.add_argument("--output_model_name", default=Cfg.output_model_name)
    parser.add_argument("--srp_dim", type=int, default=Cfg.srp_dim)
    parser.add_argument("--random_seed", type=int, default=Cfg.random_seed)
    parser.add_argument("--env", default=Cfg.env)
    args = parser.parse_args()
    return Cfg(**vars(args))
# EOF


"""
load_image_batch
Load and ImageNet-normalize images for ConvRNN inference.

INPUT:
    - image_files: list[Path] -> image paths
    - img_size: int -> square resize target

OUTPUT:
    - batch: np.ndarray -> normalized images, shape images x H x W x 3
"""
def load_image_batch(image_files, img_size):
    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    batch = []
    for image_file in image_files:
        img = Image.open(image_file).convert("RGB")
        img = img.resize((img_size, img_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - imagenet_mean) / imagenet_std
        batch.append(arr)
    # end for image_file in image_files:
    return np.stack(batch, axis=0)
# EOF


"""
save_layer_timeseries
Save one layer as both a dynamic feature tensor and static per-time feature files.

INPUT:
    - features_by_time: dict[int, list[np.ndarray]] -> time-indexed batches
    - paths: dict -> project path config
    - cfg: Cfg -> script configuration
    - layer: str -> ConvRNN layer name
    - model_label: str -> output model label
"""
def save_layer_timeseries(features_by_time, paths, cfg, layer, model_label):
    out_dir = Path(paths["data_path"]) / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    times = np.array(sorted(features_by_time.keys()), dtype=int)
    time_features = []
    for time in times:
        curr = np.concatenate(features_by_time[time], axis=0)
        curr = curr.T.astype(np.float32)
        time_features.append(curr)
        static_layer = f"{layer}_t{time:02d}"
        static_name = (
            f"{cfg.folder_name}_{model_label}_{cfg.img_size}_"
            f"{static_layer}_features_{cfg.pooling}pool.npz"
        )
        np.savez_compressed(out_dir / static_name, curr, time=time, layer=layer)
    # end for time in times:
    features = np.stack(time_features, axis=1)
    timeseries_name = (
        f"{cfg.folder_name}_{model_label}_{cfg.img_size}_"
        f"{layer}_features_timeseries_{cfg.pooling}pool.npz"
    )
    np.savez_compressed(out_dir / timeseries_name, features, times=times, layer=layer)
    print(f"saved {out_dir / timeseries_name} with shape {features.shape}")
# EOF


if __name__ == "__main__":
    cfg = parse_args()
    paths = load_paths(PROJECT_ROOT, cfg.env)
    convrnns_root = PROJECT_ROOT / "third_party" / "convrnns"
    sys.path.insert(0, str(convrnns_root))
    os.chdir(convrnns_root)

    import tensorflow as tf  # noqa: E402
    from convrnns.models.model_func import model_func  # noqa: E402
    from convrnns.utils.loader import MODEL_TO_KWARGS, get_restore_vars  # noqa: E402

    if not hasattr(tf, "Session") or not hasattr(tf, "contrib"):
        raise RuntimeError(
            "The ConvRNN repo requires TensorFlow 1.x with tf.contrib. "
            "Use a Python 3.6/3.7 environment with tensorflow==1.13.1."
        )
    # end if not hasattr(tf, "Session") ...

    layers = [layer.strip() for layer in cfg.layers.split(",") if layer.strip()]
    model_label = cfg.output_model_name or cfg.model_name
    stimuli_root = Path(paths["livingstone_lab"]) / "Stimuli" / cfg.folder_name
    image_files = list_imagefolder_files(stimuli_root)
    if len(image_files) == 0:
        raise FileNotFoundError(f"No images found under {stimuli_root}")
    # end if len(image_files) == 0:

    inputs = tf.placeholder(
        tf.float32, shape=[cfg.batch_size, cfg.img_size, cfg.img_size, 3]
    )
    y = model_func(
        inputs=inputs,
        out_layers=layers,
        image_pres=cfg.image_pres,
        times=cfg.times,
        image_off=cfg.image_off,
        include_all_times=cfg.include_all_times,
        include_logits=False,
        **MODEL_TO_KWARGS[cfg.model_name],
    )

    ckpt_path = Path(PROJECT_ROOT / cfg.ckpt_dir) / cfg.model_name / "model.ckpt"
    restore_vars = get_restore_vars(str(ckpt_path))
    sess = tf.Session()
    tf.train.Saver(var_list=restore_vars).restore(sess, str(ckpt_path))

    features_by_layer = {layer: {} for layer in layers}
    for start in range(0, len(image_files), cfg.batch_size):
        end = min(start + cfg.batch_size, len(image_files))
        curr_files = image_files[start:end]
        batch = load_image_batch(curr_files, cfg.img_size)
        if batch.shape[0] < cfg.batch_size:
            pad = np.repeat(batch[-1:], cfg.batch_size - batch.shape[0], axis=0)
            batch = np.concatenate([batch, pad], axis=0)
        # end if batch.shape[0] < cfg.batch_size:
        y_eval = sess.run(y, feed_dict={inputs: batch})
        valid_n = end - start
        for layer in layers:
            for time, arr in y_eval[layer].items():
                pooled = pool_features(arr[:valid_n], cfg.pooling)
                pooled = pooled.reshape(pooled.shape[0], -1)
                pooled = srp_project(
                    pooled,
                    n_components=cfg.srp_dim,
                    random_seed=cfg.random_seed + int(time),
                )
                features_by_layer[layer].setdefault(int(time), []).append(pooled)
            # end for time, arr in y_eval[layer].items():
        # end for layer in layers:
        print(f"processed images {end}/{len(image_files)}")
    # end for start in range(...)

    sess.close()
    for layer in layers:
        save_layer_timeseries(features_by_layer[layer], paths, cfg, layer, model_label)
    # end for layer in layers:
# EOF
