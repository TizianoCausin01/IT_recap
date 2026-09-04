import argparse
import copy
import csv
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import RidgeCV
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import ImageFolder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_SRC = PROJECT_ROOT / "python_scripts" / "src"
sys.path.insert(0, str(PROJECT_SRC))

from model_classes.temporal_experiment_models import (  # noqa: E402
    MODEL_CLASSES,
    build_temporal_experiment_model,
)


@dataclass
class Cfg:
    # Environment, cached feature, and neural recording parameters.
    env: str | None = None
    data_root: str | None = None
    stimuli_root: str | None = None
    output_dir: str | None = None
    folder_name: str = "talia_20each_tizi"
    model_name: str = "dino_v3_l"
    img_size: int = 224
    pooling: str = "mean"
    layer_names: str = (
        "layer.3.mlp.down_proj,layer.13.mlp.down_proj,"
        "layer.16.mlp.down_proj,layer.20.mlp.down_proj"
    )
    monkey_name: str = "three0"
    date: str = "250313"
    brain_area: str = "AIT"
    original_fs: int = 1000
    target_fs: int = 100
    time_start_ms: float = 0.0
    time_end_ms: float = 300.0

    # A held-out test is the default; notebook mode reproduces its 80/20 split.
    protocol: str = "holdout"  # "holdout" or "notebook"
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    split_seed: int = 0

    # Architecture sweep and optimization.
    architectures: str = ",".join(MODEL_CLASSES)
    model_seeds: str = "0"
    hidden_dim: int = 128
    time_embedding_dim: int = 32
    attention_dim: int = 64
    n_attention_heads: int = 4
    dropout: float = 0.2
    batch_size: int = 64
    epochs: int = 250
    minimum_epochs: int = 30
    patience: int = 35
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    num_workers: int = 0
    device: str = "auto"
    save_checkpoints: bool = False
    save_predictions: bool = True
    smoke_test: bool = False


"""
parse_args
Parse command-line experiment overrides into a configuration object.

OUTPUT:
    - cfg: Cfg -> data, split, architecture, and optimization settings
"""
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare recurrent and temporal-attention neural decoders."
    )
    for field_name, field_definition in Cfg.__dataclass_fields__.items():
        default = field_definition.default
        argument_name = f"--{field_name}"
        if isinstance(default, bool):
            parser.add_argument(
                argument_name,
                action=argparse.BooleanOptionalAction,
                default=default,
            )
        elif default is None:
            parser.add_argument(argument_name, default=default)
        else:
            parser.add_argument(argument_name, type=type(default), default=default)
        # end if boolean, optional string, or typed argument
    # end for configuration field
    return Cfg(**vars(parser.parse_args()))
# EOF


"""
load_project_paths
Read the active environment paths and expose useful_stuff before data imports.

INPUT:
    - cfg: Cfg -> optional environment and data-root overrides

OUTPUT:
    - paths: dict -> active project paths
"""
def load_project_paths(cfg):
    env = cfg.env or os.getenv("MY_ENV", "tiziano_mac_mini")
    with open(PROJECT_ROOT / "config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
    # end with project configuration
    if env not in config:
        raise KeyError(f"Environment {env!r} is not configured.")
    # end if environment is unavailable
    paths = dict(config[env]["paths"])
    if cfg.data_root is not None:
        paths["data_path"] = cfg.data_root
    # end if data root is overridden
    sys.path.append(paths["useful_stuff_path"])
    return paths
# EOF


"""
load_aligned_cached_data
Load notebook-matched DINO features and AIT targets without instantiating DINO.

INPUT:
    - cfg: Cfg -> recording, stimulus, and feature-cache parameters
    - paths: dict -> resolved local data paths

OUTPUT:
    - features: np.ndarray -> aligned [images, layers, embedding] features
    - targets: np.ndarray -> aligned [images, time, neurons] activity
    - image_indices: np.ndarray -> ImageFolder identity of each neural sample
"""
def load_aligned_cached_data(cfg, paths):
    # Delay imports until useful_stuff is on sys.path.
    from IT_recap.hf_feature_extraction import (
        is_valid_image_file,
        load_hf_layer_features,
    )
    from project_specific_utils.dataloader import (
        load_img_natraster,
        map_image_order_from_ann_to_monkey,
    )

    stimuli_root = Path(
        cfg.stimuli_root
        or Path(paths["livingstone_lab"]) / "Stimuli" / cfg.folder_name
    ).expanduser()
    image_dataset = ImageFolder(
        stimuli_root,
        is_valid_file=is_valid_image_file,
        allow_empty=True,
    )
    layer_names = [name.strip() for name in cfg.layer_names.split(",") if name]
    features = load_hf_layer_features(
        output_dir=Path(paths["data_path"]) / "models",
        dataset_name=cfg.folder_name,
        model_name=cfg.model_name,
        img_size=cfg.img_size,
        layer_names=layer_names,
        pooling=cfg.pooling,
    )
    raster = load_img_natraster(
        paths=paths,
        monkey_name=cfg.monkey_name,
        date=cfg.date,
        original_fs=cfg.original_fs,
        new_fs=cfg.target_fs,
        time_start_ms=cfg.time_start_ms,
        time_end_ms=cfg.time_end_ms,
        brain_area=cfg.brain_area,
    )
    image_indices = np.asarray(
        map_image_order_from_ann_to_monkey(
            paths,
            cfg.monkey_name,
            cfg.date,
            image_dataset,
        ),
        dtype=int,
    )
    targets = raster.get_array().transpose(2, 1, 0).astype(
        np.float32, copy=False
    )
    if len(image_indices) != len(targets):
        raise ValueError(
            f"Found {len(image_indices)} mapped images and {len(targets)} targets."
        )
    # end if the neural and stimulus spaces are misaligned
    features = features[image_indices].astype(np.float32, copy=False)
    return features, targets, image_indices
# EOF


"""
make_split_indices
Create deterministic image splits for strict holdout or notebook comparison.

INPUT:
    - image_indices: np.ndarray -> source-image identity for every neural sample
    - cfg: Cfg -> split fractions, seed, and protocol

OUTPUT:
    - splits: dict[str, np.ndarray] -> train, validation, and test indices
"""
def make_split_indices(image_indices, cfg):
    if not 0.0 < cfg.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one.")
    # end if validation fraction is invalid
    if cfg.protocol not in {"holdout", "notebook"}:
        raise ValueError("protocol must be 'holdout' or 'notebook'.")
    # end if protocol is invalid
    if cfg.protocol == "holdout" and not 0.0 < cfg.test_fraction < 1.0:
        raise ValueError("test_fraction must be between zero and one.")
    # end if test fraction is invalid

    image_indices = np.asarray(image_indices, dtype=int)
    unique_images = np.unique(image_indices)
    shuffled_images = np.random.default_rng(cfg.split_seed).permutation(
        unique_images
    )
    n_validation = max(1, round(len(unique_images) * cfg.validation_fraction))
    validation_images = shuffled_images[:n_validation]
    if cfg.protocol == "notebook":
        train_images = shuffled_images[n_validation:]
        test_images = validation_images
    else:
        n_test = max(1, round(len(unique_images) * cfg.test_fraction))
        if n_validation + n_test >= len(unique_images):
            raise ValueError("Validation and test fractions leave no training data.")
        # end if holdout fractions exhaust the data
        test_images = shuffled_images[n_validation : n_validation + n_test]
        train_images = shuffled_images[n_validation + n_test :]
    # end if notebook or independent-test protocol
    train = np.flatnonzero(np.isin(image_indices, train_images))
    validation = np.flatnonzero(np.isin(image_indices, validation_images))
    test = np.flatnonzero(np.isin(image_indices, test_images))
    return {"train": train, "validation": validation, "test": test}
# EOF


def set_seed(seed):
    """Seed Python, NumPy, and PyTorch for a repeatable model run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
# EOF


def resolve_device(device_name):
    """Resolve an explicit device or select CUDA, MPS, then CPU."""
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
# EOF


def make_loader(features, targets, indices, cfg, shuffle, seed):
    """Build one deterministic tensor DataLoader for an experiment partition."""
    dataset = TensorDataset(
        torch.from_numpy(features[indices]),
        torch.from_numpy(targets[indices]),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        generator=generator,
    )
# EOF


def evaluate_torch_model(model, data_loader, device, return_diagnostics=False):
    """Collect loss, predictions, targets, and optional mean attention."""
    model.eval()
    prediction_batches = []
    target_batches = []
    attention_sum = None
    attention_samples = 0
    with torch.inference_mode():
        for features, targets in data_loader:
            features = features.to(device)
            predictions, diagnostics = model(
                features,
                return_diagnostics=return_diagnostics,
            )
            prediction_batches.append(predictions.cpu())
            target_batches.append(targets)
            if diagnostics is not None and "attention" in diagnostics:
                attention = diagnostics["attention"].detach().cpu()
                batch_attention = attention.sum(dim=0)
                attention_sum = (
                    batch_attention
                    if attention_sum is None
                    else attention_sum + batch_attention
                )
                attention_samples += attention.shape[0]
            # end if attention diagnostics are available
        # end for evaluation batch
    # end with inference mode
    predictions = torch.cat(prediction_batches).numpy()
    targets = torch.cat(target_batches).numpy()
    mean_attention = (
        None if attention_sum is None else (attention_sum / attention_samples).numpy()
    )
    return predictions, targets, mean_attention
# EOF


def prediction_metrics(predictions, targets, train_target_mean):
    """Compute MSE, global R2, and correlations useful for neural prediction."""
    residual = predictions - targets
    mse = float(np.mean(residual**2))
    denominator = np.sum((targets - train_target_mean) ** 2)
    global_r2 = float(1.0 - np.sum(residual**2) / denominator)

    flat_predictions = predictions.reshape(predictions.shape[0], -1)
    flat_targets = targets.reshape(targets.shape[0], -1)
    prediction_centered = flat_predictions - flat_predictions.mean(axis=0)
    target_centered = flat_targets - flat_targets.mean(axis=0)
    correlation_denominator = np.sqrt(
        np.sum(prediction_centered**2, axis=0)
        * np.sum(target_centered**2, axis=0)
    )
    valid_outputs = correlation_denominator > 0
    output_correlations = np.full(flat_targets.shape[1], np.nan)
    output_correlations[valid_outputs] = np.sum(
        prediction_centered[:, valid_outputs]
        * target_centered[:, valid_outputs],
        axis=0,
    ) / correlation_denominator[valid_outputs]

    pattern_prediction = flat_predictions - flat_predictions.mean(axis=1, keepdims=True)
    pattern_target = flat_targets - flat_targets.mean(axis=1, keepdims=True)
    pattern_denominator = np.sqrt(
        np.sum(pattern_prediction**2, axis=1)
        * np.sum(pattern_target**2, axis=1)
    )
    valid_patterns = pattern_denominator > 0
    pattern_correlations = np.full(flat_targets.shape[0], np.nan)
    pattern_correlations[valid_patterns] = np.sum(
        pattern_prediction[valid_patterns] * pattern_target[valid_patterns],
        axis=1,
    ) / pattern_denominator[valid_patterns]

    return {
        "mse": mse,
        "global_r2": global_r2,
        "mean_output_correlation": float(np.nanmean(output_correlations)),
        "median_output_correlation": float(np.nanmedian(output_correlations)),
        "mean_image_pattern_correlation": float(np.nanmean(pattern_correlations)),
    }
# EOF


"""
train_torch_model
Optimize one architecture, select by validation MSE, and restore its best state.

INPUT:
    - model: nn.Module -> temporal decoder
    - loaders: dict -> train and validation loaders
    - cfg: Cfg -> optimization and early-stopping settings
    - device: torch.device -> compute device

OUTPUT:
    - history: list[dict] -> epoch-wise train and validation MSE
    - best_epoch: int -> selected checkpoint epoch
"""
def train_torch_model(model, loaders, cfg, device):
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(5, cfg.patience // 3),
        min_lr=cfg.learning_rate / 100.0,
    )
    cost_function = torch.nn.MSELoss()
    best_state = copy.deepcopy(model.state_dict())
    best_validation = np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_squared_error = 0.0
        train_values = 0
        for features, targets in loaders["train"]:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions, _ = model(features)
            loss = cost_function(predictions, targets)
            loss.backward()
            if cfg.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.gradient_clip
                )
            # end if gradient clipping is enabled
            optimizer.step()
            train_squared_error += loss.item() * targets.numel()
            train_values += targets.numel()
        # end for training batch

        validation_predictions, validation_targets, _ = evaluate_torch_model(
            model, loaders["validation"], device
        )
        train_mse = train_squared_error / train_values
        validation_mse = float(
            np.mean((validation_predictions - validation_targets) ** 2)
        )
        scheduler.step(validation_mse)
        history.append(
            {
                "epoch": epoch,
                "train_mse": train_mse,
                "validation_mse": validation_mse,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        if validation_mse < best_validation - 1e-8:
            best_validation = validation_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        # end if validation improved
        if (
            epoch >= cfg.minimum_epochs
            and epochs_without_improvement >= cfg.patience
        ):
            break
        # end if early stopping patience is exhausted
    # end for training epoch
    model.load_state_dict(best_state)
    return history, best_epoch
# EOF


def fit_ridge_baseline(features, targets, splits):
    """Fit the notebook's concatenated-layer RidgeCV reference."""
    flat_features = features.reshape(features.shape[0], -1)
    flat_targets = targets.reshape(targets.shape[0], -1)
    ridge = RidgeCV(alphas=np.logspace(-6, 3, 10))
    ridge.fit(flat_features[splits["train"]], flat_targets[splits["train"]])
    predictions = ridge.predict(flat_features[splits["test"]]).reshape(
        -1, targets.shape[1], targets.shape[2]
    )
    return ridge, predictions
# EOF


def save_rows(path, rows):
    """Write a homogeneous list of result dictionaries to CSV."""
    if not rows:
        return
    # end if there are no rows to save
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # end with result file
# EOF


def summarize_results(result_rows):
    """Aggregate repeated initialization seeds by architecture."""
    summaries = []
    architectures = sorted({row["architecture"] for row in result_rows})
    for architecture in architectures:
        rows = [
            row for row in result_rows if row["architecture"] == architecture
        ]
        summaries.append(
            {
                "architecture": architecture,
                "n_runs": len(rows),
                "test_mse_mean": float(np.mean([row["test_mse"] for row in rows])),
                "test_mse_std": float(np.std([row["test_mse"] for row in rows])),
                "test_global_r2_mean": float(
                    np.mean([row["test_global_r2"] for row in rows])
                ),
                "test_output_correlation_mean": float(
                    np.mean([row["test_mean_output_correlation"] for row in rows])
                ),
                "parameters_mean": int(
                    round(np.mean([row["parameters"] for row in rows]))
                ),
            }
        )
    # end for architecture
    return sorted(summaries, key=lambda row: row["test_mse_mean"])
# EOF


def main():
    cfg = parse_args()
    if cfg.smoke_test:
        cfg.epochs = min(cfg.epochs, 2)
        cfg.minimum_epochs = 1
        cfg.patience = 1
        cfg.architectures = ",".join(MODEL_CLASSES)
        cfg.model_seeds = "0"
    # end if smoke test is requested
    paths = load_project_paths(cfg)
    features, targets, image_indices = load_aligned_cached_data(cfg, paths)
    splits = make_split_indices(image_indices, cfg)
    device = resolve_device(cfg.device)
    output_dir = Path(
        cfg.output_dir
        or PROJECT_ROOT / "results" / "temporal_architecture_sweep"
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as config_file:
        json.dump(asdict(cfg), config_file, indent=2)
    # end with saved configuration
    np.savez_compressed(output_dir / "split_indices.npz", **splits)

    architectures = [
        name.strip() for name in cfg.architectures.split(",") if name.strip()
    ]
    unknown_architectures = sorted(set(architectures) - set(MODEL_CLASSES))
    if unknown_architectures:
        raise KeyError(f"Unknown architectures: {unknown_architectures}.")
    # end if an architecture name is invalid
    model_seeds = [int(seed) for seed in cfg.model_seeds.split(",")]
    print(
        f"data {features.shape} -> {targets.shape} | protocol {cfg.protocol} | "
        f"split {[len(splits[name]) for name in ('train', 'validation', 'test')]} | "
        f"device {device}"
    )

    train_target_mean = targets[splits["train"]].mean(axis=0, keepdims=True)
    constant_predictions = np.broadcast_to(
        train_target_mean,
        (len(splits["test"]), *train_target_mean.shape[1:]),
    )
    constant_metrics = prediction_metrics(
        constant_predictions,
        targets[splits["test"]],
        train_target_mean,
    )

    ridge_start = time.perf_counter()
    ridge, ridge_predictions = fit_ridge_baseline(features, targets, splits)
    ridge_seconds = time.perf_counter() - ridge_start
    ridge_metrics = prediction_metrics(
        ridge_predictions,
        targets[splits["test"]],
        train_target_mean,
    )
    print(
        f"ridge alpha {ridge.alpha_:.3g} | test MSE {ridge_metrics['mse']:.6f} | "
        f"output r {ridge_metrics['mean_output_correlation']:.4f}"
    )

    result_rows = [
        {
            "architecture": "constant_train_mean",
            "seed": -1,
            "parameters": int(train_target_mean.size),
            "best_epoch": 0,
            "train_seconds": 0.0,
            "validation_mse": float("nan"),
            **{f"test_{name}": value for name, value in constant_metrics.items()},
        },
        {
            "architecture": "ridge_cv",
            "seed": -1,
            "parameters": int(ridge.coef_.size + ridge.intercept_.size),
            "best_epoch": 0,
            "train_seconds": ridge_seconds,
            "validation_mse": float("nan"),
            **{f"test_{name}": value for name, value in ridge_metrics.items()},
        },
    ]
    if cfg.save_predictions:
        np.savez_compressed(
            output_dir / "predictions_ridge_cv.npz",
            predictions=ridge_predictions,
            targets=targets[splits["test"]],
            indices=splits["test"],
        )
    # end if predictions should be retained

    model_kwargs = {
        "n_layers": features.shape[1],
        "feature_dim": features.shape[2],
        "n_timepoints": targets.shape[1],
        "n_neurons": targets.shape[2],
        "hidden_dim": cfg.hidden_dim,
        "time_embedding_dim": cfg.time_embedding_dim,
        "attention_dim": cfg.attention_dim,
        "n_attention_heads": cfg.n_attention_heads,
        "dropout": cfg.dropout,
    }
    for architecture in architectures:
        for model_seed in model_seeds:
            set_seed(model_seed)
            loaders = {
                name: make_loader(
                    features,
                    targets,
                    indices,
                    cfg,
                    shuffle=name == "train",
                    seed=model_seed,
                )
                for name, indices in splits.items()
            }
            model = build_temporal_experiment_model(
                architecture,
                **model_kwargs,
            )
            parameters = sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
            run_start = time.perf_counter()
            history, best_epoch = train_torch_model(
                model, loaders, cfg, device
            )
            train_seconds = time.perf_counter() - run_start
            validation_predictions, validation_targets, _ = evaluate_torch_model(
                model, loaders["validation"], device
            )
            validation_mse = float(
                np.mean((validation_predictions - validation_targets) ** 2)
            )
            test_predictions, test_targets, mean_attention = evaluate_torch_model(
                model,
                loaders["test"],
                device,
                return_diagnostics=True,
            )
            metrics = prediction_metrics(
                test_predictions,
                test_targets,
                train_target_mean,
            )
            row = {
                "architecture": architecture,
                "seed": model_seed,
                "parameters": parameters,
                "best_epoch": best_epoch,
                "train_seconds": train_seconds,
                "validation_mse": validation_mse,
                **{f"test_{name}": value for name, value in metrics.items()},
            }
            result_rows.append(row)
            print(
                f"{architecture} seed {model_seed} | epoch {best_epoch:03d} | "
                f"val {validation_mse:.6f} | test {metrics['mse']:.6f} | "
                f"output r {metrics['mean_output_correlation']:.4f} | "
                f"{train_seconds:.1f}s"
            )
            run_name = f"{architecture}_seed{model_seed}"
            save_rows(output_dir / f"history_{run_name}.csv", history)
            if cfg.save_predictions:
                saved_arrays = {
                    "predictions": test_predictions,
                    "targets": test_targets,
                    "indices": splits["test"],
                }
                if mean_attention is not None:
                    saved_arrays["mean_attention"] = mean_attention
                # end if attention diagnostics are available
                np.savez_compressed(
                    output_dir / f"predictions_{run_name}.npz",
                    **saved_arrays,
                )
            # end if predictions should be retained
            if cfg.save_checkpoints:
                torch.save(
                    {"model_state": model.state_dict(), "model_kwargs": model_kwargs},
                    output_dir / f"checkpoint_{run_name}.pt",
                )
            # end if checkpoints should be retained
            save_rows(output_dir / "results.csv", result_rows)
        # end for model seed
    # end for architecture

    summaries = summarize_results(result_rows)
    save_rows(output_dir / "summary.csv", summaries)
    print("\nRanked by held-out MSE:")
    for summary in summaries:
        print(
            f"  {summary['architecture']:<30} "
            f"{summary['test_mse_mean']:.6f} +/- "
            f"{summary['test_mse_std']:.6f}"
        )
    # end for summary row
# EOF


if __name__ == "__main__":
    main()
# EOF
