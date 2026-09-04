import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import ImageFolder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
with open(PROJECT_ROOT / "config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]

# Import the project implementation and the established neural-data utilities.
for source_path in (
    str((PROJECT_ROOT / "python_scripts" / "src").resolve()),
    str(Path(paths["useful_stuff_path"]).resolve()),
):
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    # end if source path is not registered
# end for source path

from IT_recap.hf_feature_extraction import load_hf_layer_features  # noqa: E402
from IT_recap.neural_prediction_training import (  # noqa: E402
    neural_activity_timebin_mse_loss,
    test_step,
    training_step,
)
from model_classes.temporal_models import BaselineModel  # noqa: E402
from project_specific_utils.dataloader import (  # noqa: E402
    load_img_natraster,
    map_image_order_from_ann_to_monkey,
)


@dataclass
class Cfg:
    # Data and cached DINO feature configuration.
    monkey_name: str = "three0"
    date: str = "250313"
    brain_area: str = "AIT"
    folder_name: str = "talia_20each_tizi"
    model_name: str = "dino_v3_l"
    img_size: int = 224
    pooling: str = "mean"
    layer_names: list[str] = field(default_factory=lambda: [
        "layer.3.mlp.down_proj",
        "layer.13.mlp.down_proj",
        "layer.16.mlp.down_proj",
        "layer.20.mlp.down_proj",
    ])
    original_fs: int = 1000
    new_fs: int = 100
    time_start_ms: float = 0.0
    time_end_ms: float = 300.0

    # Notebook-matched full-data experiment.
    validation_fraction: float = 0.2
    random_seed: int = 0
    batch_size: int = 64
    full_epochs: int = 150
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    temporal_embedding_dim: int = 128
    value_dim: int = 128
    mlp_hidden_dim: int = 32
    dropout: float = 0.5

    # Deliberate tiny-set memorization experiment.
    tiny_samples: int = 16
    tiny_epochs: int = 1000
    tiny_learning_rate: float = 1e-3
    print_interval: int = 25
    output_path: str = "results/baseline_plateau_diagnostic.npz"
# EOC


class CachedFeatureEncoder:
    """Minimal imgANN interface needed when every forward uses cached features."""

    def __init__(self, feature_dim):
        self.model = nn.Identity()
        self.feature_dim = feature_dim

    def set_relevant_layers(self, layer_names):
        self.layer_names = list(layer_names)

    def get_layer_output_shape(self, layer_name):
        if layer_name not in self.layer_names:
            raise KeyError(f"Unknown cached layer {layer_name!r}.")
        # end if layer is not configured
        return (None, self.feature_dim)

    def create_forward_hook(self):
        # Cached features bypass the encoder, so no hook is needed.
        return None
# EOC


"""
load_aligned_data
Load cached DINO activations and align them to the natraster stimulus order.

INPUT:
    - cfg: Cfg -> diagnostic data configuration

OUTPUT:
    - features: torch.Tensor -> aligned inputs [samples, layers, embedding]
    - targets: torch.Tensor -> neural responses [samples, time, neurons]
    - image_indices: np.ndarray -> source image identity for every sample
"""
def load_aligned_data(cfg):
    raster = load_img_natraster(
        paths=paths,
        monkey_name=cfg.monkey_name,
        date=cfg.date,
        original_fs=cfg.original_fs,
        new_fs=cfg.new_fs,
        time_start_ms=cfg.time_start_ms,
        time_end_ms=cfg.time_end_ms,
        brain_area=cfg.brain_area,
    ).get_array()

    # ImageFolder supplies exactly the ordering used during feature extraction.
    dataset_path = Path(paths["livingstone_lab"]) / "Stimuli" / cfg.folder_name
    image_dataset = ImageFolder(root=dataset_path, allow_empty=True)
    image_indices = np.asarray(
        map_image_order_from_ann_to_monkey(
            paths,
            cfg.monkey_name,
            cfg.date,
            image_dataset,
        ),
        dtype=int,
    )
    cached_features = load_hf_layer_features(
        output_dir=Path(paths["data_path"]) / "models",
        dataset_name=cfg.folder_name,
        model_name=cfg.model_name,
        img_size=cfg.img_size,
        layer_names=cfg.layer_names,
        pooling=cfg.pooling,
    )

    # Reorder cached image features and transpose neural samples to the first axis.
    features = torch.from_numpy(cached_features[image_indices]).float()
    targets = torch.from_numpy(raster.transpose(2, 1, 0)).float()
    if features.shape[0] != targets.shape[0]:
        raise ValueError(
            f"Found {features.shape[0]} inputs but {targets.shape[0]} targets."
        )
    # end if inputs and targets are misaligned
    return features, targets, image_indices
# EOF


"""
make_split
Reproduce the notebook's image-grouped train/validation split.

INPUT:
    - image_indices: np.ndarray -> source image identity for every sample
    - validation_fraction: float -> fraction of unique images held out
    - random_seed: int -> NumPy split seed

OUTPUT:
    - training_indices: np.ndarray -> training sample indices
    - validation_indices: np.ndarray -> validation sample indices
"""
def make_split(image_indices, validation_fraction, random_seed):
    unique_image_indices = np.unique(image_indices)
    n_validation_images = round(len(unique_image_indices) * validation_fraction)
    split_rng = np.random.default_rng(random_seed)
    shuffled_image_indices = split_rng.permutation(unique_image_indices)
    validation_images = shuffled_image_indices[:n_validation_images]
    is_validation = np.isin(image_indices, validation_images)
    return np.flatnonzero(~is_validation), np.flatnonzero(is_validation)
# EOF


"""
make_model
Construct the notebook's trainable decoder without loading the unused DINO model.

INPUT:
    - cfg: Cfg -> architecture configuration
    - feature_dim: int -> cached embedding width
    - n_timepoints: int -> neural target time bins
    - n_neurons: int -> neural target channels
    - dropout: float -> decoder dropout probability

OUTPUT:
    - model: BaselineModel -> feature-attention neural prediction model
"""
def make_model(cfg, feature_dim, n_timepoints, n_neurons, dropout):
    encoder = CachedFeatureEncoder(feature_dim)
    return BaselineModel(
        encoder,
        layers=cfg.layer_names,
        temporal_embedding_dim=cfg.temporal_embedding_dim,
        value_dim=cfg.value_dim,
        n_timepoints=n_timepoints,
        temporal_compression_ratio=1,
        n_neurons=n_neurons,
        mlp_hidden_dim=cfg.mlp_hidden_dim,
        dropout=dropout,
        attention_granularity="feature",
    )
# EOF


"""
run_full_diagnostic
Train the notebook-matched model and measure train and validation error using the
same end-of-epoch checkpoint with dropout disabled.

INPUT:
    - cfg: Cfg -> optimization configuration
    - training_dataset: TensorDataset -> full training split
    - validation_dataset: TensorDataset -> held-out split

OUTPUT:
    - history: dict[str, np.ndarray] -> online, eval-train, and validation losses
"""
def run_full_diagnostic(cfg, training_dataset, validation_dataset):
    training_generator = torch.Generator().manual_seed(cfg.random_seed)
    training_loader = DataLoader(
        training_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=training_generator,
    )
    # A separate ordered loader avoids advancing the training shuffle generator.
    training_evaluation_loader = DataLoader(
        training_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    inputs, targets = training_dataset.tensors
    torch.manual_seed(cfg.random_seed)
    model = make_model(
        cfg,
        feature_dim=inputs.shape[-1],
        n_timepoints=targets.shape[1],
        n_neurons=targets.shape[2],
        dropout=cfg.dropout,
    )
    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    history = {
        "online_train": [],
        "eval_train": [],
        "validation": [],
    }
    initial_train = test_step(
        model,
        training_evaluation_loader,
        neural_activity_timebin_mse_loss,
        True,
    )
    initial_validation = test_step(
        model,
        validation_loader,
        neural_activity_timebin_mse_loss,
        True,
    )
    print(
        f"full epoch 000 | eval-train {initial_train:.6f} | "
        f"validation {initial_validation:.6f}",
        flush=True,
    )

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, cfg.full_epochs + 1):
        online_train = training_step(
            model,
            training_loader,
            optimizer,
            neural_activity_timebin_mse_loss,
            True,
        )
        eval_train = test_step(
            model,
            training_evaluation_loader,
            neural_activity_timebin_mse_loss,
            True,
        )
        validation = test_step(
            model,
            validation_loader,
            neural_activity_timebin_mse_loss,
            True,
        )
        history["online_train"].append(online_train)
        history["eval_train"].append(eval_train)
        history["validation"].append(validation)

        if epoch == 1 or epoch % cfg.print_interval == 0:
            print(
                f"full epoch {epoch:03d} | online-train {online_train:.6f} | "
                f"eval-train {eval_train:.6f} | validation {validation:.6f}",
                flush=True,
            )
        # end if progress should be printed
    # end for epoch
    return {key: np.asarray(values) for key, values in history.items()}
# EOF


"""
run_tiny_memorization_test
Fit the same decoder to a tiny subset without dropout or weight decay.

INPUT:
    - cfg: Cfg -> memorization configuration
    - training_dataset: TensorDataset -> full training split

OUTPUT:
    - losses: np.ndarray -> eval-mode tiny-subset MSE after every epoch
"""
def run_tiny_memorization_test(cfg, training_dataset):
    tiny_inputs = training_dataset.tensors[0][:cfg.tiny_samples]
    tiny_targets = training_dataset.tensors[1][:cfg.tiny_samples]
    tiny_dataset = TensorDataset(tiny_inputs, tiny_targets)
    tiny_loader = DataLoader(
        tiny_dataset,
        batch_size=cfg.tiny_samples,
        shuffle=False,
    )

    torch.manual_seed(cfg.random_seed)
    model = make_model(
        cfg,
        feature_dim=tiny_inputs.shape[-1],
        n_timepoints=tiny_targets.shape[1],
        n_neurons=tiny_targets.shape[2],
        dropout=0.0,
    )
    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(),
        lr=cfg.tiny_learning_rate,
        weight_decay=0.0,
    )
    optimizer.zero_grad(set_to_none=True)

    losses = []
    for epoch in range(1, cfg.tiny_epochs + 1):
        training_step(
            model,
            tiny_loader,
            optimizer,
            neural_activity_timebin_mse_loss,
            True,
        )
        loss = test_step(
            model,
            tiny_loader,
            neural_activity_timebin_mse_loss,
            True,
        )
        losses.append(loss)
        if epoch == 1 or epoch % cfg.print_interval == 0:
            print(f"tiny epoch {epoch:04d} | eval-train {loss:.8f}", flush=True)
        # end if progress should be printed
    # end for epoch
    return np.asarray(losses)
# EOF


"""
run_linear_interpolation_control
Fit an effectively unregularized linear kernel interpolator to concatenated DINO
features and report whether those features can memorize the full training targets.

INPUT:
    - training_dataset: TensorDataset -> full training split
    - validation_dataset: TensorDataset -> held-out split

OUTPUT:
    - training_mse: float -> interpolator training MSE
    - validation_mse: float -> interpolator held-out MSE
"""
def run_linear_interpolation_control(training_dataset, validation_dataset):
    training_inputs, training_targets = training_dataset.tensors
    validation_inputs, validation_targets = validation_dataset.tensors
    x_train = training_inputs.flatten(start_dim=1).numpy().astype(np.float64)
    x_validation = validation_inputs.flatten(start_dim=1).numpy().astype(np.float64)
    y_train = training_targets.flatten(start_dim=1).numpy().astype(np.float64)
    y_validation = validation_targets.flatten(start_dim=1).numpy().astype(np.float64)

    # Center and scale coordinates using training statistics; retain an intercept
    # by centering the multi-output neural targets independently.
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_scale = x_train.std(axis=0, keepdims=True)
    x_scale[x_scale < 1e-8] = 1.0
    x_train = (x_train - x_mean) / x_scale
    x_validation = (x_validation - x_mean) / x_scale
    y_mean = y_train.mean(axis=0, keepdims=True)
    centered_targets = y_train - y_mean

    # The dual system is only 621 x 621 although the primal map has 4096 inputs.
    kernel_train = x_train @ x_train.T / x_train.shape[1]
    kernel_validation = x_validation @ x_train.T / x_train.shape[1]
    kernel_scale = np.trace(kernel_train) / len(kernel_train)
    jitter = kernel_scale * 1e-10
    dual_weights = np.linalg.solve(
        kernel_train + jitter * np.eye(len(kernel_train)),
        centered_targets,
    )
    training_predictions = kernel_train @ dual_weights + y_mean
    validation_predictions = kernel_validation @ dual_weights + y_mean
    training_mse = float(np.mean((training_predictions - y_train) ** 2))
    validation_mse = float(np.mean((validation_predictions - y_validation) ** 2))
    return training_mse, validation_mse
# EOF


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose plateauing in baseline_model_dev.ipynb."
    )
    parser.add_argument("--full_epochs", type=int, default=Cfg.full_epochs)
    parser.add_argument("--tiny_epochs", type=int, default=Cfg.tiny_epochs)
    parser.add_argument("--tiny_samples", type=int, default=Cfg.tiny_samples)
    parser.add_argument("--print_interval", type=int, default=Cfg.print_interval)
    parser.add_argument("--output_path", type=str, default=Cfg.output_path)
    return parser.parse_args()
# EOF


def main():
    args = parse_args()
    cfg = Cfg(
        full_epochs=args.full_epochs,
        tiny_epochs=args.tiny_epochs,
        tiny_samples=args.tiny_samples,
        print_interval=args.print_interval,
        output_path=args.output_path,
    )
    features, targets, image_indices = load_aligned_data(cfg)
    training_indices, validation_indices = make_split(
        image_indices,
        cfg.validation_fraction,
        cfg.random_seed,
    )
    training_dataset = TensorDataset(
        features[training_indices],
        targets[training_indices],
    )
    validation_dataset = TensorDataset(
        features[validation_indices],
        targets[validation_indices],
    )
    print(
        f"data | train {len(training_dataset)} | validation "
        f"{len(validation_dataset)} | input {tuple(features.shape[1:])} | "
        f"target {tuple(targets.shape[1:])}",
        flush=True,
    )

    # A time-neuron-specific training mean is the relevant constant predictor.
    training_mean = training_dataset.tensors[1].mean(dim=0, keepdim=True)
    constant_train_mse = float(
        (training_dataset.tensors[1] - training_mean).square().mean()
    )
    constant_validation_mse = float(
        (validation_dataset.tensors[1] - training_mean).square().mean()
    )
    print(
        f"constant baseline | train {constant_train_mse:.6f} | "
        f"validation {constant_validation_mse:.6f}",
        flush=True,
    )

    full_history = run_full_diagnostic(
        cfg,
        training_dataset,
        validation_dataset,
    )
    tiny_losses = run_tiny_memorization_test(cfg, training_dataset)
    linear_train_mse, linear_validation_mse = run_linear_interpolation_control(
        training_dataset,
        validation_dataset,
    )
    print(
        f"linear interpolator | train {linear_train_mse:.10f} | "
        f"validation {linear_validation_mse:.6f}",
        flush=True,
    )

    output_path = PROJECT_ROOT / cfg.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        full_online_train=full_history["online_train"],
        full_eval_train=full_history["eval_train"],
        full_validation=full_history["validation"],
        tiny_eval_train=tiny_losses,
        constant_train_mse=constant_train_mse,
        constant_validation_mse=constant_validation_mse,
        linear_train_mse=linear_train_mse,
        linear_validation_mse=linear_validation_mse,
    )
    print(f"saved {output_path}", flush=True)
# EOF


if __name__ == "__main__":
    main()
# EOF
