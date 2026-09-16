from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from torch.utils.data import Dataset


class NeuralLayerFeatureDataset(Dataset):
    """
    Pair each neural presentation with its image's concatenated PCA layer target.

    INPUT (__getitem__):
        - index: int -> neural-presentation row

    OUTPUT:
        - neural_sequence: torch.Tensor -> [time, neural_channels]
        - layer_target: torch.Tensor -> concatenated PCA coordinates [sum(D_l)]
    """

    """
    __init__
    Align neural trials and static image targets through source image indices.

    INPUT:
        - neural_activity: np.ndarray -> [neurons, time, presentations]
        - image_targets: np.ndarray -> [source_images, sum(D_l)]
        - image_indices: np.ndarray -> source image index per presentation
        - channel_mean: np.ndarray -> train-only neural mean [1, 1, neurons]
        - channel_std: np.ndarray -> train-only neural std [1, 1, neurons]

    OUTPUT:
        - None
    """
    def __init__(
        self,
        neural_activity,
        image_targets,
        image_indices,
        channel_mean,
        channel_std,
    ):
        neural_activity = np.asarray(neural_activity)
        image_targets = np.asarray(image_targets)
        self.image_indices = np.asarray(image_indices, dtype=int)
        if neural_activity.ndim != 3:
            raise ValueError(
                "neural_activity must have shape [neurons, time, presentations]."
            )
        # end if neural activity has the wrong shape
        if image_targets.ndim != 2:
            raise ValueError("image_targets must have shape [images, features].")
        # end if image targets have the wrong shape
        if self.image_indices.shape != (neural_activity.shape[2],):
            raise ValueError(
                "image_indices must contain one source index per presentation."
            )
        # end if neural trials and image identities do not align
        if (
            self.image_indices.size == 0
            or self.image_indices.min() < 0
            or self.image_indices.max() >= len(image_targets)
        ):
            raise IndexError("image_indices are empty or exceed image_targets.")
        # end if source indices cannot address the target array

        # Convert to [presentations, time, neurons] before train-only scaling.
        neural_sequences = neural_activity.transpose(2, 1, 0).astype(
            np.float32,
            copy=False,
        )
        channel_mean = np.asarray(channel_mean, dtype=np.float32)
        channel_std = np.asarray(channel_std, dtype=np.float32)
        expected_shape = (1, 1, neural_sequences.shape[-1])
        standardizer_shapes_are_invalid = (
            channel_mean.shape != expected_shape
            or channel_std.shape != expected_shape
        )
        if standardizer_shapes_are_invalid:
            raise ValueError(
                f"channel_mean and channel_std must have shape {expected_shape}."
            )
        # end if standardization statistics do not broadcast by channel

        self.neural_sequences = torch.as_tensor(
            (neural_sequences - channel_mean) / channel_std,
            dtype=torch.float32,
        )
        self.image_targets = torch.as_tensor(image_targets, dtype=torch.float32)

    def __len__(self):
        return len(self.neural_sequences)
    # EOF

    def __getitem__(self, index):
        image_idx = self.image_indices[index]
        return self.neural_sequences[index], self.image_targets[image_idx]
    # EOF
# EOC


"""
fit_neural_channel_standardizer
Fit one mean and standard deviation per neural channel using training trials only.

INPUT:
    - neural_activity: np.ndarray -> [neurons, time, presentations]
    - training_trial_indices: array-like -> presentation indices in the train split

OUTPUT:
    - channel_mean: np.ndarray -> [1, 1, neurons]
    - channel_std: np.ndarray -> nonzero [1, 1, neurons]
"""
def fit_neural_channel_standardizer(neural_activity, training_trial_indices):
    neural_sequences = np.asarray(neural_activity).transpose(2, 1, 0)
    training_trial_indices = np.asarray(training_trial_indices, dtype=int)
    if training_trial_indices.ndim != 1 or training_trial_indices.size == 0:
        raise ValueError("training_trial_indices must be a non-empty vector.")
    # end if the training split is empty

    training_sequences = neural_sequences[training_trial_indices]
    channel_mean = training_sequences.mean(axis=(0, 1), keepdims=True)
    channel_std = training_sequences.std(axis=(0, 1), keepdims=True)
    channel_std = np.where(channel_std > 0.0, channel_std, 1.0)
    return channel_mean.astype(np.float32), channel_std.astype(np.float32)
# EOF


"""
fit_layer_pcas
Fit one train-only PCA per ANN layer and transform all source images.

Whitening gives every retained PCA coordinate unit training variance, preventing
high-variance layers or leading components from dominating the joint MSE.

INPUT:
    - layer_features: np.ndarray -> [source_images, layers, embedding]
    - training_image_indices: array-like -> unique source images in training
    - n_components: int -> maximum retained components for every layer
    - random_seed: int -> PCA randomized-SVD seed
    - whiten: bool -> whether PCA scores have unit training variance

OUTPUT:
    - transformed_targets: np.ndarray -> [source_images, layers * components]
    - pcas: list[PCA] -> fitted layer-specific inverse transforms
    - target_slices: list[slice] -> concatenated coordinates belonging to layers
"""
def fit_layer_pcas(
    layer_features,
    training_image_indices,
    n_components=64,
    random_seed=0,
    whiten=True,
):
    layer_features = np.asarray(layer_features)
    training_image_indices = np.unique(
        np.asarray(training_image_indices, dtype=int)
    )
    if layer_features.ndim != 3:
        raise ValueError(
            "layer_features must have shape [source_images, layers, embedding]."
        )
    # end if layer features have the wrong shape
    if n_components <= 0 or training_image_indices.size < 2:
        raise ValueError("PCA needs positive components and at least two images.")
    # end if PCA cannot be fitted

    maximum_components = min(
        int(n_components),
        len(training_image_indices) - 1,
        layer_features.shape[-1],
    )
    transformed_layers = []
    pcas = []
    target_slices = []
    target_start = 0
    for layer_idx in range(layer_features.shape[1]):
        pca = PCA(
            n_components=maximum_components,
            whiten=whiten,
            svd_solver="randomized",
            random_state=random_seed,
        )
        pca.fit(layer_features[training_image_indices, layer_idx])
        transformed_layer = pca.transform(layer_features[:, layer_idx])
        transformed_layers.append(transformed_layer.astype(np.float32))
        pcas.append(pca)
        target_slices.append(
            slice(target_start, target_start + maximum_components)
        )
        target_start += maximum_components
    # end for ANN layer
    return np.concatenate(transformed_layers, axis=1), pcas, target_slices
# EOF


"""
layer_balanced_mse_loss
Give every target layer equal influence regardless of its retained PCA width.

INPUT:
    - predictions: torch.Tensor -> concatenated predictions [batch, sum(D_l)]
    - targets: torch.Tensor -> concatenated PCA targets [batch, sum(D_l)]
    - target_slices: list[slice] -> coordinate range for each target layer

OUTPUT:
    - loss: torch.Tensor -> mean of per-layer elementwise MSE values
"""
def layer_balanced_mse_loss(predictions, targets, target_slices):
    if predictions.shape != targets.shape or predictions.ndim != 2:
        raise ValueError(
            "Predictions and targets must share shape [batch, sum(D_l)]."
        )
    # end if predictions and targets do not align
    layer_losses = [
        (predictions[:, target_slice] - targets[:, target_slice]).square().mean()
        for target_slice in target_slices
    ]
    if not layer_losses:
        raise ValueError("target_slices must contain at least one layer.")
    # end if no target layer was defined
    return torch.stack(layer_losses).mean()
# EOF


"""
train_feature_decoder_epoch
Train a neural-to-feature decoder for one complete epoch.

INPUT:
    - model: torch.nn.Module -> neural-to-layer-feature decoder
    - data_loader: DataLoader -> aligned neural sequences and PCA targets
    - optimizer: torch.optim.Optimizer -> parameter optimizer
    - target_slices: list[slice] -> concatenated target ranges
    - device: torch.device | str -> compute device

OUTPUT:
    - mean_loss: float -> sample-weighted training MSE
"""
def train_feature_decoder_epoch(
    model,
    data_loader,
    optimizer,
    target_slices,
    device="cpu",
):
    model.train()
    cumulative_loss = 0.0
    n_samples = 0
    for neural_sequences, targets in data_loader:
        neural_sequences = neural_sequences.to(device)
        targets = targets.to(device)
        predictions, _ = model(neural_sequences)
        loss = layer_balanced_mse_loss(predictions, targets, target_slices)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        batch_size = neural_sequences.shape[0]
        cumulative_loss += loss.item() * batch_size
        n_samples += batch_size
    # end for training batch
    return cumulative_loss / n_samples
# EOF


"""
evaluate_feature_decoder
Evaluate deterministic PCA-feature regression over one complete loader.

INPUT:
    - model: torch.nn.Module -> neural-to-layer-feature decoder
    - data_loader: DataLoader -> aligned neural sequences and PCA targets
    - target_slices: list[slice] -> concatenated target ranges
    - device: torch.device | str -> compute device

OUTPUT:
    - mean_loss: float -> sample-weighted validation MSE
"""
def evaluate_feature_decoder(
    model,
    data_loader,
    target_slices,
    device="cpu",
):
    model.eval()
    cumulative_loss = 0.0
    n_samples = 0
    with torch.no_grad():
        for neural_sequences, targets in data_loader:
            neural_sequences = neural_sequences.to(device)
            targets = targets.to(device)
            predictions, _ = model(neural_sequences)
            loss = layer_balanced_mse_loss(predictions, targets, target_slices)
            batch_size = neural_sequences.shape[0]
            cumulative_loss += loss.item() * batch_size
            n_samples += batch_size
        # end for validation batch
    # end with no gradient tracking
    return cumulative_loss / n_samples
# EOF


"""
fit_feature_decoder
Optimize with validation-based early stopping and restore the best parameters.

INPUT:
    - model: torch.nn.Module -> initialized neural-to-feature decoder
    - training_loader: DataLoader -> shuffled training data
    - validation_loader: DataLoader -> fixed validation data
    - target_slices: list[slice] -> concatenated layer ranges
    - learning_rate: float -> AdamW learning rate
    - weight_decay: float -> AdamW weight decay
    - max_epochs: int -> maximum number of complete epochs
    - patience: int -> epochs without improvement before stopping
    - device: torch.device | str -> compute device
    - epoch_callback: callable | None -> called as epoch_callback(model, epoch)
      before training (epoch 0) and after every epoch; a returned
      dict[str, float] is appended to history under the same keys

OUTPUT:
    - history: dict[str, list[float] | float | int] -> loss traces, callback
      metrics, and best epoch
"""
def fit_feature_decoder(
    model,
    training_loader,
    validation_loader,
    target_slices,
    learning_rate,
    weight_decay,
    max_epochs,
    patience,
    device="cpu",
    epoch_callback=None,
):
    if min(learning_rate, max_epochs, patience) <= 0 or weight_decay < 0:
        raise ValueError("Optimization settings must be positive and valid.")
    # end if optimization settings are invalid
    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    optimizer.zero_grad(set_to_none=True)
    initial_validation_loss = evaluate_feature_decoder(
        model,
        validation_loader,
        target_slices,
        device=device,
    )
    history = {
        "training_losses": [],
        "validation_losses": [initial_validation_loss],
        "best_epoch": 0,
        "best_validation_loss": initial_validation_loss,
    }
    best_state = deepcopy(model.state_dict())
    epochs_without_improvement = 0

    def record_epoch_metrics(epoch):
        # Optional per-epoch diagnostics, e.g. stim_r or reconstruction plots.
        if epoch_callback is None:
            return
        # end if no callback was requested
        metrics = epoch_callback(model, epoch) or {}
        for metric_name, metric_value in metrics.items():
            history.setdefault(metric_name, []).append(metric_value)
        # end for returned metric
    # EOF

    record_epoch_metrics(0)
    for epoch in range(1, max_epochs + 1):
        training_loss = train_feature_decoder_epoch(
            model,
            training_loader,
            optimizer,
            target_slices,
            device=device,
        )
        validation_loss = evaluate_feature_decoder(
            model,
            validation_loader,
            target_slices,
            device=device,
        )
        history["training_losses"].append(training_loss)
        history["validation_losses"].append(validation_loss)
        record_epoch_metrics(epoch)

        if validation_loss < history["best_validation_loss"]:
            history["best_epoch"] = epoch
            history["best_validation_loss"] = validation_loss
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        # end if validation improved
        if epochs_without_improvement >= patience:
            break
        # end if early stopping patience is exhausted
    # end for optimization epoch
    model.load_state_dict(best_state)
    return history
# EOF


"""
collect_feature_decoder_outputs
Collect predictions, targets, attention, and hidden sequences.

Evaluation-mode outputs are deterministic unless the model samples noise at
evaluation time. In that case, predictions and attention are averaged over
n_repeats independent draws; the GRU states themselves stay deterministic.

INPUT:
    - model: torch.nn.Module -> fitted neural-to-feature decoder
    - data_loader: DataLoader -> evaluation data in a fixed order
    - device: torch.device | str -> compute device
    - n_repeats: int -> forward passes averaged per batch

OUTPUT:
    - predictions: np.ndarray -> [samples, sum(D_l)]
    - targets: np.ndarray -> [samples, sum(D_l)]
    - attention: np.ndarray -> [samples, target_layers, attended_items]
    - hidden_states: np.ndarray -> [samples, time, hidden_dim]
"""
def collect_feature_decoder_outputs(model, data_loader, device="cpu", n_repeats=1):
    if n_repeats <= 0:
        raise ValueError("n_repeats must be positive.")
    # end if no forward pass was requested
    prediction_batches = []
    target_batches = []
    attention_batches = []
    hidden_batches = []
    model.eval()
    with torch.no_grad():
        for neural_sequences, targets in data_loader:
            neural_sequences = neural_sequences.to(device)
            repeat_predictions = []
            repeat_attention = []
            for _ in range(n_repeats):
                predictions, attention, hidden_states = model(
                    neural_sequences,
                    return_hidden_states=True,
                )
                repeat_predictions.append(predictions)
                repeat_attention.append(attention)
            # end for independent noise draw
            prediction_batches.append(
                torch.stack(repeat_predictions).mean(dim=0).cpu().numpy()
            )
            target_batches.append(targets.numpy())
            attention_batches.append(
                torch.stack(repeat_attention).mean(dim=0).cpu().numpy()
            )
            hidden_batches.append(hidden_states.cpu().numpy())
        # end for evaluation batch
    # end with no gradient tracking
    return (
        np.concatenate(prediction_batches),
        np.concatenate(target_batches),
        np.concatenate(attention_batches),
        np.concatenate(hidden_batches),
    )
# EOF


"""
average_rows_by_image
Average repeated presentation rows into one deterministic row per image identity.

INPUT:
    - values: np.ndarray -> [presentations, ...]
    - image_indices: np.ndarray -> image identity for every presentation

OUTPUT:
    - image_values: np.ndarray -> [unique_images, ...]
    - unique_images: np.ndarray -> sorted source image identities
"""
def average_rows_by_image(values, image_indices):
    values = np.asarray(values)
    image_indices = np.asarray(image_indices)
    if len(values) != len(image_indices):
        raise ValueError("values and image_indices must have the same rows.")
    # end if presentation metadata do not align
    unique_images = np.unique(image_indices)
    image_values = np.stack([
        values[image_indices == image_idx].mean(axis=0)
        for image_idx in unique_images
    ])
    return image_values, unique_images
# EOF


"""
score_layer_predictions
Score image-averaged PCA predictions separately for every target layer.

Repeated presentations are averaged by image first, so images with more
repetitions do not dominate. R2 and flattened r measure coordinate recovery;
the RDM Spearman rho asks whether the decoded image geometry is preserved.

INPUT:
    - predictions: np.ndarray -> [presentations, sum(D_l)]
    - targets: np.ndarray -> [presentations, sum(D_l)]
    - image_indices: np.ndarray -> image identity for every presentation
    - target_slices: list[slice] -> coordinate range for each target layer

OUTPUT:
    - scores: dict[str, np.ndarray] -> "r2", "feature_r", "rdm_rho", each [layers]
"""
def score_layer_predictions(predictions, targets, image_indices, target_slices):
    image_predictions, prediction_images = average_rows_by_image(
        predictions, image_indices
    )
    image_targets, target_images = average_rows_by_image(targets, image_indices)
    if not np.array_equal(prediction_images, target_images):
        raise RuntimeError("Prediction and target image orders differ.")
    # end if image-level rows are misaligned

    scores = {"r2": [], "feature_r": [], "rdm_rho": []}
    for target_slice in target_slices:
        predicted_layer = image_predictions[:, target_slice]
        target_layer = image_targets[:, target_slice]
        scores["r2"].append(
            r2_score(target_layer, predicted_layer, multioutput="variance_weighted")
        )
        scores["feature_r"].append(
            np.corrcoef(target_layer.ravel(), predicted_layer.ravel())[0, 1]
        )
        # Condensed correlation-distance RDMs over validation images.
        scores["rdm_rho"].append(spearmanr(
            pdist(target_layer, metric="correlation"),
            pdist(predicted_layer, metric="correlation"),
        ).statistic)
    # end for target layer
    return {name: np.asarray(values) for name, values in scores.items()}
# EOF


"""
plot_layer_feature_reconstruction
Plot target and predicted PCA coordinates for one sample and every target layer.

INPUT:
    - prediction: np.ndarray -> concatenated predicted coordinates [sum(D_l)]
    - target: np.ndarray -> concatenated target coordinates [sum(D_l)]
    - target_slices: list[slice] -> coordinate range for each target layer
    - layer_labels: list[str] -> concise labels in target order
    - title: str -> figure title

OUTPUT:
    - None: displays the reconstruction figure
"""
def plot_layer_feature_reconstruction(
    prediction,
    target,
    target_slices,
    layer_labels,
    title,
):
    fig, axes = plt.subplots(
        len(target_slices),
        1,
        figsize=(9, 2.5 * len(target_slices)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    for axis, target_slice, layer_label in zip(
        axes,
        target_slices,
        layer_labels,
    ):
        coordinate_indices = np.arange(target_slice.stop - target_slice.start)
        axis.plot(
            coordinate_indices,
            target[target_slice],
            marker="o",
            markersize=3,
            label="Target",
        )
        axis.plot(
            coordinate_indices,
            prediction[target_slice],
            marker="o",
            markersize=3,
            label="Prediction",
        )
        axis.set_ylabel(f"{layer_label}\nwhitened score")
        axis.grid(alpha=0.3)
    # end for target layer
    axes[0].legend()
    axes[-1].set_xlabel("PCA component")
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
# EOF
