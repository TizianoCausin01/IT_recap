import matplotlib.pyplot as plt
import numpy as np
import torch


"""
aggregate_attention_by_layer
Convert whole-layer or feature-level attention into comparable layer weights.

Feature-level weights are averaged over embedding coordinates and then
renormalized over layers. This is equivalent to each layer's total attention
mass, while retaining the interpretation of an average feature weight.

INPUT:
    - attention_weights: torch.Tensor -> [batch, time, layers] or
      [batch, time, layers, embedding]

OUTPUT:
    - layer_attention: torch.Tensor -> normalized [batch, time, layers] weights
"""
def aggregate_attention_by_layer(attention_weights):
    if attention_weights.ndim == 3:
        return attention_weights
    # end if attention already operates over whole layers
    if attention_weights.ndim != 4:
        raise ValueError(
            "attention_weights must have shape [batch, time, layers] or "
            "[batch, time, layers, embedding]."
        )
    # end if attention weights have an unsupported shape

    # Average the feature weights within each layer, then normalize the layer
    # axis so the displayed values sum to one at every image and time bin.
    mean_feature_attention = attention_weights.mean(dim=-1)
    normalization = mean_feature_attention.sum(dim=-1, keepdim=True)
    return mean_feature_attention / normalization.clamp_min(
        torch.finfo(mean_feature_attention.dtype).eps
    )
# EOF


"""
neural_activity_timebin_mse_loss
Compute aligned MSE independently at each time bin, then average bins.

INPUT:
    - predictions: torch.Tensor -> predicted activity [batch, time, neurons]
    - targets: torch.Tensor -> recorded activity [batch, time, neurons]

OUTPUT:
    - loss: torch.Tensor -> scalar mean of the independent time-bin MSEs
"""
def neural_activity_timebin_mse_loss(predictions, targets):
    if predictions.ndim != 3 or targets.ndim != 3:
        raise ValueError(
            "Predictions and targets must both have shape [batch, time, neurons]."
        )
    # end if either neural tensor does not have three axes
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Prediction shape {predictions.shape} does not match "
            f"target shape {targets.shape}."
        )
    # end if predictions and targets are misaligned

    # Error at t uses only prediction t and target t. Averaging over batch and
    # neurons gives one independent loss for every target time bin.
    squared_error = (predictions - targets).square()
    timebin_losses = squared_error.mean(dim=(0, 2))
    return timebin_losses.mean()
# EOF


"""
training_step
Train the neural regression model for one complete epoch.

INPUT:
    - net: torch.nn.Module -> neural prediction model
    - data_loader: DataLoader -> training inputs and neural targets
    - optimizer: torch.optim.Optimizer -> parameter optimizer
    - cost_function: callable -> scalar neural regression loss
    - use_precomputed_features: bool -> whether inputs bypass the image encoder
    - device: torch.device | str -> training device

OUTPUT:
    - mean_loss: float -> sample-weighted training MSE
"""
def training_step(
    net,
    data_loader,
    optimizer,
    cost_function,
    use_precomputed_features,
    device="cpu",
):
    samples = 0
    cumulative_loss = 0.0

    # Train the decoder; BaselineModel keeps its frozen backbone in eval mode.
    net.train()
    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Compute the aligned time-bin loss and update trainable parameters.
        predictions, _ = net(
            inputs,
            use_precomputed_features=use_precomputed_features,
        )
        loss = cost_function(predictions, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Weight every batch mean by its number of samples for the epoch mean.
        batch_samples = inputs.shape[0]
        samples += batch_samples
        cumulative_loss += loss.item() * batch_samples
    # end for training batch

    return cumulative_loss / samples
# EOF


"""
test_step
Evaluate neural regression loss over the complete validation set.

INPUT:
    - net: torch.nn.Module -> neural prediction model
    - data_loader: DataLoader -> validation inputs and neural targets
    - cost_function: callable -> scalar neural regression loss
    - use_precomputed_features: bool -> whether inputs bypass the image encoder
    - device: torch.device | str -> evaluation device

OUTPUT:
    - mean_loss: float -> sample-weighted validation MSE
"""
def test_step(
    net,
    data_loader,
    cost_function,
    use_precomputed_features,
    device="cpu",
):
    samples = 0
    cumulative_loss = 0.0
    net.eval()

    # Validation never builds gradients or updates model parameters.
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions, _ = net(
                inputs,
                use_precomputed_features=use_precomputed_features,
            )
            loss = cost_function(predictions, targets)

            batch_samples = inputs.shape[0]
            samples += batch_samples
            cumulative_loss += loss.item() * batch_samples
        # end for validation batch
    # end with no gradient tracking

    return cumulative_loss / samples
# EOF


"""
plot_mean_channel_reconstruction
Plot one validation sample's channel-averaged neural time course.

INPUT:
    - net: torch.nn.Module -> trained neural prediction model
    - validation_dataset: Dataset -> fixed validation subset
    - sample_index: int -> validation sample displayed across epochs
    - epoch: int -> current epoch shown in the plot title
    - use_precomputed_features: bool -> whether inputs bypass the image encoder
    - time_start_ms: float -> time represented by the first target sample
    - sampling_frequency: float -> target sampling frequency in Hz
    - device: torch.device | str -> evaluation device

OUTPUT:
    - None: displays the reconstruction plot
"""
def plot_mean_channel_reconstruction(
    net,
    validation_dataset,
    sample_index,
    epoch,
    use_precomputed_features,
    time_start_ms,
    sampling_frequency,
    device="cpu",
):
    if not 0 <= sample_index < len(validation_dataset):
        raise IndexError(
            f"Reconstruction sample {sample_index} is outside the "
            f"validation set of length {len(validation_dataset)}."
        )
    # end if sample_index is invalid

    # Reconstruct the same validation example without tracking gradients.
    model_input, target = validation_dataset[sample_index]
    net.eval()
    with torch.no_grad():
        prediction, _ = net(
            model_input.unsqueeze(0).to(device),
            use_precomputed_features=use_precomputed_features,
        )
    # end with no gradient tracking

    # Average across neural channels while preserving the temporal dimension.
    target_trace = target.mean(dim=-1).cpu().numpy()
    prediction_trace = prediction[0].mean(dim=-1).cpu().numpy()
    time_ms = (
        time_start_ms
        + np.arange(target_trace.shape[0]) * 1000.0 / sampling_frequency
    )

    fig, axis = plt.subplots(figsize=(8, 4))
    axis.plot(time_ms, target_trace, linewidth=2, label="Target")
    axis.plot(
        time_ms,
        prediction_trace,
        linewidth=2,
        label="Reconstruction",
    )
    axis.set_xlabel("Time (ms)")
    axis.set_ylabel("Mean neural activity across channels")
    axis.set_title(
        f"Validation sample {sample_index}, mean of "
        f"{target.shape[-1]} channels, epoch {epoch:03d}"
    )
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    plt.show()
# EOF


"""
collect_concatenated_layer_regression_data
Collect concatenated ANN-layer features and flattened neural targets.

INPUT:
    - net: BaselineModel -> model that resolves cached or online features
    - data_loader: DataLoader -> aligned inputs and neural targets
    - use_precomputed_features: bool -> whether inputs bypass the image encoder
    - device: torch.device | str -> feature-extraction device

OUTPUT:
    - features: np.ndarray -> samples by concatenated layer features
    - targets: np.ndarray -> samples by flattened time-neuron targets
"""
def collect_concatenated_layer_regression_data(
    net,
    data_loader,
    use_precomputed_features,
    device="cpu",
):
    feature_batches = []
    target_batches = []

    # Keep the frozen image backbone deterministic during feature collection.
    net.eval()
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            layer_features = net._resolve_layer_features(
                inputs,
                use_precomputed_features,
            )
            feature_batches.append(
                layer_features.flatten(start_dim=1).cpu().numpy()
            )
            target_batches.append(targets.flatten(start_dim=1).cpu().numpy())
        # end for data batch
    # end with no gradient tracking

    features = np.concatenate(feature_batches, axis=0)
    targets = np.concatenate(target_batches, axis=0)
    return features, targets
# EOF
