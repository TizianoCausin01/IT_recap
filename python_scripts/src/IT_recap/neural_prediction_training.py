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
neural_activity_weighted_mse_loss
Average aligned neural-response errors using one fixed weight per time/channel.

The weights are normalized by their sum, so a uniform matrix reproduces the
ordinary elementwise MSE and changing the overall weight scale has no effect.

INPUT:
    - predictions: torch.Tensor -> predicted activity [batch, time, neurons]
    - targets: torch.Tensor -> recorded activity [batch, time, neurons]
    - weights: array-like -> non-negative importance weights [time, neurons]

OUTPUT:
    - loss: torch.Tensor -> scalar weighted mean-squared error
"""
def neural_activity_weighted_mse_loss(predictions, targets, weights):
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

    weights = torch.as_tensor(
        weights,
        device=predictions.device,
        dtype=predictions.dtype,
    ).detach()
    expected_shape = predictions.shape[1:]
    if weights.shape != expected_shape:
        raise ValueError(
            f"Weight shape {weights.shape} does not match "
            f"[time, neurons] shape {expected_shape}."
        )
    # end if weights do not align with output cells
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("weights must contain finite, non-negative values.")
    # end if weights cannot define a valid weighted mean

    weight_sum = weights.sum()
    if weight_sum <= 0:
        raise ValueError("At least one weight must be positive.")
    # end if every output cell was suppressed

    # First average each (time, neuron) error over images, then weight cells.
    cell_mse = (predictions - targets).square().mean(dim=0)
    return (cell_mse * weights).sum() / weight_sum
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
minimum_repetition_weighted_mse
Score each image by the lowest weighted MSE among its repeated presentations.

The time-neuron weights are applied within each trial first. The minimum is
then selected over whole-trial MSE values for one image, and image minima are
averaged so images with more repetitions do not receive more weight.

INPUT:
    - predictions: np.ndarray -> predicted activity [trials, time, neurons]
    - targets: np.ndarray -> recorded activity [trials, time, neurons]
    - image_ids: array-like -> source-image identifier for every trial
    - weights: array-like -> non-negative importance weights [time, neurons]

OUTPUT:
    - mean_minimum_mse: float -> mean minimum trial MSE across unique images
"""
def minimum_repetition_weighted_mse(
    predictions,
    targets,
    image_ids,
    weights,
):
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    image_ids = np.asarray(image_ids)
    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()
    # end if weights are stored as a tensor
    weights = np.asarray(weights, dtype=np.float64)

    if predictions.ndim != 3 or predictions.shape != targets.shape:
        raise ValueError(
            "predictions and targets must share shape [trials, time, neurons]."
        )
    # end if predictions and targets are misaligned
    if image_ids.ndim != 1 or len(image_ids) != len(predictions):
        raise ValueError("image_ids must provide one identifier per trial.")
    # end if image identifiers are misaligned
    if weights.shape != predictions.shape[1:]:
        raise ValueError(
            f"Weight shape {weights.shape} does not match "
            f"[time, neurons] shape {predictions.shape[1:]}."
        )
    # end if weights do not align with output cells
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("weights must contain finite, non-negative values.")
    # end if weights cannot define a valid weighted mean
    weight_sum = weights.sum()
    if weight_sum <= 0:
        raise ValueError("At least one weight must be positive.")
    # end if every output cell was suppressed

    # Reduce time and neurons within each complete trial before comparing
    # repetitions. This selects a real presentation, rather than assembling an
    # unrealistically favorable target from different repetitions cell by cell.
    trial_mse = (
        np.square(predictions - targets) * weights[None]
    ).sum(axis=(1, 2)) / weight_sum
    image_minimum_mse = [
        trial_mse[image_ids == image_id].min()
        for image_id in np.unique(image_ids)
    ]
    return float(np.mean(image_minimum_mse))
# EOF


"""
crossvalidated_neural_response_mse
Estimate an empirical MSE floor by predicting held-out repetitions with one
recorded neural response from the same image.

For every resample, repetitions of each image are divided into selection and
evaluation folds. The single selection-fold response with the lowest mean
weighted MSE to the other selection responses is retained. Its weighted MSE
to every evaluation-fold response is then averaged. Selection and evaluation
are disjoint, so the chosen response is never judged on the trials that chose
it. Images receive equal weight regardless of their repetition count.

The minimum over resamples matches the optimistic ceiling requested for the
minimum-repetition validation metric. The full resample distribution is also
returned because that minimum necessarily decreases as more resamples are
run; the mean is the less optimistic cross-validated reference.

INPUT:
    - trial_targets: np.ndarray -> responses [trials, time, neurons]
    - image_ids: array-like -> source-image identifier for every trial
    - weights: array-like -> non-negative importance weights [time, neurons]
    - n_resamples: int -> number of repeated selection/evaluation splits
    - evaluation_fraction: float -> fraction of repetitions held out per image
    - seed: int -> split reproducibility

OUTPUT:
    - result: dict -> resample MSE values and their minimum, mean, standard
      deviation, and 5th/95th percentiles
"""
def crossvalidated_neural_response_mse(
    trial_targets,
    image_ids,
    weights,
    n_resamples=200,
    evaluation_fraction=0.5,
    seed=0,
):
    trial_targets = np.asarray(trial_targets, dtype=np.float64)
    image_ids = np.asarray(image_ids)
    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()
    # end if weights are stored as a tensor
    weights = np.asarray(weights, dtype=np.float64)

    if trial_targets.ndim != 3:
        raise ValueError("trial_targets must have shape [trials, time, neurons].")
    # end if targets do not have the expected axes
    if image_ids.ndim != 1 or len(image_ids) != len(trial_targets):
        raise ValueError("image_ids must provide one identifier per trial.")
    # end if image identifiers are misaligned
    if weights.shape != trial_targets.shape[1:]:
        raise ValueError(
            f"Weight shape {weights.shape} does not match "
            f"[time, neurons] shape {trial_targets.shape[1:]}."
        )
    # end if weights do not align with output cells
    if not np.isfinite(trial_targets).all():
        raise ValueError("trial_targets must contain only finite values.")
    # end if a response cannot define an MSE
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("weights must contain finite, non-negative values.")
    # end if weights cannot define a valid weighted mean
    if weights.sum() <= 0:
        raise ValueError("At least one weight must be positive.")
    # end if every output cell was suppressed
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive.")
    # end if no cross-validation resamples were requested
    if not 0.0 < evaluation_fraction < 1.0:
        raise ValueError("evaluation_fraction must be between zero and one.")
    # end if the fold fraction is invalid

    rows_by_image = [
        np.flatnonzero(image_ids == image_id)
        for image_id in np.unique(image_ids)
    ]
    if min(len(rows) for rows in rows_by_image) < 3:
        raise ValueError(
            "Every image needs at least three repetitions: two for selecting "
            "one representative response and one for held-out evaluation."
        )
    # end if an image cannot support disjoint selection and evaluation

    # Trial-to-trial distances do not change between splits. Precomputing one
    # small matrix per image makes hundreds of resamples inexpensive.
    weight_sum = weights.sum()
    flattened_weight_scale = np.sqrt(weights.ravel() / weight_sum)
    pairwise_mse_by_image = []
    for rows in rows_by_image:
        scaled_targets = (
            trial_targets[rows].reshape(len(rows), -1)
            * flattened_weight_scale[None]
        )
        squared_norm = np.square(scaled_targets).sum(axis=1)
        pairwise_mse = (
            squared_norm[:, None]
            + squared_norm[None, :]
            - 2.0 * scaled_targets @ scaled_targets.T
        )
        pairwise_mse_by_image.append(np.maximum(pairwise_mse, 0.0))
    # end for repeated image

    rng = np.random.default_rng(seed)
    resample_mse = []
    for _ in range(n_resamples):
        image_mse = []
        for rows, pairwise_mse in zip(rows_by_image, pairwise_mse_by_image):
            shuffled_rows = rng.permutation(len(rows))
            n_evaluation = int(round(len(rows) * evaluation_fraction))
            n_evaluation = np.clip(n_evaluation, 1, len(rows) - 2)
            evaluation_rows = shuffled_rows[:n_evaluation]
            selection_rows = shuffled_rows[n_evaluation:]

            # Exclude each candidate's zero self-comparison before selecting
            # the response most representative of the selection fold.
            selection_mse = pairwise_mse[np.ix_(
                selection_rows, selection_rows
            )]
            candidate_mse = selection_mse.sum(axis=1) / (
                len(selection_rows) - 1
            )
            best_response_index = selection_rows[np.argmin(candidate_mse)]
            image_mse.append(
                pairwise_mse[evaluation_rows, best_response_index].mean()
            )
        # end for repeated image
        resample_mse.append(np.mean(image_mse))
    # end for cross-validation resample

    resample_mse = np.asarray(resample_mse)
    return {
        "resample_mse": resample_mse,
        "minimum_mse": float(resample_mse.min()),
        "mean_mse": float(resample_mse.mean()),
        "standard_deviation": float(resample_mse.std(ddof=1))
        if n_resamples > 1 else 0.0,
        "percentile_05": float(np.percentile(resample_mse, 5)),
        "percentile_95": float(np.percentile(resample_mse, 95)),
    }
# EOF


"""
minimum_repetition_test_step
Evaluate a neural regressor using the best-matching repetition of each image.

The validation loader must be deterministic: image_ids must follow its dataset
order exactly. Predictions and targets are collected over all batches before
the image groups are scored, so repetitions may cross batch boundaries.

INPUT:
    - net: torch.nn.Module -> neural prediction model
    - data_loader: DataLoader -> unshuffled repeated-trial validation loader
    - use_precomputed_features: bool -> whether inputs bypass the image encoder
    - image_ids: array-like -> source-image identifier for every loader sample
    - weights: array-like -> non-negative importance weights [time, neurons]
    - device: torch.device | str -> evaluation device

OUTPUT:
    - mean_loss: float -> mean minimum weighted trial MSE across images
"""
def minimum_repetition_test_step(
    net,
    data_loader,
    use_precomputed_features,
    image_ids,
    weights,
    device="cpu",
):
    prediction_batches = []
    target_batches = []
    net.eval()

    # Collect the complete validation pool because repetitions of one image can
    # occur in different batches.
    with torch.no_grad():
        for inputs, targets in data_loader:
            predictions, _ = net(
                inputs.to(device),
                use_precomputed_features=use_precomputed_features,
            )
            prediction_batches.append(predictions.cpu().numpy())
            target_batches.append(targets.numpy())
        # end for validation batch
    # end with no gradient tracking

    predictions = np.concatenate(prediction_batches, axis=0)
    targets = np.concatenate(target_batches, axis=0)
    return minimum_repetition_weighted_mse(
        predictions,
        targets,
        image_ids,
        weights,
    )
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


"""
stimulus_correlation
Pearson correlation across images, computed independently at every
(time, channel) pair. This is the "stim_r" used to score TVSD decoders.

INPUT:
    - predictions: np.ndarray -> [images, time, channels]
    - targets: np.ndarray -> [images, time, channels]

OUTPUT:
    - correlations: np.ndarray -> [time, channels], NaN where a series is flat
"""
def stimulus_correlation(predictions, targets):
    centered_predictions = predictions - predictions.mean(axis=0, keepdims=True)
    centered_targets = targets - targets.mean(axis=0, keepdims=True)
    denominator = np.sqrt(
        np.square(centered_predictions).sum(axis=0)
        * np.square(centered_targets).sum(axis=0)
    )
    return np.divide(
        (centered_predictions * centered_targets).sum(axis=0),
        denominator,
        out=np.full(denominator.shape, np.nan),
        where=denominator > 0,
    )
# EOF


"""
mean_stimulus_correlation
Average the finite across-image Pearson correlations over all time/channel
cells. Constant prediction or target cells are excluded rather than silently
treated as zero correlation.

INPUT:
    - predictions: np.ndarray -> [images, time, channels]
    - targets: np.ndarray -> [images, time, channels]

OUTPUT:
    - mean_correlation: float -> mean finite stim_r, or NaN if none are defined
"""
def mean_stimulus_correlation(predictions, targets):
    correlations = stimulus_correlation(predictions, targets)
    finite_correlations = correlations[np.isfinite(correlations)]
    if finite_correlations.size == 0:
        return float("nan")
    # end if no target cell defines a correlation
    return float(finite_correlations.mean())
# EOF


"""
aggregate_trials_by_image
Collapse repeated presentations of the same image into one response per image.

The reducer choice matters scientifically: "mean" is the usual higher-SNR
estimate of the evoked response, while "min" keeps the lower envelope over
repetitions and is therefore noisier and biased downward.

INPUT:
    - trial_values: np.ndarray -> [presentations, time, channels]
    - image_ids: np.ndarray -> zero-based image identifier per presentation
    - n_images: int -> number of distinct images
    - reducer: str -> "mean" or "min" over the repetitions of each image

OUTPUT:
    - image_values: np.ndarray -> [images, time, channels]
"""
def aggregate_trials_by_image(trial_values, image_ids, n_images, reducer="mean"):
    if reducer not in {"mean", "min"}:
        raise ValueError("reducer must be either 'mean' or 'min'.")
    # end if the repetition reducer is unsupported
    reduce_function = np.mean if reducer == "mean" else np.min

    image_values = []
    for image_id in range(n_images):
        repetition_rows = trial_values[image_ids == image_id]
        if len(repetition_rows) == 0:
            raise ValueError(f"Image {image_id} has no presentations.")
        # end if an image is missing from the evaluation pool
        image_values.append(reduce_function(repetition_rows, axis=0))
    # end for repeated-test image
    return np.stack(image_values)
# EOF


"""
split_half_reliability
Estimate the per-(time, channel) reliability of the repetition-aggregated
response by repeatedly splitting each image's repetitions into two halves.

With reducer "mean" the Spearman-Brown correction extrapolates the half-set
reliability to the full repetition count, and its square root is the usual
noise ceiling on stim_r. That correction assumes averaging, so it is not
applied for reducer "min": the returned values are then the raw reliability of
a half-sized minimum, i.e. a reference line rather than a true ceiling.

INPUT:
    - trial_targets: np.ndarray -> [presentations, time, channels]
    - image_ids: np.ndarray -> zero-based image identifier per presentation
    - n_images: int -> number of distinct images
    - reducer: str -> "mean" or "min" over the repetitions in each half
    - n_resamples: int -> random half-splits averaged over
    - seed: int -> split reproducibility

OUTPUT:
    - ceiling: np.ndarray -> [time, channels] stim_r reference values
"""
def split_half_reliability(
    trial_targets,
    image_ids,
    n_images,
    reducer="mean",
    n_resamples=40,
    seed=0,
):
    rng = np.random.default_rng(seed)
    rows_by_image = [
        np.flatnonzero(image_ids == image_id) for image_id in range(n_images)
    ]
    reduce_function = np.mean if reducer == "mean" else np.min

    half_correlations = []
    for _ in range(n_resamples):
        first_half, second_half = [], []
        for rows in rows_by_image:
            shuffled_rows = rng.permutation(rows)
            midpoint = len(shuffled_rows) // 2
            first_half.append(
                reduce_function(trial_targets[shuffled_rows[:midpoint]], axis=0)
            )
            second_half.append(
                reduce_function(trial_targets[shuffled_rows[midpoint:]], axis=0)
            )
        # end for repeated-test image
        half_correlations.append(
            stimulus_correlation(np.stack(first_half), np.stack(second_half))
        )
    # end for half-split resample

    half_reliability = np.nanmean(half_correlations, axis=0)
    if reducer != "mean":
        # No valid extrapolation from half to full repetitions for a minimum.
        return np.clip(half_reliability, 0.0, 1.0)
    # end if the reducer is not an average

    full_reliability = np.clip(
        2 * half_reliability / (1 + half_reliability), 0.0, 1.0
    )
    return np.sqrt(full_reliability)
# EOF


"""
optimally_rescaled_mse
Mean squared error after giving every (bin, channel) its best gain and offset.

Plain MSE on standardized targets rewards shrinkage: predictions pulled toward
the mean are penalized less by the trial noise they cannot explain, so a
heavily penalized ridge can beat a decoder that is more selective but more
dispersed. Rescaling each (bin, channel) prediction by the gain that minimizes
its error, and re-centring it on the target mean, removes exactly that
advantage and leaves the error that no rescaling can fix.

The gain is estimated on the same data it is applied to, which is what makes
this a diagnostic rather than a held-out score: read it against plain MSE to
see how much of a model's error is scale rather than shape.

INPUT:
    - predictions: np.ndarray -> [presentations, time, channels]
    - targets: np.ndarray -> [presentations, time, channels]

OUTPUT:
    - mse: float -> error remaining after the optimal per-cell rescaling
"""
def optimally_rescaled_mse(predictions, targets):
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must share a shape.")
    # end if the inputs disagree

    target_mean = targets.mean(axis=0, keepdims=True)
    centered_predictions = predictions - predictions.mean(axis=0, keepdims=True)
    centered_targets = targets - target_mean
    prediction_variance = (centered_predictions ** 2).mean(axis=0)
    covariance = (centered_predictions * centered_targets).mean(axis=0)
    # A constant prediction has no gain to fit; leaving it at one keeps it at
    # the target mean, which is the best a constant can do anyway.
    usable = prediction_variance > 1e-12
    optimal_scale = np.ones_like(prediction_variance)
    optimal_scale[usable] = covariance[usable] / prediction_variance[usable]
    rescaled = centered_predictions * optimal_scale + target_mean
    return float(np.mean((rescaled - targets) ** 2))
# EOF


"""
population_pattern_correlation
Correlate the predicted and observed population pattern within each image.

stim_r correlates across *images* for each (bin, site), so Pearson's own
centring removes each site's mean over images -- the term that says which sites
are generally responsive. This flips the axis: for one image and one bin it
correlates the predicted 320-site pattern against the observed one. Pearson
then removes that image's mean over sites, a single scalar, and leaves the
per-site offsets in place. Those offsets are identical for every image and are
usually far larger than the stimulus-driven deviation, so an uncentred score is
dominated by structure every model reproduces.

Passing center_across_images subtracts the mean over images from each (bin,
site) first, so only the stimulus-driven deviation is correlated. That is the
comparable quantity; the uncentred version is kept because the difference
between the two is the size of the nuisance term.

INPUT:
    - predictions: np.ndarray -> [presentations, time, channels]
    - targets: np.ndarray -> [presentations, time, channels]
    - center_across_images: bool -> remove each (bin, site) mean over images

OUTPUT:
    - correlations: np.ndarray -> [presentations, time], NaN where a pattern
      is constant and its correlation is undefined
"""
def population_pattern_correlation(
    predictions, targets, center_across_images=True
):
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if predictions.shape != targets.shape or predictions.ndim != 3:
        raise ValueError(
            "predictions and targets must share shape "
            "[presentations, time, channels]."
        )
    # end if the inputs disagree or have the wrong rank
    if center_across_images:
        predictions = predictions - predictions.mean(axis=0, keepdims=True)
        targets = targets - targets.mean(axis=0, keepdims=True)
    # end if the per-site offsets are removed first

    # Pearson's own centring, over the channel axis being correlated.
    centered_predictions = predictions - predictions.mean(axis=2, keepdims=True)
    centered_targets = targets - targets.mean(axis=2, keepdims=True)
    covariance = (centered_predictions * centered_targets).sum(axis=2)
    scale = np.sqrt(
        (centered_predictions ** 2).sum(axis=2)
        * (centered_targets ** 2).sum(axis=2)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        correlations = np.where(scale > 0, covariance / scale, np.nan)
    # end with suppressed warnings on constant patterns
    return correlations
# EOF


"""
disjoint_half_stimulus_correlation
stim_r with predictions and targets averaged over disjoint repetition halves.

A decoder that reads a presentation's own recorded activity can raise the
ordinary stim_r without carrying any extra image information: predictions and
targets are then averaged over the same repetitions, so noise shared between
the observed bins and the predicted bins of a trial survives into both averages
and correlates. Averaging the predictions over one half of an image's
repetitions and the targets over the other half closes that path, because no
trial contributes to both sides.

Half as many repetitions enter each average, so these values sit below the
ordinary stim_r for every model alike, an image-only model included. The metric
is meant for comparing models under it, not for reading off an absolute level;
passing the targets in as the predictions gives the matched reference, i.e. the
uncorrected split-half reliability.

INPUT:
    - trial_predictions: np.ndarray -> [presentations, time, channels]
    - trial_targets: np.ndarray -> [presentations, time, channels]
    - image_ids: np.ndarray -> zero-based image identifier per presentation
    - n_images: int -> number of distinct images
    - n_resamples: int -> random half-splits averaged over
    - seed: int -> split reproducibility

OUTPUT:
    - correlations: np.ndarray -> [time, channels] cross-half stim_r
"""
def disjoint_half_stimulus_correlation(
    trial_predictions,
    trial_targets,
    image_ids,
    n_images,
    n_resamples=40,
    seed=0,
):
    rng = np.random.default_rng(seed)
    rows_by_image = [
        np.flatnonzero(image_ids == image_id) for image_id in range(n_images)
    ]

    correlations = []
    for _ in range(n_resamples):
        halves = {"first": [[], []], "second": [[], []]}
        for rows in rows_by_image:
            shuffled_rows = rng.permutation(rows)
            midpoint = len(shuffled_rows) // 2
            for half_name, half_rows in (
                ("first", shuffled_rows[:midpoint]),
                ("second", shuffled_rows[midpoint:]),
            ):
                halves[half_name][0].append(
                    trial_predictions[half_rows].mean(axis=0)
                )
                halves[half_name][1].append(
                    trial_targets[half_rows].mean(axis=0)
                )
            # end for repetition half
        # end for repeated-test image
        # Both orientations are scored, so every repetition contributes to a
        # prediction average once and to a target average once.
        correlations.append(
            stimulus_correlation(
                np.stack(halves["first"][0]), np.stack(halves["second"][1])
            )
        )
        correlations.append(
            stimulus_correlation(
                np.stack(halves["second"][0]), np.stack(halves["first"][1])
            )
        )
    # end for half-split resample
    return np.nanmean(correlations, axis=0)
# EOF


"""
image_centered_mse_loss
MSE computed after removing each batch's mean response from both tensors.

The grand-mean PSTH is image-independent, so a plain MSE is dominated by
reproducing it. Centering the batch deletes that component from the objective
and leaves only the image-specific structure the decoder is meant to explain.
Amplitude still matters, unlike in the correlation loss.

INPUT:
    - predictions: torch.Tensor -> predicted activity [batch, time, neurons]
    - targets: torch.Tensor -> recorded activity [batch, time, neurons]

OUTPUT:
    - loss: torch.Tensor -> scalar MSE of the image-specific residuals
"""
def image_centered_mse_loss(predictions, targets):
    centered_predictions = predictions - predictions.mean(dim=0, keepdim=True)
    centered_targets = targets - targets.mean(dim=0, keepdim=True)
    return (centered_predictions - centered_targets).square().mean()
# EOF


"""
stimulus_correlation_loss
One minus the across-image Pearson correlation, per (time, neuron).

This is the training-time form of the stim_r evaluation metric. It is invariant
to any per-(time, neuron) affine rescaling of the predictions, so it removes
every incentive to spend capacity on output amplitude or on the shared PSTH and
rewards only correct ordering of images.

INPUT:
    - predictions: torch.Tensor -> predicted activity [batch, time, neurons]
    - targets: torch.Tensor -> recorded activity [batch, time, neurons]
    - eps: float -> denominator floor for flat batches

OUTPUT:
    - loss: torch.Tensor -> scalar in [0, 2], zero at perfect correlation
"""
def stimulus_correlation_loss(predictions, targets, eps=1e-6):
    if predictions.shape[0] < 2:
        raise ValueError(
            "The correlation loss needs at least two images per batch."
        )
    # end if the batch cannot define an across-image correlation

    centered_predictions = predictions - predictions.mean(dim=0, keepdim=True)
    centered_targets = targets - targets.mean(dim=0, keepdim=True)
    numerator = (centered_predictions * centered_targets).sum(dim=0)
    denominator = torch.sqrt(
        centered_predictions.square().sum(dim=0)
        * centered_targets.square().sum(dim=0)
    ).clamp_min(eps)
    return 1.0 - (numerator / denominator).mean()
# EOF


"""
response_spread_penalty
Squared gap between the across-image spread of predictions and a target spread.

This is the direct anti-flatness term: it pushes every (time, neuron) output to
vary across images as much as the recorded response does. Because the targets
supplied during training are single trials, their spread is dominated by trial
noise, so target_spread_fraction should be the single-trial reliability rather
than one -- otherwise the penalty asks the decoder to predict the noise.

INPUT:
    - predictions: torch.Tensor -> predicted activity [batch, time, neurons]
    - targets: torch.Tensor -> recorded activity [batch, time, neurons]
    - target_spread_fraction: float -> fraction of the target spread to match

OUTPUT:
    - penalty: torch.Tensor -> scalar, zero when the spreads agree
"""
def response_spread_penalty(predictions, targets, target_spread_fraction=1.0):
    prediction_spread = predictions.std(dim=0)
    target_spread = targets.std(dim=0) * target_spread_fraction
    return (prediction_spread - target_spread).square().mean()
# EOF


"""
make_neural_prediction_loss
Combine the aligned MSE with the flatness-related penalties.

Weights of zero drop their term entirely, so the default reproduces
neural_activity_timebin_mse_loss exactly.

INPUT:
    - mse_weight: float -> weight on the aligned time-bin MSE
    - centered_mse_weight: float -> weight on the image-centered MSE
    - correlation_weight: float -> weight on the across-image correlation loss
    - spread_weight: float -> weight on the across-image spread penalty
    - target_spread_fraction: float -> spread the penalty aims at

OUTPUT:
    - cost_function: callable -> (predictions, targets) -> scalar loss
"""
def make_neural_prediction_loss(
    mse_weight=1.0,
    centered_mse_weight=0.0,
    correlation_weight=0.0,
    spread_weight=0.0,
    target_spread_fraction=1.0,
):
    def cost_function(predictions, targets):
        loss = predictions.new_zeros(())
        if mse_weight != 0.0:
            loss = loss + mse_weight * neural_activity_timebin_mse_loss(
                predictions, targets
            )
        # end if the aligned MSE term is active
        if centered_mse_weight != 0.0:
            loss = loss + centered_mse_weight * image_centered_mse_loss(
                predictions, targets
            )
        # end if the image-centered MSE term is active
        if correlation_weight != 0.0:
            loss = loss + correlation_weight * stimulus_correlation_loss(
                predictions, targets
            )
        # end if the correlation term is active
        if spread_weight != 0.0:
            loss = loss + spread_weight * response_spread_penalty(
                predictions, targets, target_spread_fraction
            )
        # end if the spread penalty is active
        return loss
    # EOF
    return cost_function
# EOF


"""
collect_neural_predictions
Run a neural prediction model over one deterministic loader and collect its
predictions and targets in loader order.

INPUT:
    - net: torch.nn.Module -> neural prediction model
    - data_loader: DataLoader -> inputs and neural targets
    - use_precomputed_features: bool -> whether inputs bypass the image encoder
    - device: torch.device | str -> evaluation device

OUTPUT:
    - predictions: np.ndarray -> [samples, time, neurons] predictions
    - targets: np.ndarray -> [samples, time, neurons] recorded activity
"""
def collect_neural_predictions(
    net,
    data_loader,
    use_precomputed_features,
    device="cpu",
):
    prediction_batches = []
    target_batches = []

    net.eval()
    with torch.no_grad():
        for inputs, targets in data_loader:
            predictions, _ = net(
                inputs.to(device),
                use_precomputed_features=use_precomputed_features,
            )
            prediction_batches.append(predictions.cpu().numpy())
            target_batches.append(targets.numpy())
        # end for evaluation batch
    # end with no gradient tracking

    if not prediction_batches:
        raise ValueError("Cannot collect predictions from an empty loader.")
    # end if the evaluation loader is empty
    return (
        np.concatenate(prediction_batches, axis=0),
        np.concatenate(target_batches, axis=0),
    )
# EOF


"""
collect_image_level_predictions
Collect model outputs and optionally average repeated presentations so every
image contributes one prediction and one target in sorted image-ID order.

INPUT:
    - net: torch.nn.Module -> neural prediction model
    - data_loader: DataLoader -> deterministic inputs and neural targets
    - use_precomputed_features: bool -> whether inputs bypass the image encoder
    - image_ids: array-like | None -> image identifier per loader sample;
      None means that every sample is already one unique image
    - device: torch.device | str -> evaluation device

OUTPUT:
    - predictions: np.ndarray -> [images, time, neurons] predictions
    - targets: np.ndarray -> [images, time, neurons] recorded activity
"""
def collect_image_level_predictions(
    net,
    data_loader,
    use_precomputed_features,
    image_ids=None,
    device="cpu",
):
    predictions, targets = collect_neural_predictions(
        net,
        data_loader,
        use_precomputed_features,
        device=device,
    )
    if image_ids is None:
        return predictions, targets
    # end if every loader sample already represents one image

    image_ids = np.asarray(image_ids)
    if image_ids.ndim != 1 or len(image_ids) != len(predictions):
        raise ValueError(
            "image_ids must contain one identifier per loader sample."
        )
    # end if the image identities do not align with loader order
    _, compact_image_ids = np.unique(image_ids, return_inverse=True)
    n_images = int(compact_image_ids.max()) + 1
    predictions = aggregate_trials_by_image(
        predictions,
        compact_image_ids,
        n_images,
        reducer="mean",
    )
    targets = aggregate_trials_by_image(
        targets,
        compact_image_ids,
        n_images,
        reducer="mean",
    )
    return predictions, targets
# EOF


"""
evaluate_stimulus_correlation
Collect a whole subset and score it with one global across-image correlation.

Averaging per-batch correlations would understate the metric, so predictions
and targets are gathered first and correlated once over the full subset.

INPUT:
    - net: torch.nn.Module -> neural prediction model
    - data_loader: DataLoader -> inputs and neural targets
    - use_precomputed_features: bool -> whether inputs bypass the image encoder
    - device: torch.device | str -> evaluation device

OUTPUT:
    - correlations: np.ndarray -> [time, neurons] across-image Pearson r
    - mse: float -> aligned MSE over the same subset
"""
def evaluate_stimulus_correlation(
    net,
    data_loader,
    use_precomputed_features,
    device="cpu",
):
    predictions, targets = collect_neural_predictions(
        net,
        data_loader,
        use_precomputed_features,
        device=device,
    )
    mse = float(np.mean((predictions - targets) ** 2))
    return stimulus_correlation(predictions, targets), mse
# EOF
