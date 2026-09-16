"""
Training objectives, input augmentation, and weight averaging for the TVSD
time-bin decoders.

Every decoder in this project so far is trained under plain MSE on standardized
targets, and every one of them is then *reported* with stim_r, a correlation
across the 100 test images. Those two disagree: MSE on single-trial targets is
dominated by trial noise and is minimized by shrinking predictions toward the
mean, which costs image selectivity. The losses here let an experiment optimize
the quantity it is scored on, and the augmentations attack the other half of
the problem -- 22,248 training images, each seen once, is enough data to overfit
a spatial readout.
"""

import copy

import torch
from torch import nn
from torch.nn import functional as F


"""
correlation_loss
One minus the mean Pearson correlation across the batch, per (bin, site).

This is the training-time twin of stim_r: it is scale-free, so it cannot be
lowered by shrinking predictions, and it rewards getting the *ordering* of
images right at each site. It needs a batch wide enough for a stable
correlation -- a few hundred images is comfortable, a few dozen is not.

INPUT:
    - predictions: torch.Tensor -> [batch, time, neurons]
    - targets: torch.Tensor -> [batch, time, neurons]
    - epsilon: float -> guard against zero-variance outputs

OUTPUT:
    - loss: torch.Tensor -> scalar in [0, 2]
"""
def correlation_loss(predictions, targets, epsilon=1e-6):
    if predictions.shape[0] < 2:
        raise ValueError("A correlation loss needs at least two images.")
    # end if the batch cannot support a correlation
    centered_predictions = predictions - predictions.mean(dim=0, keepdim=True)
    centered_targets = targets - targets.mean(dim=0, keepdim=True)
    numerator = (centered_predictions * centered_targets).sum(dim=0)
    denominator = torch.sqrt(
        (centered_predictions**2).sum(dim=0) * (centered_targets**2).sum(dim=0)
        + epsilon
    )
    return 1.0 - (numerator / denominator).mean()
# EOF


"""
build_objective
Return the training loss named by an experiment configuration.

INPUT:
    - name: str -> "mse", "huber", "correlation", or "mse_correlation"
    - correlation_weight: float -> weight of the correlation term in the mix
    - huber_delta: float -> transition point of the Huber loss

OUTPUT:
    - objective: callable -> (predictions, targets) -> scalar loss
"""
def build_objective(name, correlation_weight=0.5, huber_delta=1.0):
    if name == "mse":
        return F.mse_loss
    if name == "huber":
        # Single-trial MUA has heavy tails; Huber stops a handful of large
        # residuals from owning the gradient.
        return lambda predictions, targets: F.huber_loss(
            predictions, targets, delta=huber_delta
        )
    if name == "correlation":
        return correlation_loss
    if name == "mse_correlation":
        return lambda predictions, targets: (
            (1.0 - correlation_weight) * F.mse_loss(predictions, targets)
            + correlation_weight * correlation_loss(predictions, targets)
        )
    raise KeyError(f"Unknown objective {name!r}.")
# EOF


"""
mixup_batch
Convex-combine a batch with a shuffled copy of itself.

Mixup is usually described for classification, but the neural-prediction case
is the easy one: the mapping from features to firing rate is close to linear in
the mixing coefficient, so a mixed input with the matching mixed target is an
almost-valid extra training example. It is the only augmentation here that
creates new (input, target) pairs rather than perturbing existing ones.

INPUT:
    - inputs: list[torch.Tensor] -> tensors to mix along the batch axis
    - targets: torch.Tensor -> [batch, ...] targets mixed with the same weight
    - alpha: float -> Beta(alpha, alpha) concentration; 0 disables mixing

OUTPUT:
    - mixed_inputs: list[torch.Tensor] -> the mixed inputs
    - mixed_targets: torch.Tensor -> the mixed targets
"""
def mixup_batch(inputs, targets, alpha):
    if alpha <= 0.0:
        return inputs, targets
    # end if mixing is disabled
    weight = float(
        torch.distributions.Beta(alpha, alpha).sample().clamp(0.0, 1.0)
    )
    permutation = torch.randperm(targets.shape[0], device=targets.device)
    mixed_inputs = [
        weight * tensor + (1.0 - weight) * tensor[permutation]
        if tensor.numel() > 0
        else tensor
        for tensor in inputs
    ]
    return mixed_inputs, weight * targets + (1.0 - weight) * targets[permutation]
# EOF


"""
random_translate_maps
Shift every feature map in the batch by its own random offset.

The factorized readout gives each site a fixed mask over the map, so it can
memorize exactly which cell of a 13x13 grid a training image put its evidence
in. Jittering the map by up to a cell or two forces the masks to describe a
receptive field with some width instead. Offsets are sub-pixel and bilinear,
and anything shifted in from outside is zero.

INPUT:
    - feature_maps: torch.Tensor -> [batch, channels, height, width]
    - max_shift: float -> largest offset, in map cells

OUTPUT:
    - shifted: torch.Tensor -> the translated maps
"""
def random_translate_maps(feature_maps, max_shift):
    if max_shift <= 0.0 or feature_maps.numel() == 0:
        return feature_maps
    # end if translation is disabled
    batch_size, _, height, width = feature_maps.shape
    # grid_sample works in normalized [-1, 1] coordinates, so a shift of one
    # cell is 2 / size of the axis.
    offsets = (torch.rand(batch_size, 2, device=feature_maps.device) * 2.0 - 1.0)
    offsets = offsets * max_shift
    offsets = offsets * torch.tensor(
        [2.0 / width, 2.0 / height], device=feature_maps.device
    )
    affine = torch.zeros(batch_size, 2, 3, device=feature_maps.device)
    affine[:, 0, 0] = 1.0
    affine[:, 1, 1] = 1.0
    affine[:, :, 2] = offsets
    grid = F.affine_grid(affine, feature_maps.shape, align_corners=False)
    return F.grid_sample(
        feature_maps, grid, padding_mode="zeros", align_corners=False
    )
# EOF


"""
drop_feature_groups
Zero whole cached depths at random, rescaling what survives.

The pooled branch reads three I-JEPA depths that are highly redundant, so it
can lean on one of them entirely. Dropping a whole depth is the group-level
version of dropout and makes every depth carry the prediction on its own.

INPUT:
    - layer_features: torch.Tensor -> [batch, layers, embedding]
    - probability: float -> chance of dropping each depth of each image

OUTPUT:
    - features: torch.Tensor -> the masked features
"""
def drop_feature_groups(layer_features, probability):
    if probability <= 0.0 or layer_features.numel() == 0:
        return layer_features
    # end if group dropout is disabled
    keep = (
        torch.rand(
            layer_features.shape[0], layer_features.shape[1], 1,
            device=layer_features.device,
        )
        > probability
    ).float()
    # Inverted dropout: keep the expected scale so evaluation needs no change.
    return layer_features * keep / max(1e-6, 1.0 - probability)
# EOF


"""
add_relative_noise
Perturb a tensor with Gaussian noise scaled by its own batch spread.

INPUT:
    - tensor: torch.Tensor -> tensor to perturb
    - noise_std: float -> noise scale, as a fraction of the tensor's own std

OUTPUT:
    - noisy: torch.Tensor -> the perturbed tensor
"""
def add_relative_noise(tensor, noise_std):
    if noise_std <= 0.0 or tensor.numel() == 0:
        return tensor
    # end if this noise source is disabled
    return tensor + torch.randn_like(tensor) * (noise_std * tensor.detach().std())
# EOF


class WeightAverage:
    """
    An exponential moving average of the weights, evaluated alongside the model.

    Single-trial targets make every gradient step noisy, so the last iterate
    bounces around the minimum it found. Averaging the trajectory is the
    cheapest way to stop paying for that bounce, and it costs one extra copy of
    a model that is a few million parameters at most.
    """

    """
    __init__
    Take the first snapshot of the tracked model.

    INPUT:
        - model: nn.Module -> the model being optimized
        - decay: float -> EMA decay per optimizer step

    OUTPUT:
        - None
    """
    def __init__(self, model, decay=0.999):
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must lie strictly between zero and one.")
        # end if the decay is invalid
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        # end for averaged parameter

    """
    update
    Fold one optimizer step into the average.

    INPUT:
        - model: nn.Module -> the model after the step

    OUTPUT:
        - None
    """
    @torch.no_grad()
    def update(self, model):
        averaged = dict(self.module.state_dict())
        for name, value in model.state_dict().items():
            if value.dtype.is_floating_point:
                averaged[name].mul_(self.decay).add_(value, alpha=1.0 - self.decay)
            else:
                # Buffers such as BatchNorm's step counter are copied, not mixed.
                averaged[name].copy_(value)
            # end if this entry can be averaged
        # end for tracked tensor
    # EOF
# EOC


"""
masked_prediction_loss
Score a multi-task prediction on its primary sites and its auxiliary sites.

INPUT:
    - predictions: torch.Tensor -> [batch, time, primary + auxiliary]
    - targets: torch.Tensor -> [batch, time, primary + auxiliary]
    - n_primary: int -> number of sites the experiment is scored on
    - objective: callable -> loss over (predictions, targets)
    - auxiliary_weight: float -> weight of the auxiliary term

OUTPUT:
    - loss: torch.Tensor -> scalar training loss
"""
def masked_prediction_loss(
    predictions, targets, n_primary, objective, auxiliary_weight
):
    loss = objective(predictions[..., :n_primary], targets[..., :n_primary])
    if predictions.shape[-1] > n_primary and auxiliary_weight > 0.0:
        loss = loss + auxiliary_weight * objective(
            predictions[..., n_primary:], targets[..., n_primary:]
        )
    # end if auxiliary sites are being predicted too
    return loss
# EOF


"""
fit_output_calibration
Fit one affine correction per (bin, site) on a held-out split.

A decoder trained on a correlation objective is free to predict on any scale,
and one trained on MSE deliberately shrinks toward the mean. Both leave raw MSE
saying something about prediction scale rather than about image selectivity.
Regressing the held-out target on the prediction removes exactly that, and
because the correction is affine within each (bin, site) it cannot change
stim_r at all -- it only makes the reported MSE comparable across objectives.

Fitting on the validation split keeps this honest: those are held-out images the
decoder was selected on, never the 100 test images it is scored on.

INPUT:
    - predictions: np.ndarray -> [presentations, time, sites] validation output
    - targets: np.ndarray -> [presentations, time, sites] validation response
    - epsilon: float -> guard against a constant prediction

OUTPUT:
    - slope: np.ndarray -> [time, sites] multiplicative correction
    - intercept: np.ndarray -> [time, sites] additive correction
"""
def fit_output_calibration(predictions, targets, epsilon=1e-8):
    import numpy as np

    prediction_mean = predictions.mean(axis=0)
    target_mean = targets.mean(axis=0)
    centered_predictions = predictions - prediction_mean
    covariance = (centered_predictions * (targets - target_mean)).mean(axis=0)
    variance = (centered_predictions**2).mean(axis=0)
    # A site the decoder never varies gets left alone at its own mean.
    slope = np.where(variance > epsilon, covariance / np.maximum(variance, epsilon), 0.0)
    return slope, target_mean - slope * prediction_mean
# EOF


"""
apply_output_calibration
Apply a fitted per-(bin, site) affine correction to new predictions.

INPUT:
    - predictions: np.ndarray -> [presentations, time, sites]
    - slope: np.ndarray -> [time, sites] multiplicative correction
    - intercept: np.ndarray -> [time, sites] additive correction

OUTPUT:
    - calibrated: np.ndarray -> the corrected predictions
"""
def apply_output_calibration(predictions, slope, intercept):
    return predictions * slope + intercept
# EOF
