"""
Consistency-weighted MSE for the TVSD time-bin decoders, and the MSE noise
ceiling of the dataset that loss is read against.

Every TVSD decoder so far minimizes an unweighted MSE over the (bin, site)
cells of the target, which spends the same gradient on a site whose response is
pure noise as on a site with a split-half reliability of 0.95. Roughly 5% of
the monkey F IT cells are effectively dead, and the loss cannot know that.
The weights here are the per-cell split-half reliability of the target itself,
so the objective concentrates on the part of the recording that carries
stimulus information at all.

How concentrated is a free parameter. ``consistency_weights`` maps the
reliability map through a softmax with temperature: at a high temperature every
cell is weighted alike and the loss is exactly the plain MSE, while at a low
one the objective collapses onto the few most consistent cells.
``effective_number_of_cells`` is the readable summary of that spikiness -- the
number of cells the loss is effectively fitting -- and is the quantity to
report next to a temperature, since the temperature itself is only meaningful
relative to the spread of the reliability map.

TVSD repeats only its 100 test images, so both the weights and the ceiling can
only be estimated there. The weights are therefore a property of the *sites*
measured on held-out images, not of the training images; they carry no
image-level information into training, but they are not blind to the test
recording either, which is the caveat any result from this module inherits.
"""

import numpy as np
import torch

from IT_recap.neural_prediction_training import (
    aggregate_trials_by_image,
    neural_activity_weighted_mse_loss,
)


# Temperature is a softmax scale, so "no weighting" is its infinite limit. The
# grid strings in the experiment scripts spell that limit "uniform".
UNIFORM_TEMPERATURE = float("inf")


"""
consistency_weights
Turn a per-cell reliability map into softmax weights at one temperature.

The softmax is taken over the flattened (bin, site) cells and returned
normalized to a mean of one, so a uniform map is the all-ones matrix and the
weighted loss reduces exactly to the plain MSE. Only the *ratios* between
weights matter to the loss, which normalizes by their sum.

INPUT:
    - reliability: np.ndarray -> [time, channels] split-half reliability in [0, 1]
    - temperature: float -> softmax scale; np.inf gives uniform weights

OUTPUT:
    - weights: np.ndarray -> [time, channels] non-negative weights, mean one
"""
def consistency_weights(reliability, temperature):
    reliability = np.asarray(reliability, dtype=np.float64)
    if reliability.ndim != 2:
        raise ValueError("reliability must be a [time, channels] map.")
    # end if the reliability map has the wrong rank
    if not np.isfinite(reliability).all():
        raise ValueError("reliability must be finite; NaN cells are undefined.")
    # end if the reliability map has undefined cells
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    # end if the softmax scale is invalid

    if np.isinf(temperature):
        return np.ones_like(reliability)
    # end if this is the unweighted limit

    # Subtracting the maximum keeps exp() finite at the small temperatures that
    # make the weighting spiky; it cancels in the normalization.
    logits = reliability / temperature
    weights = np.exp(logits - logits.max())
    return weights / weights.mean()
# EOF


"""
effective_number_of_cells
How many (bin, site) cells a weight map effectively fits.

This is the participation ratio of the normalized weights: it equals the cell
count for uniform weights and drops toward one as the loss concentrates on a
few cells, which makes it the interpretable axis for a temperature sweep.

INPUT:
    - weights: np.ndarray -> [time, channels] non-negative weights

OUTPUT:
    - n_effective: float -> effective number of weighted cells
"""
def effective_number_of_cells(weights):
    weights = np.asarray(weights, dtype=np.float64)
    normalized = weights / weights.sum()
    return float(1.0 / np.sum(normalized**2))
# EOF


"""
describe_weights
Summarize one weight map for the run table and the printed report.

INPUT:
    - weights: np.ndarray -> [time, channels] weights
    - reliability: np.ndarray -> [time, channels] reliability the weights came from

OUTPUT:
    - description: dict -> effective cell count and the weighted reliability
"""
def describe_weights(weights, reliability):
    n_cells = int(np.asarray(weights).size)
    n_effective = effective_number_of_cells(weights)
    return {
        "n_effective_cells": round(n_effective, 1),
        "effective_cell_fraction": round(n_effective / n_cells, 4),
        "weight_max_over_mean": round(
            float(np.max(weights) / np.mean(weights)), 3
        ),
        # The reliability the loss actually sees, i.e. the ceiling of the cells
        # it is spending its gradient on.
        "weighted_reliability": round(weighted_mean(reliability, weights), 4),
    }
# EOF


"""
weighted_mean
Weighted mean of a per-cell map, ignoring cells whose value is NaN.

INPUT:
    - values: np.ndarray -> [time, channels] per-cell quantity
    - weights: np.ndarray -> [time, channels] non-negative weights

OUTPUT:
    - mean: float -> weighted mean over the finite cells
"""
def weighted_mean(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.shape != weights.shape:
        raise ValueError(
            f"Value shape {values.shape} does not match weight shape "
            f"{weights.shape}."
        )
    # end if the two maps are misaligned
    finite = np.isfinite(values)
    weight_sum = weights[finite].sum()
    if weight_sum <= 0:
        raise ValueError("No finite cell carries a positive weight.")
    # end if nothing is left to average
    return float((values[finite] * weights[finite]).sum() / weight_sum)
# EOF


"""
make_weighted_mse_objective
Build the training loss that weights each (bin, site) by its consistency.

The returned callable has the (predictions, targets) signature the shared
training loop expects, and reduces to torch.nn.functional.mse_loss when the
weights are uniform.

INPUT:
    - weights: np.ndarray -> [time, channels] non-negative weights
    - device: torch.device -> device the training tensors live on

OUTPUT:
    - objective: callable -> (predictions, targets) -> scalar weighted MSE
"""
def make_weighted_mse_objective(weights, device):
    weight_tensor = torch.as_tensor(
        np.asarray(weights, dtype=np.float32), device=device
    )
    def objective(predictions, targets):
        # The weight tensor may have been built on the training device while a
        # validation pass runs elsewhere, so it follows the predictions.
        return neural_activity_weighted_mse_loss(
            predictions, targets, weight_tensor.to(predictions.device)
        )
    # EOF
    return objective
# EOF


"""
mse_noise_ceiling
The MSE a perfect model would still pay on the repetition-averaged target.

Scoring averages the repetitions of each test image, but that average is itself
noisy, so even a decoder that predicted the true mean response exactly would
not reach zero MSE. Two independent estimates of that floor are returned:

    - "repetitions": the within-image variance across repetitions divided by
      the repetition count, which is the sampling variance of the mean and
      needs no reliability model at all;
    - "reliability": var(image means) * (1 - r_full), where r_full is the
      Spearman-Brown corrected reliability of the average, i.e. the square of
      the stim_r ceiling. This is the same split the stim_r ceiling implies.

They rest on different assumptions -- the first on repetitions being
exchangeable, the second on the reliability estimate -- so agreement between
them is the evidence that the floor is real.

INPUT:
    - trial_targets: np.ndarray -> [presentations, time, channels] test targets
    - image_ids: np.ndarray -> zero-based image identifier per presentation
    - n_images: int -> number of repeated test images
    - ceiling: np.ndarray -> [time, channels] stim_r noise ceiling
    - null_prediction: np.ndarray -> [time, channels] predictor a model must beat

OUTPUT:
    - floors: dict -> the two [time, channels] floor maps, the null MSE map,
      the mean repetition count, and their unweighted means
"""
def mse_noise_ceiling(
    trial_targets, image_ids, n_images, ceiling, null_prediction
):
    image_ids = np.asarray(image_ids)
    repetition_counts = np.array(
        [int(np.sum(image_ids == image_id)) for image_id in range(n_images)]
    )
    if repetition_counts.min() < 2:
        raise ValueError("Every test image needs at least two repetitions.")
    # end if an image cannot support a within-image variance

    # Within-image variance per cell, pooled over images with the usual
    # unbiased estimator, then divided by the repetitions each mean averages.
    within_variances = np.stack(
        [
            trial_targets[image_ids == image_id].var(axis=0, ddof=1)
            for image_id in range(n_images)
        ]
    )
    repetition_floor = (
        within_variances / repetition_counts[:, None, None]
    ).mean(axis=0)

    image_targets = aggregate_trials_by_image(
        trial_targets, image_ids, n_images, reducer="mean"
    )
    image_variance = image_targets.var(axis=0)
    reliable_fraction = np.clip(ceiling, 0.0, 1.0) ** 2
    reliability_floor = image_variance * (1.0 - reliable_fraction)

    # What a model that ignores the image entirely pays, so the distance
    # between a decoder and the floor can be read as a fraction of the range.
    null_mse = ((null_prediction[None] - image_targets) ** 2).mean(axis=0)
    return {
        "repetition_floor": repetition_floor,
        "reliability_floor": reliability_floor,
        "null_mse": null_mse,
        "image_variance": image_variance,
        "mean_repetitions": float(repetition_counts.mean()),
        "mean_repetition_floor": float(repetition_floor.mean()),
        "mean_reliability_floor": float(reliability_floor.mean()),
        "mean_null_mse": float(null_mse.mean()),
    }
# EOF


"""
score_against_floor
Express one model's test MSE relative to the noise floor and the null model.

INPUT:
    - test_mse_map: np.ndarray -> [time, channels] per-cell test MSE
    - floors: dict -> output of mse_noise_ceiling
    - weights: np.ndarray -> [time, channels] weights of the loss being reported

OUTPUT:
    - row: dict -> plain and weighted MSE, the floors under those weights, and
      the fraction of the reducible MSE range the model closed
"""
def score_against_floor(test_mse_map, floors, weights):
    uniform = np.ones_like(test_mse_map)
    row = {}
    for label, weight_map in (("", uniform), ("weighted_", weights)):
        model_mse = weighted_mean(test_mse_map, weight_map)
        floor = weighted_mean(floors["repetition_floor"], weight_map)
        null_mse = weighted_mean(floors["null_mse"], weight_map)
        reducible = null_mse - floor
        row[f"{label}test_mse"] = round(model_mse, 5)
        row[f"{label}mse_floor"] = round(floor, 5)
        row[f"{label}mse_above_floor"] = round(model_mse - floor, 5)
        # 1.0 means the model reached the floor, 0.0 means it did no better
        # than predicting the training mean of every cell.
        row[f"{label}reducible_mse_explained"] = round(
            (null_mse - model_mse) / reducible if reducible > 0 else float("nan"),
            4,
        )
    # end for weighting of the report
    return row
# EOF


"""
per_cell_test_mse
Per-(bin, site) MSE of one model's repetition-averaged predictions.

INPUT:
    - trial_predictions: np.ndarray -> [presentations, time, channels]
    - trial_targets: np.ndarray -> [presentations, time, channels]
    - image_ids: np.ndarray -> zero-based image identifier per presentation
    - n_images: int -> number of repeated test images

OUTPUT:
    - mse_map: np.ndarray -> [time, channels] mean squared error over images
"""
def per_cell_test_mse(trial_predictions, trial_targets, image_ids, n_images):
    image_predictions = aggregate_trials_by_image(
        trial_predictions, image_ids, n_images, reducer="mean"
    )
    image_targets = aggregate_trials_by_image(
        trial_targets, image_ids, n_images, reducer="mean"
    )
    return ((image_predictions - image_targets) ** 2).mean(axis=0)
# EOF


"""
parse_temperature_grid
Read the comma-separated temperature grid of an experiment configuration.

"uniform" spells the infinite-temperature anchor, so a grid always carries the
plain-MSE reference next to the weighted settings it is judged against.

INPUT:
    - grid: str -> comma-separated temperatures, e.g. "uniform,0.5,0.1"

OUTPUT:
    - temperatures: list[float] -> parsed values, np.inf for "uniform"
"""
def parse_temperature_grid(grid):
    temperatures = []
    for entry in grid.split(","):
        entry = entry.strip()
        if not entry:
            continue
        # end if the grid has a trailing separator
        value = (
            UNIFORM_TEMPERATURE
            if entry.lower() in {"uniform", "inf", "none"}
            else float(entry)
        )
        if value <= 0.0:
            raise ValueError(f"Temperature {entry!r} must be positive.")
        # end if a grid entry is not a valid temperature
        temperatures.append(value)
    # end for grid entry
    if not temperatures:
        raise ValueError("The temperature grid is empty.")
    # end if nothing was requested
    return temperatures
# EOF


"""
temperature_label
Short, filename-safe name of one temperature setting.

INPUT:
    - temperature: float -> softmax scale, np.inf for the unweighted anchor

OUTPUT:
    - label: str -> "uniform" or e.g. "T0.10"
"""
def temperature_label(temperature):
    return "uniform" if np.isinf(temperature) else f"T{temperature:g}"
# EOF
