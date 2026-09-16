import numpy as np
import torch

from IT_recap.convrnn_analysis import rdm_vector
from IT_recap.neural_prediction_training import aggregate_trials_by_image
from IT_recap.static_drsa import prepare_rdm_for_rsa


"""
time_axis_ms
Create a millisecond time axis from a sampling frequency.

INPUT:
    - n_time: int -> number of timepoints
    - fs: float -> sampling frequency in Hz
    - start_ms: float -> first timepoint in ms

OUTPUT:
    - times_ms: np.ndarray -> time axis in ms
"""
def time_axis_ms(n_time, fs, start_ms=0.0):
    return start_ms + np.arange(n_time, dtype=float) * (1000.0 / fs)
# EOF


"""
rdm_time_series
Compute one condensed RDM per timepoint of an image-response time series.

INPUT:
    - activity: np.ndarray -> [images, time, channels] responses
    - rdm_metric: str -> RDM distance metric passed to rdm_vector

OUTPUT:
    - rdms: np.ndarray -> [time, image pairs] condensed RDM time series
"""
def rdm_time_series(activity, rdm_metric="correlation"):
    activity = np.asarray(activity, dtype=np.float64)
    if activity.ndim != 3:
        raise ValueError(
            "Expected [images, time, channels] activity, got "
            f"{activity.shape}."
        )
    # end if activity has the wrong rank

    # rdm_vector expects [features, images], hence the transpose per bin.
    rdms = [
        rdm_vector(activity[:, time_idx, :].T, metric=rdm_metric)
        for time_idx in range(activity.shape[1])
    ]
    return np.stack(rdms, axis=0)
# EOF


"""
prepared_rdm_time_series
Rank, center, and unit-normalize every RDM of a time series so that dynamic
RSA reduces to one matrix product between two prepared time series.

INPUT:
    - activity: np.ndarray -> [images, time, channels] responses
    - rdm_metric: str -> RDM distance metric
    - rsa_metric: str -> "spearman" or "pearson" RDM comparison metric

OUTPUT:
    - prepared: np.ndarray -> [time, image pairs] centered unit-norm RDMs
"""
def prepared_rdm_time_series(activity, rdm_metric="correlation", rsa_metric="spearman"):
    rdms = rdm_time_series(activity, rdm_metric=rdm_metric)
    return np.stack(
        [prepare_rdm_for_rsa(rdm, rsa_metric) for rdm in rdms],
        axis=0,
    )
# EOF


"""
drsa_from_prepared
Correlate every timepoint of one prepared RDM series with every timepoint of
another. Passing the same series twice gives the dRSA autocorrelation.

INPUT:
    - prepared_rows: np.ndarray -> [time_rows, image pairs] prepared RDMs
    - prepared_columns: np.ndarray -> [time_columns, image pairs] prepared RDMs

OUTPUT:
    - drsa: np.ndarray -> [time_rows, time_columns] RSA similarity matrix
"""
def drsa_from_prepared(prepared_rows, prepared_columns):
    if prepared_rows.shape[1] != prepared_columns.shape[1]:
        raise ValueError(
            "Prepared RDM series describe different image sets: "
            f"{prepared_rows.shape[1]} and {prepared_columns.shape[1]} pairs."
        )
    # end if the two RDM series are not comparable
    return (prepared_rows @ prepared_columns.T).astype(np.float32)
# EOF


"""
rdm_split_half_reliability
Estimate how reliably the image geometry is measured at every time bin. The
repeated presentations of each image are split in half, each half is averaged,
and the two RDM time series are correlated bin by bin. Spearman-Brown corrects
the split-half value up to the full repetition set, so the result is the
ceiling on any dRSA computed from these responses: a bin whose RDM is pure
noise cannot correlate with anything, including its own neighbours.

INPUT:
    - trial_values: np.ndarray -> [presentations, time, channels] responses
    - image_indices: np.ndarray -> image identifier of every presentation
    - rdm_metric: str -> RDM distance metric
    - rsa_metric: str -> RDM comparison metric
    - n_resamples: int -> number of random half-splits to average
    - seed: int -> random seed for the half-splits

OUTPUT:
    - reliability: np.ndarray -> [time] Spearman-Brown corrected RDM reliability
"""
def rdm_split_half_reliability(
    trial_values,
    image_indices,
    rdm_metric="correlation",
    rsa_metric="spearman",
    n_resamples=10,
    seed=0,
):
    trial_values = np.asarray(trial_values)
    unique_image_indices, compact_image_ids = np.unique(
        np.asarray(image_indices),
        return_inverse=True,
    )
    repetition_rows = [
        np.flatnonzero(compact_image_ids == image_id)
        for image_id in range(len(unique_image_indices))
    ]
    if min(len(rows) for rows in repetition_rows) < 2:
        raise ValueError(
            "Every image needs at least two presentations to split in half."
        )
    # end if an image cannot be split

    rng = np.random.default_rng(seed)
    half_reliabilities = []
    for _ in range(n_resamples):
        first_half = []
        second_half = []
        for rows in repetition_rows:
            shuffled_rows = rng.permutation(rows)
            split_point = len(shuffled_rows) // 2
            first_half.append(trial_values[shuffled_rows[:split_point]].mean(axis=0))
            second_half.append(trial_values[shuffled_rows[split_point:]].mean(axis=0))
        # end for image
        prepared_first = prepared_rdm_time_series(
            np.stack(first_half), rdm_metric, rsa_metric
        )
        prepared_second = prepared_rdm_time_series(
            np.stack(second_half), rdm_metric, rsa_metric
        )
        half_reliabilities.append(
            np.diagonal(drsa_from_prepared(prepared_first, prepared_second))
        )
    # end for half-split resample

    half_reliability = np.mean(half_reliabilities, axis=0)
    # Spearman-Brown holds only for a positive half-set correlation; a bin whose
    # halves disagree carries no measurable geometry at all.
    return np.where(
        half_reliability > 0.0,
        2.0 * half_reliability / (1.0 + half_reliability),
        0.0,
    )
# EOF


"""
off_diagonal_limit
Largest absolute off-diagonal value of a square dRSA matrix. Autocorrelation
diagonals are trivially one, so they must not set the color scale.

INPUT:
    - matrix: np.ndarray -> square dRSA matrix

OUTPUT:
    - limit: float -> symmetric color limit for the informative entries
"""
def off_diagonal_limit(matrix):
    off_diagonal = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    return float(np.nanmax(np.abs(off_diagonal)))
# EOF


"""
average_time_series_by_image
Collapse repeated presentations of the same image into one response per image
and return the responses in the order of the sorted image identifiers.

INPUT:
    - trial_values: np.ndarray -> [trials, time, channels] responses
    - image_indices: np.ndarray -> image identifier of every trial

OUTPUT:
    - image_values: np.ndarray -> [images, time, channels] mean responses
    - unique_image_indices: np.ndarray -> sorted image identifiers
"""
def average_time_series_by_image(trial_values, image_indices):
    unique_image_indices, compact_image_ids = np.unique(
        np.asarray(image_indices),
        return_inverse=True,
    )
    image_values = aggregate_trials_by_image(
        np.asarray(trial_values),
        compact_image_ids,
        n_images=len(unique_image_indices),
        reducer="mean",
    )
    return image_values, unique_image_indices
# EOF


"""
collect_gru_time_series
Run one ShowAttendTellGRUModel over a loader and keep the two time series the
dynamic RSA needs: the predicted neural population and the GRU hidden state.

INPUT:
    - net: ShowAttendTellGRUModel -> trained or freshly initialized model
    - data_loader: DataLoader -> aligned inputs and neural targets, unshuffled
    - use_precomputed_features: bool -> whether inputs bypass the image encoder
    - device: torch.device | str -> evaluation device

OUTPUT:
    - predictions: np.ndarray -> [trials, time, neurons] predicted activity
    - hidden_states: np.ndarray -> [trials, time, hidden_dim] recurrent states
    - targets: np.ndarray -> [trials, time, neurons] measured activity
"""
def collect_gru_time_series(
    net,
    data_loader,
    use_precomputed_features,
    device="cpu",
):
    prediction_batches = []
    hidden_batches = []
    target_batches = []

    net.eval()
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            predictions, _, hidden_states = net(
                inputs,
                use_precomputed_features=use_precomputed_features,
                return_hidden_states=True,
            )
            prediction_batches.append(predictions.cpu().numpy())
            hidden_batches.append(hidden_states.cpu().numpy())
            target_batches.append(targets.numpy())
        # end for data batch
    # end with no gradient tracking

    return (
        np.concatenate(prediction_batches, axis=0),
        np.concatenate(hidden_batches, axis=0),
        np.concatenate(target_batches, axis=0),
    )
# EOF


"""
plot_drsa_matrix
Draw one dynamic-RSA matrix with millisecond axes and the identity line.

INPUT:
    - ax: matplotlib.axes.Axes -> target axes
    - drsa: np.ndarray -> [time_rows, time_columns] RSA matrix
    - row_times_ms: np.ndarray -> vertical-axis time points in ms
    - column_times_ms: np.ndarray -> horizontal-axis time points in ms
    - title: str -> axes title
    - row_label: str -> vertical-axis label
    - column_label: str -> horizontal-axis label
    - vmin, vmax: float | None -> shared color limits across panels
    - cmap: str -> matplotlib colormap name

OUTPUT:
    - image: matplotlib.image.AxesImage -> handle for a shared colorbar
"""
def plot_drsa_matrix(
    ax,
    drsa,
    row_times_ms,
    column_times_ms,
    title,
    row_label="Neural time (ms)",
    column_label="Model time (ms)",
    vmin=None,
    vmax=None,
    cmap="RdBu_r",
):
    # imshow places the first row at the bottom so both axes read as real time.
    image = ax.imshow(
        drsa,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=(
            column_times_ms[0],
            column_times_ms[-1],
            row_times_ms[0],
            row_times_ms[-1],
        ),
    )
    # The identity line marks the temporally aligned RSA read off the diagonal.
    diagonal_start = max(column_times_ms[0], row_times_ms[0])
    diagonal_end = min(column_times_ms[-1], row_times_ms[-1])
    if diagonal_end > diagonal_start:
        ax.plot(
            [diagonal_start, diagonal_end],
            [diagonal_start, diagonal_end],
            color="black",
            linestyle="--",
            linewidth=0.8,
        )
    # end if the two time axes overlap
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(column_label)
    ax.set_ylabel(row_label)
    return image
# EOF
