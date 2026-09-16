"""
Do anti-flatness penalties make the TVSD decoder discriminate images better?

A plain MSE decoder shrinks its per-image prediction toward the grand-mean PSTH
by exactly the correlation it achieves -- that is what minimizing MSE requires,
but it leaves the output dominated by the image-independent response. This
script tests three ways of pushing back, all on the same BaselineModel:

    spread     match the across-image standard deviation of the recorded
               response (the direct "not flat enough" penalty)
    centered   MSE on the image-centered residual, deleting the shared PSTH
               from the objective
    corr       one minus the across-image Pearson correlation, the training-time
               form of the stim_r evaluation metric

Because stim_r is invariant to any per-(time, channel) rescaling, the spread
penalty can only change stim_r by changing what is learned, not by inflating
the output. Reporting the calibration ratio alongside stim_r separates those.

Model selection uses validation stim_r for every experiment, so runs trained on
different objectives stay comparable.
"""

import argparse
import copy
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "python_scripts" / "src")
)

from IT_recap.neural_prediction_training import (  # noqa: E402
    aggregate_trials_by_image,
    evaluate_stimulus_correlation,
    make_neural_prediction_loss,
    split_half_reliability,
    stimulus_correlation,
    training_step,
)
from IT_recap.tvsd_experiments import (  # noqa: E402
    N_TEST_IMAGES,
    PROJECT_ROOT,
    build_datasets,
    build_loaders,
    build_variant_model,
    fit_ridge_reference,
    load_cached_data,
    load_project_paths,
    predict_test_trials,
    resolve_device,
    score_predictions,
    standardize_targets,
)


# Each experiment pairs a decoder with a training objective. "model" selects
# the decoder (default "baseline"); the remaining keys go straight to
# make_neural_prediction_loss, except "spread_scale", which multiplies the
# reliability-derived target spread the spread penalty aims at.
#
# The Gaussian-sphere pseudo-layer decoder is deliberately absent: it was
# rejected by its own attention (5% of the mass) and is no longer carried.
EXPERIMENTS = {
    # Objective sweep on the plain decoder, kept as the reference set.
    "mse": {},
    "spread_1": {"spread_weight": 1.0},
    "spread_10": {"spread_weight": 10.0},
    "spread_full": {"spread_weight": 10.0, "spread_scale": 5.5},
    "centered_mse": {"mse_weight": 0.0, "centered_mse_weight": 1.0},
    "corr_0.3": {"correlation_weight": 0.3},
    "corr_1": {"correlation_weight": 1.0},
    "corr_3": {"correlation_weight": 3.0},
    "corr_only": {"mse_weight": 0.0, "correlation_weight": 1.0},

    # MSE + Pearson correlation on the temporal-embedding-noise decoder.
    "tnoise_mse": {"model": "temporal_noise"},
    "tnoise_corr_0.3": {"model": "temporal_noise", "correlation_weight": 0.3},
    "tnoise_corr_1": {"model": "temporal_noise", "correlation_weight": 1.0},
    "tnoise_corr_3": {"model": "temporal_noise", "correlation_weight": 3.0},
    "tnoise_centered": {
        "model": "temporal_noise",
        "mse_weight": 0.0,
        "centered_mse_weight": 1.0,
    },
}

# Fixed categorical slots, assigned by entity so a curve keeps its colour.
EXPERIMENT_COLORS = {
    "ridge": "#2a78d6",
    "mse": "#eb6834",
    "spread_1": "#1baf7a",
    "spread_10": "#4a3aa7",
    "spread_full": "#e87ba4",
    "centered_mse": "#eda100",
    "corr_0.3": "#008300",
    "corr_1": "#e34948",
    "corr_3": "#87919c",
    "corr_only": "#0e7c86",
    "tnoise_mse": "#eb6834",
    "tnoise_corr_0.3": "#1baf7a",
    "tnoise_corr_1": "#4a3aa7",
    "tnoise_corr_3": "#e34948",
    "tnoise_centered": "#eda100",
}


@dataclass
class Cfg:
    # Caches produced by train_tvsd_baseline_marimo.py; nothing is recomputed.
    env: str | None = None
    mua_file_name: str = "f_THINGS_MUA_trials.mat"
    feature_archive_name: str = "tvsd_monkeyF_dino_v3_l_224_features.npz"
    output_dir: str | None = None

    area: str = "IT"
    target_fs: int = 100
    time_start_ms: float = 0.0
    time_end_ms: float = 200.0
    response_onset_ms: float = 50.0

    layer_names: list[str] = field(
        default_factory=lambda: [
            "layer.3.mlp.down_proj",
            "layer.13.mlp.down_proj",
            "layer.16.mlp.down_proj",
            "layer.20.mlp.down_proj",
        ]
    )

    # Optimization, identical to the noise-regularization comparison.
    validation_fraction: float = 0.1
    random_seed: int = 0
    batch_size: int = 64
    num_workers: int = 0
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    temporal_embedding_dim: int = 128
    value_dim: int = 128
    mlp_hidden_dim: int = 64
    dropout: float = 0.2
    attention_granularity: str = "layer"

    # Temporal-embedding jitter, used by the tnoise_* experiments.
    temporal_noise_std: float = 0.25
    relative_temporal_noise: bool = True
    temporal_noise_in_eval: bool = False
    match_noise_norm: bool = True
    noise_layer_in_eval: bool = True

    noise_ceiling_resamples: int = 40
    experiments: str = ",".join(EXPERIMENTS)
    model_seeds: str = "0"
    device: str = "auto"
    smoke_test: bool = False


"""
parse_args
Parse command-line overrides into a configuration object.

OUTPUT:
    - cfg: Cfg -> data, split, and optimization settings
"""
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for field_name, field_definition in Cfg.__dataclass_fields__.items():
        if field_name == "layer_names":
            continue
        # end if the layer list is fixed by the cached feature archive
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
            parser.add_argument(
                argument_name, type=type(default), default=default
            )
        # end if boolean, optional string, or typed argument
    # end for configuration field
    return Cfg(**vars(parser.parse_args()))
# EOF


"""
single_trial_spread_fraction
Convert the 30-repetition noise ceiling into the signal fraction of a single
trial's across-image spread.

The ceiling reported for this pool is stim_r, so the reliability of the 30-rep
mean is its square. Inverting Spearman-Brown gives the single-trial
reliability, whose square root is the fraction of a single trial's across-image
standard deviation that is actually stimulus driven. That is the spread an
honest anti-flatness penalty should ask a decoder to reproduce.

INPUT:
    - ceiling_stim_r: float -> mean split-half ceiling over the response window
    - n_repetitions: int -> repetitions behind the ceiling estimate

OUTPUT:
    - spread_fraction: float -> stimulus-driven share of single-trial spread
"""
def single_trial_spread_fraction(ceiling_stim_r, n_repetitions=30):
    mean_reliability = float(np.clip(ceiling_stim_r, 0.0, 0.999) ** 2)
    single_trial_reliability = mean_reliability / (
        n_repetitions - (n_repetitions - 1) * mean_reliability
    )
    return float(np.sqrt(np.clip(single_trial_reliability, 0.0, 1.0)))
# EOF


"""
train_with_objective
Optimize one objective and keep the epoch with the best validation stim_r.

Selecting on correlation rather than MSE is what makes objectives comparable:
a correlation-trained decoder has no reason to produce a well-scaled output, so
its validation MSE is not a meaningful selection signal.

INPUT:
    - model: BaselineModel -> decoder to optimize
    - loaders: dict -> train and validation loaders
    - cost_function: callable -> training objective
    - cfg: Cfg -> optimization settings
    - response_slice: slice -> time bins counted as driven response
    - device: torch.device -> compute device

OUTPUT:
    - history: dict -> per-epoch training loss, validation MSE, validation r
    - best_epoch: int -> selected checkpoint epoch
    - best_validation_r: float -> validation stim_r of the restored checkpoint
"""
def train_with_objective(
    model, loaders, cost_function, cfg, response_slice, device
):
    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    optimizer.zero_grad(set_to_none=True)

    validation_correlations, validation_mse = evaluate_stimulus_correlation(
        model, loaders["validation"], True, device
    )
    best_validation_r = float(
        np.nanmean(validation_correlations[response_slice])
    )
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    history = {
        "initial_validation_r": best_validation_r,
        "initial_validation_mse": validation_mse,
        "train_loss": [],
        "validation_mse": [],
        "validation_r": [],
    }

    for epoch in range(1, cfg.epochs + 1):
        training_loss = training_step(
            model,
            loaders["train"],
            optimizer,
            cost_function,
            use_precomputed_features=True,
            device=device,
        )
        validation_correlations, validation_mse = (
            evaluate_stimulus_correlation(
                model, loaders["validation"], True, device
            )
        )
        validation_r = float(
            np.nanmean(validation_correlations[response_slice])
        )
        history["train_loss"].append(training_loss)
        history["validation_mse"].append(validation_mse)
        history["validation_r"].append(validation_r)

        if validation_r > best_validation_r:
            best_validation_r = validation_r
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        # end if this epoch is the best validation checkpoint
        print(
            f"    epoch {epoch:03d}/{cfg.epochs:03d} | loss {training_loss:.6f}"
            f" | val MSE {validation_mse:.6f} | val r {validation_r:.4f}"
        )
    # end for optimization epoch

    model.load_state_dict(best_state)
    return history, best_epoch, best_validation_r
# EOF


"""
flatness_diagnostics
Measure how much of the output actually varies with the image.

INPUT:
    - image_predictions: np.ndarray -> [images, time, channels] predictions
    - image_targets: np.ndarray -> [images, time, channels] targets
    - stim_r: float -> mean across-image correlation over the response window
    - response_slice: slice -> time bins counted as driven response

OUTPUT:
    - diagnostics: dict -> spread ratio, calibration, image-specific share
"""
def flatness_diagnostics(
    image_predictions, image_targets, stim_r, response_slice
):
    prediction_window = image_predictions[:, response_slice]
    target_window = image_targets[:, response_slice]

    spread_ratio = float(
        prediction_window.std(axis=0).mean()
        / target_window.std(axis=0).mean()
    )
    # An MSE-optimal predictor satisfies spread_ratio == stim_r exactly, so the
    # ratio of the two is one when the shrinkage is exactly right.
    calibration = float(spread_ratio / stim_r) if stim_r > 0 else float("nan")

    image_specific = float(prediction_window.var(axis=0).mean())
    common_psth = float(prediction_window.mean(axis=0).var())
    return {
        "spread_ratio": round(spread_ratio, 4),
        "calibration": round(calibration, 3),
        "image_specific_share": round(
            image_specific / (image_specific + common_psth), 3
        ),
    }
# EOF


"""
plot_flatness_sweep
Plot validation stim_r per epoch and the test stim_r time courses.

INPUT:
    - cfg: Cfg -> time-axis settings
    - histories: dict -> per-experiment training histories
    - correlations: dict -> per-experiment [time, channels] test stim_r
    - ceiling: np.ndarray -> [time, channels] noise ceiling
    - result_rows: list[dict] -> scored experiments, for the scatter panel
    - output_path: Path -> destination figure file

OUTPUT:
    - None: writes the figure to disk
"""
def plot_flatness_sweep(
    cfg, histories, correlations, ceiling, result_rows, output_path
):
    response_time_ms = np.arange(
        cfg.time_start_ms, cfg.time_end_ms, 1000 / cfg.target_fs
    )
    figure, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    for name, history in histories.items():
        curve = [history["initial_validation_r"], *history["validation_r"]]
        axes[0].plot(
            np.arange(len(curve)),
            curve,
            linewidth=2,
            color=EXPERIMENT_COLORS[name],
            label=name,
        )
    # end for training objective
    axes[0].set(
        xlabel="Epoch",
        ylabel="Validation stim_r (response window)",
        title="Model selection signal",
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    for name, channel_time_correlations in correlations.items():
        axes[1].plot(
            response_time_ms,
            np.nanmedian(channel_time_correlations, axis=1),
            linewidth=2,
            color=EXPERIMENT_COLORS[name],
            label=name,
        )
    # end for scored experiment
    axes[1].plot(
        response_time_ms,
        np.nanmedian(ceiling, axis=1),
        color="black",
        linestyle=":",
        linewidth=2,
        label="Noise ceiling",
    )
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].axvline(
        cfg.response_onset_ms, color="gray", linewidth=1, linestyle="--"
    )
    axes[1].set(
        xlabel="Time from image onset (ms)",
        ylabel="Median stim_r across MUA sites",
        title="100 test images, mean over 30 repetitions",
    )
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    # The decisive panel: did fixing the flatness buy any discrimination?
    for row in result_rows:
        axes[2].scatter(
            row["calibration"],
            row["mean_stim_r_response"],
            s=110,
            color=EXPERIMENT_COLORS[row["experiment"]],
            zorder=3,
            label=row["experiment"],
        )
    # end for scored experiment
    axes[2].axvline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=1,
        label="MSE-optimal shrinkage",
    )
    axes[2].set(
        xlabel="Calibration (output spread / MSE-optimal spread)",
        ylabel="Test stim_r (response window)",
        title="Flatness fixed vs. discrimination gained",
    )
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
# EOF


def main():
    cfg = parse_args()
    if cfg.smoke_test:
        cfg.epochs = min(cfg.epochs, 2)
        cfg.noise_ceiling_resamples = 4
    # end if a smoke run was requested

    paths = load_project_paths(cfg)
    device = resolve_device(cfg.device)
    output_dir = Path(
        cfg.output_dir or PROJECT_ROOT / "results" / "tvsd_flatness_penalty"
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    targets, train_features, test_features, allmat = load_cached_data(cfg, paths)
    datasets, indices, (channel_mean, channel_scale) = build_datasets(
        cfg, targets, train_features, test_features, allmat
    )
    loaders = build_loaders(cfg, datasets)
    n_timepoints, n_neurons = targets.shape[1], targets.shape[2]

    fit_targets = standardize_targets(
        targets[indices["train"]], channel_mean, channel_scale
    )
    validation_targets = standardize_targets(
        targets[indices["validation"]], channel_mean, channel_scale
    )
    test_trial_targets = standardize_targets(
        targets[indices["test"]], channel_mean, channel_scale
    )
    test_image_ids = allmat[indices["test"], 2] - 1
    fit_target_mean = fit_targets.mean(axis=0)
    response_slice = slice(
        int(round(cfg.response_onset_ms / (1000 / cfg.target_fs))), n_timepoints
    )

    ceiling = split_half_reliability(
        test_trial_targets,
        test_image_ids,
        N_TEST_IMAGES,
        reducer="mean",
        n_resamples=cfg.noise_ceiling_resamples,
        seed=cfg.random_seed,
    )
    ceiling_stim_r = float(np.nanmean(ceiling[response_slice]))
    spread_fraction = single_trial_spread_fraction(ceiling_stim_r)
    print(
        f"device {device} | fit {len(indices['train']):,} presentations | "
        f"ceiling stim_r {ceiling_stim_r:.3f} | single-trial stimulus-driven "
        f"share of across-image spread {spread_fraction:.3f}"
    )

    image_targets = aggregate_trials_by_image(
        test_trial_targets, test_image_ids, N_TEST_IMAGES, reducer="mean"
    )

    result_rows = []
    correlations = {}
    histories = {}

    # --- ridge reference, unchanged from the noise-regularization run ---
    ridge_trial_predictions, ridge_validation_mse, ridge_alpha, _ = (
        fit_ridge_reference(
            cfg,
            train_features,
            test_features,
            allmat,
            indices,
            fit_targets,
            validation_targets,
        )
    )
    ridge_metrics, ridge_correlations = score_predictions(
        ridge_trial_predictions,
        test_trial_targets,
        test_image_ids,
        "mean",
        fit_target_mean,
        ceiling,
        response_slice,
    )
    correlations["ridge"] = ridge_correlations
    ridge_image_predictions = aggregate_trials_by_image(
        ridge_trial_predictions, test_image_ids, N_TEST_IMAGES, reducer="mean"
    )
    result_rows.append(
        {
            "experiment": "ridge",
            "best_epoch": None,
            "validation_r": None,
            "validation_mse": round(ridge_validation_mse, 5),
            "ridge_alpha": ridge_alpha,
            **ridge_metrics,
            **flatness_diagnostics(
                ridge_image_predictions,
                image_targets,
                ridge_metrics["mean_stim_r_response"],
                response_slice,
            ),
        }
    )

    # --- one decoder per training objective ---
    # BaselineModel wraps the frozen encoder even though cached features bypass
    # its forward pass, so it is built once and shared by every objective.
    from useful_stuff.image_processing.computational_models import imgANN

    encoder = imgANN(
        model_name="dino_v3_l",
        pkg="hf",
        img_size=224,
        pooling="mean",
        dtype=torch.float32,
        attn_implementation="sdpa",
        repo_url="facebook/dinov3-vitl16-pretrain-lvd1689m",
        trust_remote_code=True,
    )
    requested = [
        name.strip() for name in cfg.experiments.split(",") if name.strip()
    ]
    model_seeds = [int(seed) for seed in cfg.model_seeds.split(",")]
    unknown = sorted(set(requested) - set(EXPERIMENTS))
    if unknown:
        raise KeyError(f"Unknown experiments: {unknown}.")
    # end if an experiment name is invalid

    for name in requested:
        loss_kwargs = dict(EXPERIMENTS[name])
        decoder_name = loss_kwargs.pop("model", "baseline")
        # The spread target is derived from this pool's own reliability unless
        # an experiment deliberately overshoots it.
        spread_scale = loss_kwargs.pop("spread_scale", 1.0)
        if loss_kwargs.get("spread_weight", 0.0) != 0.0:
            loss_kwargs["target_spread_fraction"] = (
                spread_fraction * spread_scale
            )
        # end if this objective carries a spread penalty
        for model_seed in model_seeds:
            print(
                f"training {name} seed {model_seed} | {decoder_name} | "
                f"{loss_kwargs}"
            )

            torch.manual_seed(model_seed)
            model = build_variant_model(
                decoder_name, encoder, cfg, n_timepoints, n_neurons
            ).to(device)

            history, best_epoch, best_validation_r = train_with_objective(
                model,
                loaders,
                make_neural_prediction_loss(**loss_kwargs),
                cfg,
                response_slice,
                device,
            )
            if model_seed == model_seeds[0]:
                histories[name] = history
            # end if this is the objective's reference seed

            trial_predictions, trial_targets, _ = predict_test_trials(
                model, loaders["test"], device
            )
            metrics, channel_time_correlations = score_predictions(
                trial_predictions,
                trial_targets,
                test_image_ids,
                "mean",
                fit_target_mean,
                ceiling,
                response_slice,
            )
            if model_seed == model_seeds[0]:
                correlations[name] = channel_time_correlations
            # end if this is the objective's reference seed
            image_predictions = aggregate_trials_by_image(
                trial_predictions, test_image_ids, N_TEST_IMAGES, reducer="mean"
            )
            result_rows.append(
                {
                    "experiment": name,
                    "decoder": decoder_name,
                    "model_seed": model_seed,
                    "best_epoch": best_epoch,
                    "validation_r": round(best_validation_r, 4),
                    "validation_mse": round(
                        history["validation_mse"][best_epoch - 1]
                        if best_epoch > 0
                        else history["initial_validation_mse"],
                        5,
                    ),
                    "ridge_alpha": None,
                    **metrics,
                    **flatness_diagnostics(
                        image_predictions,
                        image_targets,
                        metrics["mean_stim_r_response"],
                        response_slice,
                    ),
                }
            )
            print(
                f"  best epoch {best_epoch} | validation r {best_validation_r:.4f}"
                f" | test stim_r {metrics['mean_stim_r_response']:.4f}"
            )
        # end for model initialization seed
    # end for training objective

    with open(output_dir / "config.json", "w") as config_file:
        json.dump(asdict(cfg), config_file, indent=2)
    # end with saved configuration
    with open(output_dir / "results.json", "w") as results_file:
        json.dump(result_rows, results_file, indent=2)
    # end with saved metrics
    plot_flatness_sweep(
        cfg,
        histories,
        correlations,
        ceiling,
        [
            row
            for row in result_rows
            if row["experiment"] in EXPERIMENT_COLORS
            and row.get("model_seed", model_seeds[0]) == model_seeds[0]
        ],
        output_dir / "flatness_penalty_comparison.png",
    )

    header = (
        f"{'objective':<14}{'sd':>3}{'ep':>4}{'val r':>8}{'test r':>8}{'frac ceil':>11}"
        f"{'test MSE':>10}{'VE':>9}{'spread':>8}{'calib':>7}{'img var%':>10}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in result_rows:
        # Ridge has no epochs and no validation correlation to report.
        epoch_text = "-" if row["best_epoch"] is None else str(row["best_epoch"])
        validation_text = (
            "-" if row["validation_r"] is None else f"{row['validation_r']:.4f}"
        )
        seed_text = "-" if "model_seed" not in row else str(row["model_seed"])
        print(
            f"{row['experiment']:<14}{seed_text:>3}{epoch_text:>4}{validation_text:>8}"
            f"{row['mean_stim_r_response']:>8.4f}"
            f"{row['fraction_of_ceiling']:>11.3f}"
            f"{row['test_mse']:>10.4f}{row['variance_explained_vs_mean']:>9.4f}"
            f"{row['spread_ratio']:>8.3f}{row['calibration']:>7.2f}"
            f"{row['image_specific_share']:>10.3f}"
        )
    # end for result row
    print(f"\nsaved to {output_dir}")
# EOF


if __name__ == "__main__":
    main()
# EOC
