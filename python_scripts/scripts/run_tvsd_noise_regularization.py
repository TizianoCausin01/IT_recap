"""
Compare two noise-regularized BaselineModel variants against the ridge baseline.

Both variants keep the notebook's TVSD monkey F pipeline unchanged -- same
frozen DINOv3 layers, same seeded split of the 22,248 unique-image
presentations, same per-channel target standardization, same 100 repeated test
images -- and differ only in where noise enters the decoder:

    noise_layer     one extra attended pseudo-layer drawn uniformly from the
                    1024-d Gaussian sphere; scored by aggregating the 30
                    repetitions of each test image with an element-wise MINIMUM
    temporal_noise  Gaussian jitter added to the learned temporal embeddings
                    before they are projected into query space

The plain BaselineModel and a RidgeCV map on the concatenated layers are
trained and scored alongside them as references.
"""

import argparse
import copy
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402


sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "python_scripts" / "src")
)

from IT_recap.neural_prediction_training import (  # noqa: E402
    neural_activity_timebin_mse_loss,
    split_half_reliability,
    test_step,
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


# Fixed categorical slots (validated for colorblind separation), assigned by
# entity so a variant keeps its colour in every panel.
VARIANT_COLORS = {
    "ridge": "#2a78d6",
    "baseline": "#eb6834",
    "noise_layer": "#1baf7a",
    "temporal_noise": "#4a3aa7",
}
VARIANT_LABELS = {
    "ridge": "RidgeCV",
    "baseline": "BaselineModel",
    "noise_layer": "BaselineModel + noise layer",
    "temporal_noise": "BaselineModel + temporal-embedding noise",
}
TORCH_VARIANTS = ("baseline", "noise_layer", "temporal_noise")


@dataclass
class Cfg:
    # Caches produced by train_tvsd_baseline_marimo.py; nothing is recomputed.
    env: str | None = None
    mua_file_name: str = "f_THINGS_MUA_trials.mat"
    feature_archive_name: str = "tvsd_monkeyF_dino_v3_l_224_features.npz"
    output_dir: str | None = None

    # Neural target window, identical to the notebook's cache name.
    area: str = "IT"
    target_fs: int = 100
    time_start_ms: float = 0.0
    time_end_ms: float = 200.0
    response_onset_ms: float = 50.0

    # Frozen encoder that BaselineModel wraps even when cached features are used.
    model_name: str = "dino_v3_l"
    model_source: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    img_size: int = 224
    pooling: str = "mean"
    layer_names: list[str] = field(
        default_factory=lambda: [
            "layer.3.mlp.down_proj",
            "layer.13.mlp.down_proj",
            "layer.16.mlp.down_proj",
            "layer.20.mlp.down_proj",
        ]
    )
    trust_remote_code: bool = True
    attn_implementation: str = "sdpa"

    # Split and optimization, mirroring the notebook so numbers stay comparable.
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

    # Variant-specific noise settings.
    match_noise_norm: bool = True
    noise_layer_in_eval: bool = True
    temporal_noise_std: float = 0.25
    relative_temporal_noise: bool = True
    temporal_noise_in_eval: bool = False

    # Evaluation and bookkeeping.
    noise_ceiling_resamples: int = 40
    variants: str = ",".join(TORCH_VARIANTS)
    device: str = "auto"
    smoke_test: bool = False


"""
parse_args
Parse command-line overrides into a configuration object.

OUTPUT:
    - cfg: Cfg -> data, split, architecture, noise, and optimization settings
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
train_variant
Optimize one decoder and restore the epoch with the lowest validation MSE.

INPUT:
    - model: BaselineModel -> decoder to optimize
    - loaders: dict -> train and validation loaders
    - cfg: Cfg -> optimization settings
    - device: torch.device -> compute device

OUTPUT:
    - history: dict -> initial, per-epoch train, and per-epoch validation MSE
    - best_epoch: int -> selected checkpoint epoch, zero for the initialization
    - best_validation_loss: float -> validation MSE of the restored checkpoint
"""
def train_variant(model, loaders, cfg, device):
    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    optimizer.zero_grad(set_to_none=True)
    cost_function = neural_activity_timebin_mse_loss

    initial_validation_loss = test_step(
        model,
        loaders["validation"],
        cost_function,
        use_precomputed_features=True,
        device=device,
    )
    best_validation_loss = initial_validation_loss
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    training_losses, validation_losses = [], []

    for epoch in range(1, cfg.epochs + 1):
        training_loss = training_step(
            model,
            loaders["train"],
            optimizer,
            cost_function,
            use_precomputed_features=True,
            device=device,
        )
        validation_loss = test_step(
            model,
            loaders["validation"],
            cost_function,
            use_precomputed_features=True,
            device=device,
        )
        training_losses.append(training_loss)
        validation_losses.append(validation_loss)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        # end if this epoch is the best validation checkpoint
        print(
            f"    epoch {epoch:03d}/{cfg.epochs:03d} | "
            f"train {training_loss:.6f} | validation {validation_loss:.6f}"
        )
    # end for optimization epoch

    model.load_state_dict(best_state)
    history = {
        "initial_validation_mse": initial_validation_loss,
        "train_mse": training_losses,
        "validation_mse": validation_losses,
    }
    return history, best_epoch, best_validation_loss
# EOF




"""
plot_comparison
Draw validation curves and the two repetition-aggregated stim_r time courses.

INPUT:
    - cfg: Cfg -> time-axis settings
    - histories: dict -> per-variant training histories
    - correlations: dict -> per-variant, per-aggregation [time, channels] stim_r
    - ridge_validation_mse: float -> ridge reference on the same validation set
    - ceilings: dict -> per-aggregation [time, channels] reference values
    - attentions: dict -> per-variant mean [time, attended items] attention
    - layer_names: list[str] -> hooked ANN layers, without the noise pseudo-layer
    - output_path: Path -> destination figure file

OUTPUT:
    - None: writes the figure to disk
"""
def plot_comparison(
    cfg,
    histories,
    correlations,
    ridge_validation_mse,
    ceilings,
    attentions,
    layer_names,
    output_path,
):
    response_time_ms = np.arange(
        cfg.time_start_ms, cfg.time_end_ms, 1000 / cfg.target_fs
    )
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()

    # Panel 1: validation MSE against the single ridge reference value.
    for variant, history in histories.items():
        validation_curve = [
            history["initial_validation_mse"],
            *history["validation_mse"],
        ]
        axes[0].plot(
            np.arange(len(validation_curve)),
            validation_curve,
            linewidth=2,
            color=VARIANT_COLORS[variant],
            label=VARIANT_LABELS[variant],
        )
    # end for decoder variant
    axes[0].axhline(
        ridge_validation_mse,
        linewidth=2,
        linestyle="--",
        color=VARIANT_COLORS["ridge"],
        label=f"{VARIANT_LABELS['ridge']} ({ridge_validation_mse:.4f})",
    )
    axes[0].set(
        xlabel="Epoch",
        ylabel="Validation MSE (standardized)",
        title="Held-out unique images",
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    # Panels 2 and 3: the same metric under the two repetition aggregations.
    for panel_index, reducer in enumerate(("mean", "min"), start=1):
        axis = axes[panel_index]
        for variant in ("ridge", *TORCH_VARIANTS):
            axis.plot(
                response_time_ms,
                np.nanmedian(correlations[variant][reducer], axis=1),
                linewidth=2,
                color=VARIANT_COLORS[variant],
                label=VARIANT_LABELS[variant],
            )
        # end for scored variant
        axis.plot(
            response_time_ms,
            np.nanmedian(ceilings[reducer], axis=1),
            color="black",
            linestyle=":",
            linewidth=2,
            label="Reliability reference",
        )
        axis.axhline(0, color="black", linewidth=1)
        axis.axvline(
            cfg.response_onset_ms, color="gray", linewidth=1, linestyle="--"
        )
        axis.set(
            xlabel="Time from image onset (ms)",
            ylabel="Median stim_r across MUA sites",
            title=f"100 test images, {reducer} over 30 repetitions",
        )
        axis.legend(fontsize=8)
        axis.grid(alpha=0.25)
    # end for repetition aggregation

    # Panel 4: attention depth profile of the noise-layer variant. Layer depth
    # is ordered, so it gets one hue light-to-dark; the noise slot, which is not
    # part of that ordering, is drawn as a separate black dashed line.
    axis = axes[3]
    if "noise_layer" in attentions:
        mean_attention = attentions["noise_layer"]
        layer_shades = plt.cm.Blues(np.linspace(0.35, 0.95, len(layer_names)))
        for layer_index, layer_name in enumerate(layer_names):
            axis.plot(
                response_time_ms,
                mean_attention[:, layer_index],
                linewidth=2,
                color=layer_shades[layer_index],
                label=layer_name,
            )
        # end for hooked DINO layer
        axis.plot(
            response_time_ms,
            mean_attention[:, -1],
            linewidth=2,
            linestyle="--",
            color="black",
            label="Gaussian-sphere noise",
        )
        axis.axhline(
            1.0 / mean_attention.shape[1],
            color="gray",
            linewidth=1,
            linestyle=":",
            label="Uniform over attended items",
        )
    # end if the noise-layer variant was trained
    axis.set(
        xlabel="Time from image onset (ms)",
        ylabel="Mean attention weight",
        title="Where the noise-layer variant spends attention",
    )
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)

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
    from useful_stuff.image_processing.computational_models import imgANN

    device = resolve_device(cfg.device)
    output_dir = Path(
        cfg.output_dir or PROJECT_ROOT / "results" / "tvsd_noise_regularization"
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    targets, train_features, test_features, allmat = load_cached_data(cfg, paths)
    datasets, indices, (channel_mean, channel_scale) = build_datasets(
        cfg, targets, train_features, test_features, allmat
    )
    loaders = build_loaders(cfg, datasets)
    n_timepoints, n_neurons = targets.shape[1], targets.shape[2]
    print(
        f"device {device} | fit {len(indices['train']):,} | validation "
        f"{len(indices['validation']):,} | test {len(indices['test']):,} "
        f"presentations | target [{n_timepoints} time x {n_neurons} sites]"
    )

    # Materialize the standardized targets once; both the ridge fit and every
    # aggregation below reuse them.
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

    # MUA drive begins around 50 ms; earlier bins carry no stimulus signal and
    # would dilute every correlation summary equally.
    response_slice = slice(
        int(round(cfg.response_onset_ms / (1000 / cfg.target_fs))), n_timepoints
    )
    ceilings = {
        reducer: split_half_reliability(
            test_trial_targets,
            test_image_ids,
            N_TEST_IMAGES,
            reducer=reducer,
            n_resamples=cfg.noise_ceiling_resamples,
            seed=cfg.random_seed,
        )
        for reducer in ("mean", "min")
    }

    result_rows = []
    correlations = {}
    histories = {}
    attentions = {}

    # --- ridge reference on the identical split ---
    print("fitting RidgeCV reference")
    ridge_trial_predictions, ridge_validation_mse, ridge_alpha, ridge_parameters = (
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
    correlations["ridge"] = {}
    for reducer in ("mean", "min"):
        metrics, channel_time_correlations = score_predictions(
            ridge_trial_predictions,
            test_trial_targets,
            test_image_ids,
            reducer,
            fit_target_mean,
            ceilings[reducer],
            response_slice,
        )
        correlations["ridge"][reducer] = channel_time_correlations
        result_rows.append(
            {
                "variant": "ridge",
                "validation_mse": round(ridge_validation_mse, 5),
                "best_epoch": None,
                "trainable_parameters": ridge_parameters,
                "ridge_alpha": ridge_alpha,
                "noise_attention_mass": None,
                **metrics,
            }
        )
    # end for repetition aggregation
    print(
        f"  ridge alpha {ridge_alpha:g} | validation MSE "
        f"{ridge_validation_mse:.5f}"
    )

    # --- torch decoder variants on the shared frozen encoder ---
    encoder = imgANN(
        model_name=cfg.model_name,
        pkg="hf",
        img_size=cfg.img_size,
        pooling=cfg.pooling,
        dtype=torch.float32,
        attn_implementation=cfg.attn_implementation,
        repo_url=cfg.model_source,
        trust_remote_code=cfg.trust_remote_code,
    )
    requested_variants = [
        name.strip() for name in cfg.variants.split(",") if name.strip()
    ]
    for variant in requested_variants:
        print(f"training {variant}")
        torch.manual_seed(cfg.random_seed)
        model = build_variant_model(
            variant, encoder, cfg, n_timepoints, n_neurons
        ).to(device)
        trainable_parameters = sum(
            parameter.numel()
            for parameter in model.get_trainable_parameters()
        )
        history, best_epoch, best_validation_loss = train_variant(
            model, loaders, cfg, device
        )
        histories[variant] = history

        trial_predictions, trial_targets, mean_attention = predict_test_trials(
            model, loaders["test"], device
        )
        attentions[variant] = mean_attention

        # For the noise-layer variant the last attended item is pure noise, so
        # its weight measures how much prediction budget the model discards.
        noise_attention_mass = None
        if variant == "noise_layer":
            noise_attention_mass = round(float(mean_attention[:, -1].mean()), 4)
        # end if the variant carries a noise pseudo-layer
        correlations[variant] = {}
        for reducer in ("mean", "min"):
            metrics, channel_time_correlations = score_predictions(
                trial_predictions,
                trial_targets,
                test_image_ids,
                reducer,
                fit_target_mean,
                ceilings[reducer],
                response_slice,
            )
            correlations[variant][reducer] = channel_time_correlations
            result_rows.append(
                {
                    "variant": variant,
                    "validation_mse": round(best_validation_loss, 5),
                    "best_epoch": best_epoch,
                    "trainable_parameters": trainable_parameters,
                    "ridge_alpha": None,
                    "noise_attention_mass": noise_attention_mass,
                    **metrics,
                }
            )
        # end for repetition aggregation

        # Store only decoder weights; the frozen backbone would add ~1.2 GB.
        decoder_state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
            if not name.startswith("encoder_backbone.")
        }
        torch.save(
            {
                "variant": variant,
                "model_state_dict": decoder_state,
                "cfg": asdict(cfg),
                "channel_mean": channel_mean,
                "channel_scale": channel_scale,
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
                "mean_test_attention": mean_attention,
            },
            output_dir / f"{variant}_decoder.pt",
        )
        print(
            f"  best validation MSE {best_validation_loss:.5f} at epoch "
            f"{best_epoch} | {trainable_parameters:,} trainable parameters"
        )
    # end for decoder variant

    with open(output_dir / "config.json", "w") as config_file:
        json.dump(asdict(cfg), config_file, indent=2)
    # end with saved configuration
    with open(output_dir / "results.json", "w") as results_file:
        json.dump(result_rows, results_file, indent=2)
    # end with saved metrics
    np.savez_compressed(
        output_dir / "stim_r_time_courses.npz",
        **{
            f"{variant}_{reducer}": correlations[variant][reducer]
            for variant in correlations
            for reducer in correlations[variant]
        },
        ceiling_mean=ceilings["mean"],
        ceiling_min=ceilings["min"],
    )
    plot_comparison(
        cfg,
        histories,
        correlations,
        ridge_validation_mse,
        ceilings,
        attentions,
        cfg.layer_names,
        output_dir / "noise_regularization_comparison.png",
    )

    # Printed table doubles as the accessible alternative to the figure.
    header = (
        f"{'variant':<16}{'agg':>6}{'val MSE':>10}{'test MSE':>10}"
        f"{'VE vs mean':>12}{'mean r':>9}{'reliab.':>9}{'frac ceil':>11}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in result_rows:
        ceiling_text = (
            "n/a"
            if row["fraction_of_ceiling"] is None
            else f"{row['fraction_of_ceiling']:.3f}"
        )
        print(
            f"{row['variant']:<16}{row['aggregation']:>6}"
            f"{row['validation_mse']:>10.5f}{row['test_mse']:>10.5f}"
            f"{row['variance_explained_vs_mean']:>12.4f}"
            f"{row['mean_stim_r_response']:>9.4f}"
            f"{row['target_reliability']:>9.4f}{ceiling_text:>11}"
        )
    # end for result row
    print(f"\nsaved to {output_dir}")
# EOF


if __name__ == "__main__":
    main()
# EOC
