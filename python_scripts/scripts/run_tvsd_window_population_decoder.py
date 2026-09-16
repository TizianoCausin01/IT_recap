"""
Fit the TVSD monkey F IT population vector in the high-SNR response window.

The neural target is collapsed to a single vector per presentation: the mean
baseline-corrected MUA over the requested window (75-175 ms by default, snapped
to the 10 ms bins of the cached target). Three frozen DINOv3 depths (3, 13, 20)
enter as tokens, one self-attention block lets them exchange information, and an
MLP reads out all 320 IT sites at once under plain MSE.

A RidgeCV map from the same concatenated depths to the same window-averaged
target is fitted on the identical split and scored through the identical code
path, so the two numbers differ only by the mapping class. The comparison is
reported per site, with a paired signed-rank test over the 320 sites.

Everything reuses the shared TVSD protocol in IT_recap.tvsd_experiments: the
seeded split of the 22,248 unique-image presentations, train-only per-channel
standardization, and the untouched 100 repeated test images.
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
from scipy.stats import wilcoxon  # noqa: E402


sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "python_scripts" / "src")
)

from IT_recap.neural_prediction_training import (  # noqa: E402
    split_half_reliability,
    stimulus_correlation,
)
from IT_recap.tvsd_experiments import (  # noqa: E402
    N_TEST_IMAGES,
    PROJECT_ROOT,
    average_targets_over_window,
    build_datasets,
    fit_ridge_map,
    gather_presentation_features,
    load_cached_data,
    load_project_paths,
    make_tensor_loader,
    resolve_device,
    score_predictions,
    select_window_bin_indices,
    standardize_targets,
)
from model_classes.window_models import (  # noqa: E402
    LayerAttentionPopulationDecoder,
)


# Fixed categorical slots reused from the other TVSD experiment scripts, so a
# method keeps its colour across the whole comparison series. Validated for
# colourblind separation; the ceiling is a reference line, not a series.
METHOD_COLORS = {"ridge": "#2a78d6", "attention_mlp": "#eb6834"}
METHOD_LABELS = {
    "ridge": "RidgeCV (linear)",
    "attention_mlp": "Layer self-attention + MLP",
}
CEILING_COLOR = "#8a8a86"
GRID_COLOR = "#e3e3e0"

# Sites are grouped into this many equal bins of their own noise ceiling for
# the reliability panel; 320 raw per-site traces are unreadable.
N_CEILING_BINS = 10

# Only the repetition mean is scored: the window average already is the
# high-SNR summary, and a minimum over repetitions has no usable noise ceiling.
REDUCER = "mean"


@dataclass
class Cfg:
    # Caches produced by train_tvsd_baseline_marimo.py; nothing is recomputed.
    env: str | None = None
    mua_file_name: str = "f_THINGS_MUA_trials.mat"
    feature_archive_name: str = "tvsd_monkeyF_dino_v3_l_224_features.npz"
    output_dir: str | None = None

    # Identity of the cached target file, not the fitted window.
    area: str = "IT"
    target_fs: int = 100
    time_start_ms: float = 0.0
    time_end_ms: float = 200.0

    # The window that is averaged into a single population vector. Edges snap
    # to the cached 10 ms bins, so the covered range is reported at run time.
    window_start_ms: float = 75.0
    window_end_ms: float = 175.0

    # Three DINOv3 depths, selected out of the four-layer cached archive.
    model_name: str = "dino_v3_l"
    layer_names: list[str] = field(
        default_factory=lambda: [
            "layer.3.mlp.down_proj",
            "layer.13.mlp.down_proj",
            "layer.20.mlp.down_proj",
        ]
    )

    # Split, identical to every other TVSD experiment in this repository.
    validation_fraction: float = 0.1
    random_seed: int = 0

    # Decoder architecture. These defaults come from a validation-MSE selected
    # sweep over dropout, width, weight decay, and token pooling: with 20k
    # training images and 320 noisy sites the decoder overfits within ten
    # epochs, so heavy regularization and mean pooling beat a wider readout.
    hidden_dim: int = 256
    n_attention_heads: int = 8
    mlp_hidden_dim: int = 512
    dropout: float = 0.5
    readout_pooling: str = "mean"

    # Optimization.
    batch_size: int = 256
    num_workers: int = 0
    epochs: int = 80
    minimum_epochs: int = 20
    patience: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-1
    gradient_clip: float = 1.0

    # Checkpoint and early-stopping criterion. MSE and stim_r disagree: MSE
    # rewards the shrinkage that ridge already applies, while stim_r is the
    # image-selectivity the experiment reports, so selection defaults to it.
    selection_metric: str = "stim_r"

    # Evaluation and bookkeeping.
    noise_ceiling_resamples: int = 40
    device: str = "auto"
    smoke_test: bool = False


"""
parse_args
Parse command-line overrides into a configuration object.

OUTPUT:
    - cfg: Cfg -> data, window, architecture, and optimization settings
"""
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    default_layer_names = Cfg.__dataclass_fields__["layer_names"].default_factory()
    for field_name, field_definition in Cfg.__dataclass_fields__.items():
        if field_name == "layer_names":
            # An archive may hold more depths than one experiment uses, so the
            # selection is a list rather than a single typed value.
            parser.add_argument(
                "--layer_names",
                default=",".join(default_layer_names),
                help="Comma-separated hooked ANN layers, in model order.",
            )
            continue
        # end if the layer selection needs list parsing
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
    arguments = vars(parser.parse_args())
    arguments["layer_names"] = [
        layer_name.strip()
        for layer_name in arguments["layer_names"].split(",")
        if layer_name.strip()
    ]
    if not arguments["layer_names"]:
        raise ValueError("layer_names must contain at least one ANN layer.")
    # end if no ANN layers were selected
    return Cfg(**arguments)
# EOF


"""
evaluate_decoder
Run the decoder over a loader and collect predictions, targets, and attention.

INPUT:
    - model: nn.Module -> trained or partially trained decoder
    - loader: DataLoader -> evaluation batches
    - device: torch.device -> compute device
    - return_diagnostics: bool -> whether to accumulate mean attention

OUTPUT:
    - predictions: np.ndarray -> [presentations, 1, channels]
    - targets: np.ndarray -> [presentations, 1, channels]
    - mean_attention: np.ndarray | None -> [layers, layers] mean attention
"""
def evaluate_decoder(model, loader, device, return_diagnostics=False):
    model.eval()
    prediction_batches, target_batches = [], []
    attention_sum, attention_samples = None, 0
    with torch.inference_mode():
        for features, targets in loader:
            predictions, diagnostics = model(
                features.to(device), return_diagnostics=return_diagnostics
            )
            prediction_batches.append(predictions.cpu())
            target_batches.append(targets)
            if diagnostics is not None and diagnostics["attention"] is not None:
                attention = diagnostics["attention"].cpu()
                batch_sum = attention.sum(dim=0)
                attention_sum = (
                    batch_sum if attention_sum is None else attention_sum + batch_sum
                )
                attention_samples += attention.shape[0]
            # end if attention diagnostics were requested
        # end for evaluation batch
    # end with inference mode

    mean_attention = (
        None
        if attention_sum is None
        else (attention_sum / attention_samples).numpy()
    )
    return (
        torch.cat(prediction_batches).numpy(),
        torch.cat(target_batches).numpy(),
        mean_attention,
    )
# EOF


"""
layer_depth_label
Reduce a hooked module name to its block index for compact printing.

DINOv3 hooks read "layer.13.mlp.down_proj" and I-JEPA hooks read
"encoder.layer.17.output.dense", so the depth is the first numeric component
rather than a fixed position.

INPUT:
    - layer_name: str -> hooked module name

OUTPUT:
    - label: str -> block index, or the full name when none is present
"""
def layer_depth_label(layer_name):
    for part in layer_name.split("."):
        if part.isdigit():
            return part
        # end if this component is the block index
    # end for name component
    return layer_name
# EOF


"""
mean_validation_stim_r
Mean across-image correlation over sites, on unique-image presentations.

The validation split holds one presentation per image, so this is the same
quantity as the reported test stim_r, only measured on single trials rather
than on the mean of 30 repetitions.

INPUT:
    - predictions: np.ndarray -> [presentations, 1, channels]
    - targets: np.ndarray -> [presentations, 1, channels]

OUTPUT:
    - mean_stim_r: float -> mean per-site correlation across images
"""
def mean_validation_stim_r(predictions, targets):
    return float(np.nanmean(stimulus_correlation(predictions, targets)))
# EOF


"""
train_decoder
Optimize the decoder under MSE and restore its best validation checkpoint.

INPUT:
    - model: nn.Module -> decoder to optimize
    - loaders: dict -> train and validation loaders
    - cfg: Cfg -> optimization and early-stopping settings
    - device: torch.device -> compute device

OUTPUT:
    - history: list[dict] -> per-epoch train MSE, validation MSE, learning rate
    - best_epoch: int -> selected checkpoint epoch
    - best_validation_mse: float -> validation MSE of the restored checkpoint
"""
def train_decoder(model, loaders, cfg, device):
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
    if cfg.selection_metric not in {"mse", "stim_r"}:
        raise ValueError("selection_metric must be either 'mse' or 'stim_r'.")
    # end if the selection criterion is unsupported

    best_state = copy.deepcopy(model.state_dict())
    best_validation_mse = np.inf
    best_validation_stim_r = -np.inf
    # Both criteria are minimized once stim_r is negated.
    best_selection_score = np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_squared_error, train_values = 0.0, 0
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

        validation_predictions, validation_targets, _ = evaluate_decoder(
            model, loaders["validation"], device
        )
        train_mse = train_squared_error / train_values
        validation_mse = float(
            np.mean((validation_predictions - validation_targets) ** 2)
        )
        validation_stim_r = mean_validation_stim_r(
            validation_predictions, validation_targets
        )
        selection_score = (
            validation_mse
            if cfg.selection_metric == "mse"
            else -validation_stim_r
        )
        # The schedule follows whichever criterion is being selected on.
        scheduler.step(selection_score)
        history.append(
            {
                "epoch": epoch,
                "train_mse": train_mse,
                "validation_mse": validation_mse,
                "validation_stim_r": validation_stim_r,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"    epoch {epoch:03d}/{cfg.epochs:03d} | train {train_mse:.6f} | "
            f"validation MSE {validation_mse:.6f} | validation stim_r "
            f"{validation_stim_r:.4f}"
        )

        if selection_score < best_selection_score - 1e-8:
            best_selection_score = selection_score
            best_validation_mse = validation_mse
            best_validation_stim_r = validation_stim_r
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        # end if this epoch improved the selection criterion

        if (
            epoch >= cfg.minimum_epochs
            and epochs_without_improvement >= cfg.patience
        ):
            print(f"    early stop after {epoch} epochs")
            break
        # end if early-stopping patience is exhausted
    # end for optimization epoch

    model.load_state_dict(best_state)
    return (
        history,
        best_epoch,
        float(best_validation_mse),
        float(best_validation_stim_r),
    )
# EOF


"""
compare_site_correlations
Compare two methods site by site with a paired signed-rank test.

INPUT:
    - decoder_r: np.ndarray -> [channels] decoder stim_r per site
    - ridge_r: np.ndarray -> [channels] ridge stim_r per site
    - ceiling: np.ndarray -> [channels] noise ceiling per site

OUTPUT:
    - comparison: dict -> paired differences, win rate, and test statistics
"""
def compare_site_correlations(decoder_r, ridge_r, ceiling):
    valid_sites = np.isfinite(decoder_r) & np.isfinite(ridge_r)
    differences = decoder_r[valid_sites] - ridge_r[valid_sites]
    statistic, p_value = wilcoxon(differences)

    # Relative gain is measured in ceiling units: an absolute stim_r gain means
    # different things at a site with ceiling 0.9 and one with ceiling 0.3.
    valid_ceiling = ceiling[valid_sites]
    usable = valid_ceiling > 0
    mean_ceiling_fraction_gain = float(
        np.mean(differences[usable] / valid_ceiling[usable])
    )
    return {
        "n_sites_compared": int(valid_sites.sum()),
        "mean_stim_r_gain": round(float(np.mean(differences)), 4),
        "median_stim_r_gain": round(float(np.median(differences)), 4),
        "sites_won_by_decoder": int((differences > 0).sum()),
        "win_rate": round(float((differences > 0).mean()), 3),
        "mean_ceiling_fraction_gain": round(mean_ceiling_fraction_gain, 4),
        "wilcoxon_statistic": float(statistic),
        "wilcoxon_p_value": float(p_value),
    }
# EOF


"""
plot_window_comparison
Draw the optimization curve, the per-site comparison, and the sorted profiles.

INPUT:
    - cfg: Cfg -> optimization settings used for the axis labels
    - history: list[dict] -> decoder training history
    - ridge_validation_stim_r: float -> ridge reference on the same validation set
    - site_correlations: dict -> per-method [channels] stim_r
    - ceiling: np.ndarray -> [channels] noise ceiling per site
    - covered_ms: tuple[float, float] -> the averaged window
    - output_path: Path -> destination figure file

OUTPUT:
    - None: writes the figure to disk
"""
def plot_window_comparison(
    cfg,
    history,
    ridge_validation_stim_r,
    site_correlations,
    ceiling,
    covered_ms,
    output_path,
):
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    for axis in axes:
        axis.grid(True, color=GRID_COLOR, linewidth=0.8)
        axis.set_axisbelow(True)
        for spine_name in ("top", "right"):
            axis.spines[spine_name].set_visible(False)
        # end for hidden spine
    # end for panel

    # --- panel 1: optimization, against the single ridge reference value ---
    epochs = [entry["epoch"] for entry in history]
    axes[0].plot(
        epochs,
        [entry["validation_stim_r"] for entry in history],
        linewidth=2,
        color=METHOD_COLORS["attention_mlp"],
        label="decoder validation",
    )
    axes[0].axhline(
        ridge_validation_stim_r,
        linewidth=2,
        color=METHOD_COLORS["ridge"],
        label="ridge validation",
    )
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("validation stim_r (single trials)")
    axes[0].set_title("Optimization")
    axes[0].legend(frameon=False, fontsize=9)

    # --- panel 2: per-site stim_r, decoder against ridge ---
    ridge_r = site_correlations["ridge"]
    decoder_r = site_correlations["attention_mlp"]
    axis_limits = (
        float(np.nanmin([ridge_r.min(), decoder_r.min()]) - 0.05),
        float(np.nanmax([ridge_r.max(), decoder_r.max()]) + 0.05),
    )
    axes[1].plot(
        axis_limits,
        axis_limits,
        linewidth=1.5,
        color=CEILING_COLOR,
        zorder=1,
    )
    axes[1].scatter(
        ridge_r,
        decoder_r,
        s=18,
        color=METHOD_COLORS["attention_mlp"],
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85,
        zorder=2,
    )
    axes[1].set_xlim(axis_limits)
    axes[1].set_ylim(axis_limits)
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("ridge stim_r")
    axes[1].set_ylabel("decoder stim_r")
    axes[1].set_title("Per-site prediction (points above the line: decoder wins)")

    # --- panel 3: binned profile against the noise ceiling ---
    # 320 raw per-site traces are unreadable, so sites are grouped into equal
    # deciles of their own ceiling. That also shows directly whether the gain
    # over ridge depends on how reliable a site is.
    site_order = np.argsort(ceiling)
    site_bins = np.array_split(site_order, N_CEILING_BINS)
    bin_centers = np.arange(len(site_bins))
    axes[2].plot(
        bin_centers,
        [ceiling[bin_sites].mean() for bin_sites in site_bins],
        linewidth=2,
        marker="o",
        markersize=8,
        color=CEILING_COLOR,
        label="noise ceiling",
    )
    for method_name in ("ridge", "attention_mlp"):
        axes[2].plot(
            bin_centers,
            [
                np.nanmean(site_correlations[method_name][bin_sites])
                for bin_sites in site_bins
            ],
            linewidth=2,
            marker="o",
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=0.8,
            color=METHOD_COLORS[method_name],
            label=METHOD_LABELS[method_name],
        )
    # end for scored method
    axes[2].set_xticks(bin_centers)
    axes[2].set_xticklabels([f"{index + 1}" for index in bin_centers])
    axes[2].set_xlabel(f"IT sites, {N_CEILING_BINS} equal bins of rising ceiling")
    axes[2].set_ylabel("mean stim_r")
    axes[2].set_title("Prediction vs ceiling, by site reliability")
    axes[2].legend(frameon=False, fontsize=9, loc="lower right")

    figure.suptitle(
        f"TVSD monkey F {cfg.area}: window-averaged population decoding from "
        f"{cfg.model_name} ({covered_ms[0]:g}-{covered_ms[1]:g} ms)"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
# EOF


def main():
    cfg = parse_args()
    if cfg.smoke_test:
        cfg.epochs = min(cfg.epochs, 3)
        cfg.minimum_epochs = 1
        cfg.noise_ceiling_resamples = 4
    # end if a smoke run was requested

    paths = load_project_paths(cfg)
    device = resolve_device(cfg.device)
    output_dir = Path(
        cfg.output_dir
        or PROJECT_ROOT
        / "results"
        / f"tvsd_window_population_decoder_{cfg.model_name}"
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    targets, train_features, test_features, allmat = load_cached_data(cfg, paths)

    # Collapse the response window into one population vector per presentation.
    # The singleton time axis is kept so the shared scoring code applies as is.
    bin_indices, covered_ms = select_window_bin_indices(
        targets.shape[1],
        cfg.target_fs,
        cfg.time_start_ms,
        cfg.window_start_ms,
        cfg.window_end_ms,
    )
    window_targets = average_targets_over_window(targets, bin_indices)
    n_neurons = window_targets.shape[2]

    _, indices, (channel_mean, channel_scale) = build_datasets(
        cfg, window_targets, train_features, test_features, allmat
    )
    print(
        f"device {device} | requested window {cfg.window_start_ms:g}-"
        f"{cfg.window_end_ms:g} ms -> bins {bin_indices[0]}-{bin_indices[-1]} "
        f"({covered_ms[0]:g}-{covered_ms[1]:g} ms, "
        f"{len(bin_indices)} bins averaged)"
    )
    print(
        f"fit {len(indices['train']):,} | validation "
        f"{len(indices['validation']):,} | test {len(indices['test']):,} "
        f"presentations | target [{n_neurons} sites] | "
        f"{len(cfg.layer_names)} DINOv3 depths"
    )

    # Materialize the standardized targets and the per-presentation features
    # once; the ridge fit and the decoder consume exactly the same arrays.
    subset_targets = {
        subset_name: standardize_targets(
            window_targets[subset_indices], channel_mean, channel_scale
        )
        for subset_name, subset_indices in indices.items()
    }
    subset_features = {
        subset_name: gather_presentation_features(
            train_features, test_features, allmat, subset_indices
        )
        for subset_name, subset_indices in indices.items()
    }
    fit_target_mean = subset_targets["train"].mean(axis=0)
    test_image_ids = allmat[indices["test"], 2] - 1

    # The single averaged bin is itself the response, so nothing is excluded.
    response_slice = slice(0, 1)
    ceiling = split_half_reliability(
        subset_targets["test"],
        test_image_ids,
        N_TEST_IMAGES,
        reducer=REDUCER,
        n_resamples=cfg.noise_ceiling_resamples,
        seed=cfg.random_seed,
    )

    result_rows = []
    site_correlations = {}

    # --- ridge reference on the identical split ---
    print("fitting RidgeCV reference")
    ridge_fit = fit_ridge_map(
        cfg,
        train_features,
        test_features,
        allmat,
        indices,
        subset_targets["train"],
        subset_targets["validation"],
    )
    ridge_validation_mse = ridge_fit["validation_mse"]
    ridge_validation_stim_r = mean_validation_stim_r(
        ridge_fit["validation_predictions"], subset_targets["validation"]
    )
    ridge_metrics, ridge_site_r = score_predictions(
        ridge_fit["trial_predictions"],
        subset_targets["test"],
        test_image_ids,
        REDUCER,
        fit_target_mean,
        ceiling,
        response_slice,
    )
    site_correlations["ridge"] = ridge_site_r[0]
    result_rows.append(
        {
            "method": "ridge",
            "validation_mse": round(ridge_validation_mse, 5),
            "validation_stim_r": round(ridge_validation_stim_r, 4),
            "best_epoch": None,
            "trainable_parameters": ridge_fit["n_coefficients"],
            "ridge_alpha": ridge_fit["alpha"],
            **ridge_metrics,
        }
    )
    print(
        f"  ridge alpha {ridge_fit['alpha']:g} | validation MSE "
        f"{ridge_validation_mse:.5f} | validation stim_r "
        f"{ridge_validation_stim_r:.4f} | test stim_r "
        f"{ridge_metrics['mean_stim_r_response']:.4f}"
    )

    # --- self-attention + MLP decoder ---
    print("training layer self-attention + MLP decoder")
    torch.manual_seed(cfg.random_seed)
    model = LayerAttentionPopulationDecoder(
        n_layers=len(cfg.layer_names),
        feature_dim=train_features.shape[2],
        n_neurons=n_neurons,
        hidden_dim=cfg.hidden_dim,
        n_attention_heads=cfg.n_attention_heads,
        mlp_hidden_dim=cfg.mlp_hidden_dim,
        dropout=cfg.dropout,
        readout_pooling=cfg.readout_pooling,
    ).to(device)
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    loaders = {
        subset_name: make_tensor_loader(
            subset_features[subset_name],
            subset_targets[subset_name],
            cfg,
            shuffle=subset_name == "train",
        )
        for subset_name in ("train", "validation", "test")
    }
    (
        history,
        best_epoch,
        best_validation_mse,
        best_validation_stim_r,
    ) = train_decoder(model, loaders, cfg, device)
    decoder_predictions, decoder_targets, mean_attention = evaluate_decoder(
        model, loaders["test"], device, return_diagnostics=True
    )
    decoder_metrics, decoder_site_r = score_predictions(
        decoder_predictions,
        decoder_targets,
        test_image_ids,
        REDUCER,
        fit_target_mean,
        ceiling,
        response_slice,
    )
    site_correlations["attention_mlp"] = decoder_site_r[0]
    result_rows.append(
        {
            "method": "attention_mlp",
            "validation_mse": round(best_validation_mse, 5),
            "validation_stim_r": round(best_validation_stim_r, 4),
            "best_epoch": best_epoch,
            "trainable_parameters": trainable_parameters,
            "ridge_alpha": None,
            **decoder_metrics,
        }
    )

    comparison = compare_site_correlations(
        site_correlations["attention_mlp"], site_correlations["ridge"], ceiling[0]
    )

    torch.save(
        {
            "model_state_dict": {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            },
            "cfg": asdict(cfg),
            "channel_mean": channel_mean,
            "channel_scale": channel_scale,
            "window_bin_indices": bin_indices,
            "covered_window_ms": covered_ms,
            "best_epoch": best_epoch,
            "best_validation_mse": best_validation_mse,
            "best_validation_stim_r": best_validation_stim_r,
            "mean_test_attention": mean_attention,
        },
        output_dir / "attention_mlp_decoder.pt",
    )
    with open(output_dir / "config.json", "w") as config_file:
        json.dump(
            {
                **asdict(cfg),
                "covered_window_ms": list(covered_ms),
                "window_bin_indices": bin_indices.tolist(),
            },
            config_file,
            indent=2,
        )
    # end with saved configuration
    with open(output_dir / "results.json", "w") as results_file:
        json.dump(
            {"rows": result_rows, "ridge_comparison": comparison},
            results_file,
            indent=2,
        )
    # end with saved metrics
    np.savez_compressed(
        output_dir / "site_stim_r.npz",
        ridge=site_correlations["ridge"],
        attention_mlp=site_correlations["attention_mlp"],
        ceiling=ceiling[0],
        mean_test_attention=mean_attention,
    )
    plot_window_comparison(
        cfg,
        history,
        ridge_validation_stim_r,
        site_correlations,
        ceiling[0],
        covered_ms,
        output_dir / "window_population_decoder_comparison.png",
    )

    # The printed table doubles as the accessible alternative to the figure.
    header = (
        f"{'method':<16}{'val MSE':>10}{'val r':>8}{'test MSE':>10}"
        f"{'VE vs mean':>12}{'mean r':>9}{'median r':>10}{'ceiling':>9}"
        f"{'frac ceil':>11}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in result_rows:
        print(
            f"{row['method']:<16}{row['validation_mse']:>10.5f}"
            f"{row['validation_stim_r']:>8.4f}{row['test_mse']:>10.5f}"
            f"{row['variance_explained_vs_mean']:>12.4f}"
            f"{row['mean_stim_r_response']:>9.4f}"
            f"{row['median_stim_r_response']:>10.4f}"
            f"{row['target_reliability']:>9.4f}"
            f"{row['fraction_of_ceiling']:>11.3f}"
        )
    # end for result row

    print(
        f"\ndecoder minus ridge, per site: mean stim_r "
        f"{comparison['mean_stim_r_gain']:+.4f}, median "
        f"{comparison['median_stim_r_gain']:+.4f}, won "
        f"{comparison['sites_won_by_decoder']}/"
        f"{comparison['n_sites_compared']} sites "
        f"({comparison['win_rate']:.1%}), Wilcoxon p="
        f"{comparison['wilcoxon_p_value']:.2e}"
    )
    if mean_attention is not None:
        print("\nmean test self-attention (rows: query depth, columns: key depth)")
        depth_labels = [layer_depth_label(name) for name in cfg.layer_names]
        print("            " + "".join(f"{label:>10}" for label in depth_labels))
        for row_index, depth_label in enumerate(depth_labels):
            weights = "".join(
                f"{value:>10.3f}" for value in mean_attention[row_index]
            )
            print(f"  layer {depth_label:<4}{weights}")
        # end for query depth
    # end if attention diagnostics were collected
    print(f"\nsaved to {output_dir}")
# EOF


if __name__ == "__main__":
    main()
# EOC
