"""
Fit the BaselineModel on Triple-N firing rates instead of TVSD MUA.

This is the control for the decoder plateau seen in the TVSD experiments. The
architecture, objective, optimizer, and scoring are the ones already used
there; the only thing that changes is the target. Triple-N units are spike
sorted and BombCell labelled, so restricting the fit to single units gives a
target with a genuinely different noise structure while the model side is held
fixed. If the decoder reaches a similar fraction of the noise ceiling on both,
the plateau is a property of the model or the DINOv3 features rather than of
the MUA.

Targets are firing rates in spikes per second, binned to target_fs and then
standardized per unit on the fit split, exactly as the MUA targets are.

Inputs this script expects on disk:

    GoodUnit_YYMMDD_*_NSD1000_LOC_gx.mat   spike rasters for one session
    Processed_sesXX_*.mat                  BombCell labels for the same session
    <stimulus feature archive>.npz         'stimulus_features' [stimuli, layers,
                                           embedding] ordered by one-based
                                           Triple-N stimulus id, plus
                                           'layer_names'

Use download_triple_n_sessions.py for the first two.
"""

import argparse
import copy
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402


ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_SRC = PROJECT_ROOT / "python_scripts" / "src"
sys.path.insert(0, str(PROJECT_SRC))

with open(PROJECT_ROOT / "config.yaml", "r") as f:
    config = yaml.safe_load(f)

from IT_recap.neural_prediction_training import (  # noqa: E402
    aggregate_attention_by_layer,
    neural_activity_timebin_mse_loss,
    split_half_reliability,
    test_step,
    training_step,
)
from IT_recap.triple_n import (  # noqa: E402
    build_triple_n_datasets,
    build_triple_n_image_metadata,
    load_triple_n_good_unit_metadata,
    load_triple_n_processed,
    preprocess_triple_n_firing_rate_targets,
    select_triple_n_units,
)
from IT_recap.tvsd_experiments import (  # noqa: E402
    N_TEST_IMAGES,
    fit_ridge_reference,
    load_project_paths,
    predict_test_trials,
    resolve_device,
    score_predictions,
    standardize_targets,
)
from model_classes.temporal_models import BaselineModel  # noqa: E402


DEFAULT_LAYERS = (
    "layer.3.mlp.down_proj",
    "layer.13.mlp.down_proj",
    "layer.16.mlp.down_proj",
    "layer.20.mlp.down_proj",
)

VARIANT_COLORS = {"ridge": "#2a78d6", "baseline": "#eb6834"}


@dataclass
class Cfg:
    # Environment and file locations. None resolves through config.yaml.
    env: str | None = None
    triple_n_dir: str | None = None
    good_unit_file: str = ""
    processed_file: str = ""
    feature_archive: str = ""
    output_dir: str | None = None

    # Target definition. The window matches the TVSD experiments so the two
    # fraction-of-ceiling numbers are directly comparable.
    time_start_ms: float = 0.0
    time_end_ms: float = 200.0
    target_fs: int = 100
    response_onset_ms: float = 50.0
    subtract_baseline: bool = False

    # Unit selection. BombCell codes: 1 single unit, 2 MUA, 3-4 non-somatic.
    unit_types: str = "1"
    min_reliability: float = 0.4

    # Frozen ANN layers behind the cached features.
    model_name: str = "dino_v3_l"
    model_source: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    img_size: int = 224
    pooling: str = "mean"
    layer_names: tuple = DEFAULT_LAYERS
    trust_remote_code: bool = True
    attn_implementation: str | None = "sdpa"

    # Split and optimization, mirroring run_tvsd_noise_regularization.py.
    n_test_images: int = N_TEST_IMAGES
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
    noise_ceiling_resamples: int = 40

    device: str = "auto"
    overwrite_cache: bool = False
    smoke_test: bool = False


"""
parse_args
Parse command-line overrides into the Triple-N baseline configuration.

OUTPUT:
    - cfg: Cfg -> file selection, target window, unit filter, and optimization
"""
def parse_args() -> Cfg:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the BaselineModel on Triple-N single-unit firing rates and "
            "score it against the same noise ceiling used for TVSD MUA."
        )
    )
    parser.add_argument("--env", default=Cfg.env, choices=list(config))
    parser.add_argument("--triple_n_dir", default=Cfg.triple_n_dir)
    parser.add_argument("--good_unit_file", default=Cfg.good_unit_file)
    parser.add_argument("--processed_file", default=Cfg.processed_file)
    parser.add_argument("--feature_archive", default=Cfg.feature_archive)
    parser.add_argument("--output_dir", default=Cfg.output_dir)
    parser.add_argument("--time_start_ms", type=float, default=Cfg.time_start_ms)
    parser.add_argument("--time_end_ms", type=float, default=Cfg.time_end_ms)
    parser.add_argument("--target_fs", type=int, default=Cfg.target_fs)
    parser.add_argument(
        "--response_onset_ms", type=float, default=Cfg.response_onset_ms
    )
    parser.add_argument("--subtract_baseline", action="store_true")
    parser.add_argument(
        "--unit_types",
        default=Cfg.unit_types,
        help="Comma-separated BombCell codes, e.g. 1 or 1,2.",
    )
    parser.add_argument(
        "--min_reliability", type=float, default=Cfg.min_reliability
    )
    parser.add_argument("--epochs", type=int, default=Cfg.epochs)
    parser.add_argument("--batch_size", type=int, default=Cfg.batch_size)
    parser.add_argument("--learning_rate", type=float, default=Cfg.learning_rate)
    parser.add_argument("--random_seed", type=int, default=Cfg.random_seed)
    parser.add_argument(
        "--noise_ceiling_resamples",
        type=int,
        default=Cfg.noise_ceiling_resamples,
    )
    parser.add_argument("--device", default=Cfg.device)
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    arguments = vars(parser.parse_args())
    return Cfg(**arguments)
# EOF


"""
resolve_session_files
Locate the GoodUnit, Processed, and feature files for the requested session.

INPUT:
    - cfg: Cfg -> explicit file names, empty for automatic discovery
    - triple_n_dir: Path -> directory holding the downloaded session files

OUTPUT:
    - good_unit_path: Path -> GoodUnit_*.mat for this session
    - processed_path: Path | None -> matching Processed_ses*.mat, if present
"""
def resolve_session_files(cfg, triple_n_dir):
    if cfg.good_unit_file:
        good_unit_path = Path(cfg.good_unit_file).expanduser()
        if not good_unit_path.is_absolute():
            good_unit_path = triple_n_dir / good_unit_path
        # end if a bare file name was given
    else:
        candidates = sorted(triple_n_dir.glob("GoodUnit_*.mat"))
        if not candidates:
            raise FileNotFoundError(
                f"No GoodUnit_*.mat in {triple_n_dir}. Run "
                "download_triple_n_sessions.py first."
            )
        # end if no session was downloaded
        good_unit_path = candidates[0]
    # end if the session file was named explicitly
    if not good_unit_path.is_file():
        raise FileNotFoundError(f"Missing GoodUnit file {good_unit_path}.")
    # end if the named session file is absent

    processed_path = None
    if cfg.processed_file:
        processed_path = Path(cfg.processed_file).expanduser()
        if not processed_path.is_absolute():
            processed_path = triple_n_dir / processed_path
        # end if a bare file name was given
    else:
        # GoodUnit files carry the YYMMDD stamp; Processed files repeat it.
        session_stamp = good_unit_path.name.split("_")[1]
        matches = sorted(triple_n_dir.glob(f"Processed_ses*{session_stamp}*.mat"))
        processed_path = matches[0] if matches else None
    # end if the summary file was named explicitly
    return good_unit_path, processed_path
# EOF


"""
train_baseline
Run the shared training loop and keep the best validation checkpoint.

INPUT:
    - model: BaselineModel -> decoder to optimize
    - loaders: dict -> train and validation DataLoaders
    - cfg: Cfg -> optimization settings
    - device: torch.device -> compute device

OUTPUT:
    - history: dict -> per-epoch training and validation MSE
    - best_epoch: int -> epoch of the restored checkpoint
"""
def train_baseline(model, loaders, cfg, device):
    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    optimizer.zero_grad(set_to_none=True)
    cost_function = neural_activity_timebin_mse_loss

    best_validation_loss = test_step(
        model, loaders["validation"], cost_function, True, device
    )
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    training_losses, validation_losses = [], []

    for epoch in range(1, cfg.epochs + 1):
        training_loss = training_step(
            model, loaders["train"], optimizer, cost_function, True, device
        )
        validation_loss = test_step(
            model, loaders["validation"], cost_function, True, device
        )
        training_losses.append(training_loss)
        validation_losses.append(validation_loss)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        # end if this epoch is the best validation checkpoint
        print(
            f"    epoch {epoch:03d}/{cfg.epochs:03d} | train "
            f"{training_loss:.6f} | validation {validation_loss:.6f}"
        )
    # end for optimization epoch

    model.load_state_dict(best_state)
    return (
        {
            "train_mse": training_losses,
            "validation_mse": validation_losses,
            "best_validation_mse": best_validation_loss,
        },
        best_epoch,
    )
# EOF


def main():
    cfg = parse_args()
    if cfg.n_test_images != N_TEST_IMAGES:
        # score_predictions aggregates against the module-level constant, so a
        # different pool size would silently mis-score the test split.
        raise SystemExit(
            f"n_test_images must stay {N_TEST_IMAGES} to reuse "
            "tvsd_experiments.score_predictions."
        )
    # end if the held-out pool size was changed
    if cfg.smoke_test:
        cfg.epochs = min(cfg.epochs, 2)
        cfg.noise_ceiling_resamples = 4
    # end if a short run was requested

    paths = load_project_paths(cfg)
    device = resolve_device(cfg.device)
    triple_n_dir = Path(
        cfg.triple_n_dir or Path(paths["data_path"]) / "data" / "triple_n"
    ).expanduser()
    output_dir = Path(
        cfg.output_dir or PROJECT_ROOT / "results" / "triple_n_baseline"
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    good_unit_path, processed_path = resolve_session_files(cfg, triple_n_dir)
    print(f"session {good_unit_path.name}")

    # --- targets: firing rates in spikes per second ---
    rate_cache_path = triple_n_dir / (
        f"{good_unit_path.stem}_{cfg.time_start_ms:g}-{cfg.time_end_ms:g}ms_"
        f"{cfg.target_fs}Hz_rate.npy"
    )
    targets = preprocess_triple_n_firing_rate_targets(
        good_unit_path,
        rate_cache_path,
        time_start_ms=cfg.time_start_ms,
        time_end_ms=cfg.time_end_ms,
        target_fs=cfg.target_fs,
        subtract_baseline=cfg.subtract_baseline,
        overwrite=cfg.overwrite_cache,
    )
    trial_image_ids, _, n_units, _ = load_triple_n_good_unit_metadata(
        good_unit_path
    )
    print(
        f"  {targets.shape[0]:,} trials | {targets.shape[1]} time bins | "
        f"{n_units} units before filtering"
    )

    # --- unit selection from the BombCell labels ---
    unit_indices = None
    if processed_path is not None:
        summary = load_triple_n_processed(processed_path)
        requested_types = tuple(
            int(code) for code in cfg.unit_types.split(",") if code.strip()
        )
        unit_indices = select_triple_n_units(
            summary,
            unit_types=requested_types,
            min_reliability=cfg.min_reliability,
        )
        label_counts = {
            label: int(np.sum(summary["unit_type_labels"] == label))
            for label in np.unique(summary["unit_type_labels"])
        }
        print(
            f"  {processed_path.name}: {label_counts} | keeping "
            f"{len(unit_indices)} units of type {requested_types} with "
            f"reliability > {cfg.min_reliability}"
        )
        if len(unit_indices) == 0:
            raise SystemExit(
                "No units survived the filter. Lower --min_reliability or "
                "widen --unit_types."
            )
        # end if the selection is empty
    else:
        print(
            "  no Processed file found; fitting all units without quality "
            "filtering. Download the matching Processed_ses*.mat to restrict "
            "the fit to sorted single units."
        )
    # end if BombCell labels are available

    # --- stimulus features ---
    feature_path = Path(cfg.feature_archive).expanduser()
    if not feature_path.is_absolute():
        feature_path = Path(paths["data_path"]) / "models" / feature_path
    # end if a bare archive name was given
    if not feature_path.is_file():
        raise FileNotFoundError(
            f"Missing stimulus feature archive {feature_path}. It must hold "
            "'stimulus_features' [stimuli, layers, embedding] ordered by "
            "one-based Triple-N stimulus id."
        )
    # end if cached features are absent
    with np.load(feature_path) as feature_archive:
        stimulus_features = feature_archive["stimulus_features"].astype(
            np.float32, copy=False
        )
        layer_names = [str(name) for name in feature_archive["layer_names"]]
    # end with stimulus feature archive
    print(
        f"  features {stimulus_features.shape} from {feature_path.name} "
        f"({len(layer_names)} layers)"
    )

    # --- image-level split and datasets ---
    metadata, train_image_ids, test_image_ids = build_triple_n_image_metadata(
        trial_image_ids,
        n_test_images=cfg.n_test_images,
        random_seed=cfg.random_seed,
    )
    (
        datasets,
        indices,
        (unit_mean, unit_scale),
        (train_features, test_features),
    ) = build_triple_n_datasets(
        targets,
        metadata,
        stimulus_features,
        train_image_ids,
        test_image_ids,
        validation_fraction=cfg.validation_fraction,
        random_seed=cfg.random_seed,
        unit_indices=unit_indices,
    )

    loader_generator = torch.Generator().manual_seed(cfg.random_seed)
    loaders = {
        "train": torch.utils.data.DataLoader(
            datasets["train"],
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            generator=loader_generator,
        ),
        "validation": torch.utils.data.DataLoader(
            datasets["validation"],
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        ),
        "test": torch.utils.data.DataLoader(
            datasets["test"],
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        ),
    }

    # Fancy-index the memmap directly so only the retained units are read.
    selected_targets = np.asarray(
        targets[:, :, unit_indices] if unit_indices is not None else targets,
        dtype=np.float32,
    )
    n_timepoints, n_neurons = selected_targets.shape[1], selected_targets.shape[2]
    print(
        f"device {device} | fit {len(indices['train']):,} | validation "
        f"{len(indices['validation']):,} | test {len(indices['test']):,} "
        f"trials | target [{n_timepoints} time x {n_neurons} units]"
    )

    fit_targets = standardize_targets(
        selected_targets[indices["train"]], unit_mean, unit_scale
    )
    validation_targets = standardize_targets(
        selected_targets[indices["validation"]], unit_mean, unit_scale
    )
    test_targets = standardize_targets(
        selected_targets[indices["test"]], unit_mean, unit_scale
    )
    fit_target_mean = fit_targets.mean(axis=0)
    test_image_positions = metadata[indices["test"], 2] - 1

    # The ceiling is measured on the same standardized targets the decoder is
    # scored against, so the fraction of ceiling is comparable to TVSD's.
    ceiling = split_half_reliability(
        test_targets,
        test_image_positions,
        cfg.n_test_images,
        reducer="mean",
        n_resamples=cfg.noise_ceiling_resamples,
        seed=cfg.random_seed,
    )
    onset_bin = int(
        round((cfg.response_onset_ms - cfg.time_start_ms) * cfg.target_fs / 1000)
    )
    response_slice = slice(onset_bin, n_timepoints)
    print(
        f"  target reliability over the driven window: "
        f"{np.nanmean(ceiling[response_slice]):.4f}"
    )

    records = []

    # --- ridge reference on the concatenated layers ---
    ridge_predictions, ridge_validation_mse, ridge_alpha, n_coefficients = (
        fit_ridge_reference(
            cfg,
            train_features,
            test_features,
            metadata,
            indices,
            fit_targets,
            validation_targets,
        )
    )
    ridge_metrics, _ = score_predictions(
        ridge_predictions,
        test_targets,
        test_image_positions,
        "mean",
        fit_target_mean,
        ceiling,
        response_slice,
    )
    records.append(
        {
            "variant": "ridge",
            "validation_mse": round(ridge_validation_mse, 5),
            "best_epoch": None,
            "trainable_parameters": n_coefficients,
            "ridge_alpha": ridge_alpha,
            **ridge_metrics,
        }
    )
    print(
        f"  ridge alpha {ridge_alpha:g} | validation MSE "
        f"{ridge_validation_mse:.5f} | fraction of ceiling "
        f"{ridge_metrics['fraction_of_ceiling']}"
    )

    # --- BaselineModel on the frozen encoder ---
    from useful_stuff.image_processing.computational_models import imgANN

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
    torch.manual_seed(cfg.random_seed)
    model = BaselineModel(
        encoder=encoder,
        layers=layer_names,
        temporal_embedding_dim=cfg.temporal_embedding_dim,
        value_dim=cfg.value_dim,
        n_timepoints=n_timepoints,
        temporal_compression_ratio=1,
        n_neurons=n_neurons,
        mlp_hidden_dim=cfg.mlp_hidden_dim,
        dropout=cfg.dropout,
        attention_granularity=cfg.attention_granularity,
    ).to(device)

    print("training baseline")
    history, best_epoch = train_baseline(model, loaders, cfg, device)
    trial_predictions, trial_targets, mean_attention = predict_test_trials(
        model, loaders["test"], device
    )
    baseline_metrics, channel_time_correlations = score_predictions(
        trial_predictions,
        trial_targets,
        test_image_positions,
        "mean",
        fit_target_mean,
        ceiling,
        response_slice,
    )
    records.append(
        {
            "variant": "baseline",
            "validation_mse": round(history["best_validation_mse"], 5),
            "best_epoch": best_epoch,
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in model.get_trainable_parameters()
            ),
            "ridge_alpha": None,
            **baseline_metrics,
        }
    )
    print(
        f"  baseline | best epoch {best_epoch} | fraction of ceiling "
        f"{baseline_metrics['fraction_of_ceiling']}"
    )

    # --- figure and artefacts ---
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(
        range(1, len(history["train_mse"]) + 1),
        history["train_mse"],
        label="train",
        color=VARIANT_COLORS["baseline"],
    )
    axes[0].plot(
        range(1, len(history["validation_mse"]) + 1),
        history["validation_mse"],
        label="validation",
        color=VARIANT_COLORS["baseline"],
        linestyle="--",
    )
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("standardized MSE")
    axes[0].set_title("BaselineModel on Triple-N firing rates")
    axes[0].legend(frameon=False, fontsize=8)

    time_axis_ms = cfg.time_start_ms + (np.arange(n_timepoints) + 0.5) * (
        1000.0 / cfg.target_fs
    )
    axes[1].plot(
        time_axis_ms,
        np.nanmean(ceiling, axis=1),
        color="#666666",
        label="noise ceiling",
    )
    axes[1].plot(
        time_axis_ms,
        np.nanmean(channel_time_correlations, axis=1),
        color=VARIANT_COLORS["baseline"],
        label="baseline stim_r",
    )
    axes[1].set_xlabel("time from stimulus onset (ms)")
    axes[1].set_ylabel("mean stim_r over units")
    axes[1].set_title("Prediction against the ceiling")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    # end for panel
    figure.tight_layout()
    figure.savefig(output_dir / "triple_n_baseline.png", dpi=160)
    plt.close(figure)

    with open(output_dir / "config.json", "w") as config_file:
        json.dump(asdict(cfg) | {"session": good_unit_path.name}, config_file, indent=2)
    # end with config file
    with open(output_dir / "results.json", "w") as results_file:
        json.dump(records, results_file, indent=2)
    # end with results file
    np.savez(
        output_dir / "time_courses.npz",
        ceiling=ceiling,
        baseline_stim_r=channel_time_correlations,
        mean_attention=mean_attention,
        time_axis_ms=time_axis_ms,
    )
    print(f"\nwrote {output_dir}")
# EOF


if __name__ == "__main__":
    main()
