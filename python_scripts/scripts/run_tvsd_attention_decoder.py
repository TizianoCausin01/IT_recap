"""
Decode three DINOv3 depths from the IT population with per-depth temporal
attention, and test the whole thing against ridge.

The question is not whether a recurrent decoder can read IT -- ridge already
can -- but whether *when* in the response a depth is readable differs by depth.
A GRU keeps every hidden state, each target depth owns a free query and pools
that sequence with its own softmax over time, and the three attention curves
are the artifact: shallow DINOv3 recoverable from early bins and deep DINOv3
from late ones would be the decoding-side counterpart of the early/late
handover reported for V4 to IT communication.

Nothing about that is worth reading unless the model earns its place, so the
sweep is run against four references rather than one:

    ridge_full      one linear map from the flattened response, penalty chosen
                    on validation over a wide grid and checked for grid edges
    ridge_mean      the same map from the time-averaged response, which says
                    how much the time axis carries at all
    mean_pool       the same GRU with a single shared pooled code for all three
                    heads -- if it ties, the per-depth attention is decoration
    independent     three unshared GRUs -- if it ties, the shared encoder is

plus a time-shuffled control that destroys the order of the response.

Deviations from the specification, both forced by the cache. The IT array has
320 channels, not ~120. And the cached response covers 0-200 ms, so 20 ms bins
give T=10 rather than the 20-30 assumed; --timebin_ms 10 doubles it at the cost
of noisier bins.
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python_scripts" / "src"))

from IT_recap.neural_decoding import (  # noqa: E402
    attention_entropy,
    count_trainable_parameters,
    evaluate_neural_decoder,
    fit_ridge_decoder,
    make_decoding_loader,
    make_ridge_predictor,
    prepare_decoding_data,
    repetition_ceiling,
    score_decoding,
    temporal_centroid,
    train_neural_decoder,
)
from IT_recap.tvsd_experiments import (  # noqa: E402
    load_project_paths,
    resolve_device,
)
from model_classes.attention_decoder_models import (  # noqa: E402
    build_neural_decoder,
)


# Shallow, mid, and deep depths of the cached DINOv3-L archive.
DEFAULT_LAYERS = [
    "layer.3.mlp.down_proj",
    "layer.13.mlp.down_proj",
    "layer.20.mlp.down_proj",
]
SHORT_LAYER_NAMES = {
    "layer.3.mlp.down_proj": "dino_l3",
    "layer.13.mlp.down_proj": "dino_l13",
    "layer.16.mlp.down_proj": "dino_l16",
    "layer.20.mlp.down_proj": "dino_l20",
}

# The ridge grid is deliberately wide: an under-regularized baseline has beaten
# every architecture in this project before, and a penalty selected at a grid
# edge is a property of the grid rather than of the data.
RIDGE_ALPHAS = np.logspace(0.0, 12.0, 49)


@dataclass
class Cfg:
    env: str | None = None
    mua_file_name: str = "f_THINGS_MUA_trials.mat"
    feature_archive_name: str = "tvsd_monkeyF_dino_v3_l_224_features.npz"
    output_dir: str | None = None

    # Neural input: which area, which window, and how coarsely binned.
    area: str = "IT"
    target_fs: int = 100
    time_start_ms: float = 0.0
    time_end_ms: float = 200.0
    window_start_ms: float = 0.0
    window_end_ms: float = 200.0
    timebin_ms: float = 20.0
    input_pca_rank: int = 0
    shuffle_time: bool = False

    # Targets: three DINOv3 depths, each reduced by its own PCA.
    layer_names: list[str] = field(default_factory=lambda: list(DEFAULT_LAYERS))
    target_ranks: list[int] = field(default_factory=lambda: [32, 32, 32])
    target_scaling: str = "layer_rms"

    # Decoder capacity. These defaults sit far below usual recurrent widths on
    # purpose: 22k single-repetition samples do not support more.
    variant: str = "attention"
    hidden_dim: int = 16
    n_layers: int = 1
    dropout: float = 0.1
    recurrent_dropout: float = 0.1
    bottleneck_dim: int = 0
    head_hidden_dim: int = 0

    # Optimization.
    learning_rate: float = 3e-3
    weight_decay: float = 1e-3
    l2_lambda: float = 0.0
    l2_scope: str = "readout"
    batch_size: int = 256
    epochs: int = 200
    patience: int = 20
    min_improvement: float = 1e-5
    gradient_clip: float = 1.0
    loss_weights: list[float] = field(default_factory=list)

    # Splits and evaluation.
    validation_fraction: float = 0.1
    heldout_fraction: float = 0.1
    ceiling_resamples: int = 20
    max_attention_panels: int = 4
    random_seed: int = 0
    seeds: int = 1

    variants: str = "all"
    num_workers: int = 0
    device: str = "auto"
    smoke_test: bool = False


# Everything the sweep varies, as overrides on the reference configuration.
VARIANTS = {
    # Reference preparation: full channels, 32 PCs per depth, true time order.
    "ridge_full": {"model_family": "ridge", "ridge_input": "full"},
    "ridge_mean": {"model_family": "ridge", "ridge_input": "time_mean"},
    "ridge_single_bin": {"model_family": "ridge", "ridge_input": "best_bin"},
    "attention": {},
    "attention_h8": {"hidden_dim": 8},
    "attention_h32": {"hidden_dim": 32},
    "attention_h64": {"hidden_dim": 64},
    # Beyond the capacity range the design calls for, kept as the
    # diagnostic that says whether a small H is what costs the model
    # against a 307k-coefficient ridge, rather than the recurrence itself.
    "attention_h128": {"hidden_dim": 128},
    "attention_2layer": {"n_layers": 2, "dropout": 0.3, "recurrent_dropout": 0.3},
    "attention_dropout0": {"dropout": 0.0, "recurrent_dropout": 0.0},
    "attention_dropout3": {"dropout": 0.3, "recurrent_dropout": 0.3},
    "attention_dropout5": {"dropout": 0.5, "recurrent_dropout": 0.5},
    "attention_wd0": {"weight_decay": 0.0},
    "attention_wd1e2": {"weight_decay": 1e-2},
    "attention_bottleneck": {"bottleneck_dim": 64},
    "attention_mlp_head": {"head_hidden_dim": 64},
    # Ridge's own penalty, converted to a mean-MSE loss with
    # ridge_equivalent_l2_lambda, applied explicitly instead of through AdamW's
    # decoupled decay -- the move that closed this gap in the encoding
    # direction. Centred on the anchor, one decade either side.
    "attention_h64_l2": {
        "hidden_dim": 64, "weight_decay": 0.0, "l2_lambda": 1.0e-2,
    },
    "attention_h64_l2_lo": {
        "hidden_dim": 64, "weight_decay": 0.0, "l2_lambda": 1.0e-3,
    },
    "attention_h64_l2_hi": {
        "hidden_dim": 64, "weight_decay": 0.0, "l2_lambda": 1.0e-1,
    },
    "attention_h64_l2_vlo": {
        "hidden_dim": 64, "weight_decay": 0.0, "l2_lambda": 1.0e-4,
    },
    # The ablations again, at the setting that actually wins. Run at the weak
    # reference setting they compare three models whose attention is uniform,
    # which says nothing about attention; only here is the comparison live.
    "mean_pool_tuned": {
        "variant": "mean_pool", "hidden_dim": 64, "weight_decay": 0.0,
        "l2_lambda": 1.0e-3,
    },
    "last_state_tuned": {
        "variant": "last_state", "hidden_dim": 64, "weight_decay": 0.0,
        "l2_lambda": 1.0e-3,
    },
    "independent_tuned": {
        "variant": "independent", "hidden_dim": 64, "weight_decay": 0.0,
        "l2_lambda": 1.0e-3,
    },
    "attention_tuned_shuffled_time": {
        "hidden_dim": 64, "weight_decay": 0.0, "l2_lambda": 1.0e-3,
        "shuffle_time": True,
    },
    "mean_pool": {"variant": "mean_pool"},
    "last_state": {"variant": "last_state"},
    "independent": {"variant": "independent"},
    # Variants below change the preparation itself, so they are grouped by the
    # preparation they share -- rebuilding one costs a full pass over the cache.
    "attention_pca64": {"input_pca_rank": 64},
    "attention_pca128": {"input_pca_rank": 128},
    "attention_rank16": {"target_ranks": [16, 16, 16]},
    "attention_rank64": {"target_ranks": [64, 64, 64]},
    "attention_shuffled_time": {"shuffle_time": True},
    "ridge_shuffled_time": {
        "model_family": "ridge",
        "ridge_input": "full",
        "shuffle_time": True,
    },
}

# Variants whose data preparation differs, so they cannot share a cached prep.
PREP_FIELDS = (
    "input_pca_rank",
    "shuffle_time",
    "target_ranks",
    "target_scaling",
    "timebin_ms",
    "window_start_ms",
    "window_end_ms",
    "random_seed",
    "validation_fraction",
    "heldout_fraction",
)


"""
parse_args
Parse command-line overrides into the experiment configuration.

OUTPUT:
    - cfg: Cfg -> input, target, decoder, and optimization settings
"""
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    list_fields = {"layer_names": str, "target_ranks": int, "loss_weights": float}
    for field_name, field_definition in Cfg.__dataclass_fields__.items():
        argument_name = f"--{field_name}"
        if field_name in list_fields:
            default = field_definition.default_factory()
            parser.add_argument(
                argument_name,
                default=",".join(str(value) for value in default),
            )
            continue
        # end if the field is a comma-separated list
        default = field_definition.default
        if isinstance(default, bool):
            parser.add_argument(argument_name, action="store_true")
        elif default is None:
            parser.add_argument(argument_name, default=default)
        else:
            parser.add_argument(argument_name, type=type(default), default=default)
        # end if boolean, optional, or typed argument
    # end for configuration field

    arguments = vars(parser.parse_args())
    for field_name, element_type in list_fields.items():
        arguments[field_name] = [
            element_type(value.strip())
            for value in str(arguments[field_name]).split(",")
            if value.strip()
        ]
    # end for list-valued field
    return Cfg(**arguments)
# EOF


"""
prep_signature
Identify the data preparation a variant needs, so variants that share one do.

INPUT:
    - cfg: Cfg -> a variant's configuration

OUTPUT:
    - signature: tuple -> hashable summary of every preparation-relevant field
"""
def prep_signature(cfg):
    return tuple(
        tuple(getattr(cfg, name)) if isinstance(getattr(cfg, name), list)
        else getattr(cfg, name)
        for name in PREP_FIELDS
    )
# EOF


"""
run_ridge_variant
Fit and score the linear reference on one prepared dataset.

The time-averaged input is the same map with the time axis collapsed first, so
the gap between the two ridges measures what the response's time course adds to
a purely linear decoder -- the number any recurrent model has to beat before
its temporal machinery means anything.

INPUT:
    - cfg: Cfg -> the variant's configuration
    - data: dict -> output of prepare_decoding_data
    - layer_names: list -> short display names of the target layers

OUTPUT:
    - row: dict -> scores and diagnostics for the results table
    - extras: dict -> the repetition-ceiling record
"""
def run_ridge_variant(cfg, data, layer_names):
    splits, repeated = data["splits"], data["repeated"]

    """
    shape_input
    Present one split's full-channel response in the ridge's input form.
    """
    def shape_input(sequences):
        if cfg.ridge_input == "time_mean":
            # Keep the [presentations, time, channels] layout with T=1 so the
            # flattening, the predictor, and the ceiling code stay identical.
            return np.asarray(sequences).mean(axis=1, keepdims=True)
        # end if the time axis is collapsed first
        return sequences
    # end def shape_input

    per_bin_r2, best_bin = None, None
    if cfg.ridge_input == "best_bin":
        # One ridge per time bin. The winner is the "best single bin" reference
        # an attention pattern that has collapsed onto one bin has to beat, and
        # the whole curve says when the response is linearly decodable at all --
        # the quantity the learned attention should be compared against.
        bin_fits, per_bin_r2 = [], []
        for bin_index in range(splits["train"]["neural_full"].shape[1]):
            bin_slice = slice(bin_index, bin_index + 1)
            bin_fit = fit_ridge_decoder(
                splits["train"]["neural_full"][:, bin_slice],
                splits["train"]["targets"],
                {
                    name: splits[name]["neural_full"][:, bin_slice]
                    for name in ("validation", "heldout")
                },
                splits["validation"]["targets"],
                RIDGE_ALPHAS,
                data["target_slices"],
            )
            bin_fits.append(bin_fit)
            per_bin_r2.append(
                score_decoding(
                    bin_fit["predictions"]["heldout"],
                    splits["heldout"]["targets"],
                    data["target_slices"],
                    layer_names,
                )
            )
        # end for neural time bin
        validation_mse = [
            float(
                np.mean(
                    np.square(
                        fit["predictions"]["validation"]
                        - splits["validation"]["targets"]
                    )
                )
            )
            for fit in bin_fits
        ]
        best_bin = int(np.argmin(validation_mse))
        ridge_fit = bin_fits[best_bin]
        bin_slice = slice(best_bin, best_bin + 1)

        """
        shape_input
        Restrict the response to the single bin selected on validation.
        """
        def shape_input(sequences):
            return np.asarray(sequences)[:, bin_slice]
        # end def shape_input
    else:
        ridge_fit = fit_ridge_decoder(
            shape_input(splits["train"]["neural_full"]),
            splits["train"]["targets"],
            {
                name: shape_input(splits[name]["neural_full"])
                for name in ("validation", "heldout")
            },
            splits["validation"]["targets"],
            RIDGE_ALPHAS,
            data["target_slices"],
        )
    # end if the reference searches over single bins
    predict = make_ridge_predictor(ridge_fit)

    heldout_scores = score_decoding(
        ridge_fit["predictions"]["heldout"],
        splits["heldout"]["targets"],
        data["target_slices"],
        layer_names,
    )
    validation_scores = score_decoding(
        ridge_fit["predictions"]["validation"],
        splits["validation"]["targets"],
        data["target_slices"],
        layer_names,
    )
    ceiling = repetition_ceiling(
        lambda sequences: predict(shape_input(sequences)),
        repeated["neural_full"],
        repeated["image_ids"],
        repeated["image_targets"],
        data["target_slices"],
        layer_names,
        n_resamples=cfg.ceiling_resamples,
        seed=cfg.random_seed,
    )

    row = {
        "heldout": heldout_scores,
        "validation": validation_scores,
        "alpha": ridge_fit["alpha"],
        "alpha_at_grid_edge": ridge_fit["alpha_at_grid_edge"],
        "trainable_parameters": ridge_fit["n_coefficients"],
        "best_epoch": 0,
        "epochs_run": 0,
        "train_seconds": 0.0,
        "attention_entropy": None,
        "best_bin": best_bin,
    }
    return row, {
        "ceiling": ceiling,
        "attention": None,
        "per_bin_r2": per_bin_r2,
    }
# EOF


"""
run_decoder_variant
Train and score one recurrent decoder on a prepared dataset.

INPUT:
    - cfg: Cfg -> the variant's configuration
    - data: dict -> output of prepare_decoding_data
    - layer_names: list -> short display names of the target layers
    - device: torch.device -> compute device
    - seed: int -> initialization and shuffling seed

OUTPUT:
    - row: dict -> scores and diagnostics for the results table
    - extras: dict -> training history, mean attention, and the ceiling record
"""
def run_decoder_variant(cfg, data, layer_names, device, seed):
    torch.manual_seed(seed)
    splits, repeated = data["splits"], data["repeated"]
    model = build_neural_decoder(
        cfg.variant,
        data["n_channels"],
        data["n_timepoints"],
        data["target_dims"],
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
        recurrent_dropout=cfg.recurrent_dropout,
        bottleneck_dim=cfg.bottleneck_dim or None,
        head_hidden_dim=cfg.head_hidden_dim or None,
    ).to(device)

    loaders = {
        name: make_decoding_loader(
            splits[name]["neural"],
            splits[name]["targets"],
            cfg,
            shuffle=(name == "train"),
        )
        for name in ("train", "validation", "heldout")
    }
    history, summary = train_neural_decoder(
        model, loaders, cfg, data["target_slices"], device, verbose=False
    )

    heldout_predictions, heldout_targets, heldout_attention = (
        evaluate_neural_decoder(model, loaders["heldout"], device)
    )
    validation_predictions, validation_targets, _ = evaluate_neural_decoder(
        model, loaders["validation"], device
    )
    heldout_scores = score_decoding(
        heldout_predictions, heldout_targets, data["target_slices"], layer_names
    )
    validation_scores = score_decoding(
        validation_predictions,
        validation_targets,
        data["target_slices"],
        layer_names,
    )

    """
    predict
    Score the trained decoder on repetition-averaged inputs, in batches.
    """
    def predict(sequences):
        model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(
                np.ascontiguousarray(sequences, dtype=np.float32)
            ).to(device)
            predictions, _ = model(tensor)
        # end with inference mode
        return predictions.cpu().numpy()
    # end def predict

    ceiling = repetition_ceiling(
        predict,
        repeated["neural"],
        repeated["image_ids"],
        repeated["image_targets"],
        data["target_slices"],
        layer_names,
        n_resamples=cfg.ceiling_resamples,
        seed=cfg.random_seed,
    )

    mean_attention, entropies = None, None
    if heldout_attention is not None:
        mean_attention = heldout_attention.mean(axis=0)
        entropies = attention_entropy(heldout_attention).tolist()
    # end if the decoder pools by attention

    row = {
        "heldout": heldout_scores,
        "validation": validation_scores,
        "alpha": None,
        "alpha_at_grid_edge": False,
        "trainable_parameters": count_trainable_parameters(model),
        "best_epoch": summary["best_epoch"],
        "epochs_run": summary["epochs_run"],
        "train_seconds": summary["train_seconds"],
        "attention_entropy": entropies,
        "best_bin": None,
    }
    extras = {
        "ceiling": ceiling,
        "history": history,
        "attention": None if mean_attention is None else mean_attention.tolist(),
    }
    return row, extras
# EOF


"""
plot_attention_curves
Overlay each target depth's mean temporal attention on the neural time axis,
above the single-bin linear decodability of the same axis.

This is the interpretability artifact the architecture exists for. The top row
is one curve per DINOv3 depth, read against the uniform-attention line so that
a flat pattern is visible as flat. The bottom row is the reference the curves
have to be read against: the held-out R2 of a ridge fitted on one time bin at a
time. Attention peaking where single-bin decoding peaks means the model found
the structure; attention peaking elsewhere, or nowhere, means it did not.

INPUT:
    - attention_by_variant: dict -> variant name to [targets, time] weights
    - per_bin_r2: list | None -> per-bin held-out score dicts, or None
    - bin_edges_ms: np.ndarray -> edges of the neural time bins
    - layer_names: list -> short display names of the target layers
    - output_path: Path -> destination PNG

OUTPUT:
    - None
"""
def plot_attention_curves(
    attention_by_variant, per_bin_r2, bin_edges_ms, layer_names, output_path
):
    if not attention_by_variant:
        return
    # end if no variant pooled by attention
    bin_centers = 0.5 * (bin_edges_ms[:-1] + bin_edges_ms[1:])
    n_panels = len(attention_by_variant)
    n_rows = 2 if per_bin_r2 else 1
    figure, axes = plt.subplots(
        n_rows,
        n_panels,
        figsize=(4.2 * n_panels, 3.2 * n_rows),
        squeeze=False,
        sharey="row",
    )
    for column, (variant_name, attention) in enumerate(
        sorted(attention_by_variant.items())
    ):
        axis = axes[0][column]
        attention = np.asarray(attention)
        for layer_index, layer_name in enumerate(layer_names):
            axis.plot(
                bin_centers,
                attention[layer_index],
                marker="o",
                markersize=3,
                label=layer_name,
            )
        # end for target depth
        # Uniform attention is the null the curves have to separate from.
        axis.axhline(
            1.0 / attention.shape[1], color="0.6", linestyle=":", linewidth=1
        )
        axis.set_title(variant_name, fontsize=10)
        if per_bin_r2:
            reference = axes[1][column]
            for layer_name in layer_names:
                reference.plot(
                    bin_centers,
                    [scores[layer_name]["r2"] for scores in per_bin_r2],
                    marker="s",
                    markersize=3,
                    label=layer_name,
                )
            # end for target depth
            reference.axhline(0.0, color="0.6", linestyle=":", linewidth=1)
            reference.set_xlabel("time from stimulus onset (ms)")
        else:
            axis.set_xlabel("time from stimulus onset (ms)")
        # end if the single-bin reference is available
    # end for attention variant
    axes[0][0].set_ylabel("mean attention weight")
    axes[0][-1].legend(frameon=False, fontsize=8)
    if per_bin_r2:
        axes[1][0].set_ylabel("single-bin ridge $R^2$ (held out)")
    # end if the reference row was drawn
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
# EOF


"""
format_row
Render one result row for the summary table.

INPUT:
    - name: str -> variant name
    - row: dict -> scores and diagnostics
    - layer_names: list -> short display names of the target layers

OUTPUT:
    - text: str -> one formatted table line
"""
def format_row(name, row, layer_names):
    per_layer = " ".join(
        f"{row['heldout'][layer_name]['r2']:>8.4f}" for layer_name in layer_names
    )
    ceiling_fraction = row.get("mean_fraction_of_ceiling", float("nan"))
    edge_flag = "*" if row["alpha_at_grid_edge"] else " "
    return (
        f"{name:<24}{row['heldout']['mean_r2']:>9.4f}"
        f"{row['heldout']['mean_component_r']:>9.4f}"
        f"{ceiling_fraction:>10.3f}  {per_layer}"
        f"{row['trainable_parameters']:>11,}{edge_flag}"
    )
# EOF


def main():
    cfg = parse_args()
    paths = load_project_paths(cfg)
    device = resolve_device(cfg.device)
    layer_names = [
        SHORT_LAYER_NAMES.get(name, name) for name in cfg.layer_names
    ]

    output_dir = Path(
        cfg.output_dir
        or PROJECT_ROOT / "results" / "tvsd_attention_decoder_dino_v3_l"
    )
    variant_dir = output_dir / "variants"
    variant_dir.mkdir(parents=True, exist_ok=True)

    requested = (
        list(VARIANTS)
        if cfg.variants == "all"
        else [name.strip() for name in cfg.variants.split(",") if name.strip()]
    )
    unknown = [name for name in requested if name not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants {unknown}; choose from {list(VARIANTS)}.")
    # end if an unknown variant was requested

    print(
        f"decoding {len(cfg.layer_names)} DINOv3 depths from {cfg.area} | "
        f"{cfg.timebin_ms:g} ms bins over "
        f"{cfg.window_start_ms:g}-{cfg.window_end_ms:g} ms | device {device}"
    )

    # Variants sharing a preparation share the prepared arrays; the cache holds
    # one at a time because each is a few hundred megabytes.
    prepared_cache = {}
    result_rows, extras_by_variant = {}, {}
    for name in requested:
        overrides = dict(VARIANTS[name])
        model_family = overrides.pop("model_family", "decoder")
        ridge_input = overrides.pop("ridge_input", "full")
        variant_cfg = replace(cfg, **overrides)
        variant_cfg.ridge_input = ridge_input

        for seed in range(cfg.random_seed, cfg.random_seed + cfg.seeds):
            run_name = name if cfg.seeds == 1 else f"{name}_seed{seed}"
            record_path = variant_dir / f"{run_name}.json"
            if record_path.is_file():
                with open(record_path, "r") as record_file:
                    record = json.load(record_file)
                # end with stored variant record
                result_rows[run_name] = record["row"]
                extras_by_variant[run_name] = record["extras"]
                print(
                    f"reusing {run_name}: heldout mean R2 "
                    f"{record['row']['heldout']['mean_r2']:.4f}"
                )
                continue
            # end if this variant has already been run

            signature = prep_signature(variant_cfg)
            if signature not in prepared_cache:
                # Only one preparation is kept alive at a time.
                prepared_cache.clear()
                print(f"preparing data for {run_name} ...")
                prepared_cache[signature] = prepare_decoding_data(
                    variant_cfg, paths
                )
                prepared = prepared_cache[signature]
                print(
                    f"  train {len(prepared['splits']['train']['neural']):,} | "
                    f"validation {len(prepared['splits']['validation']['neural']):,} | "
                    f"heldout {len(prepared['splits']['heldout']['neural']):,} | "
                    f"input [{prepared['n_timepoints']}, {prepared['n_channels']}] | "
                    f"targets {prepared['target_dims']}"
                )
            # end if this preparation is not cached
            data = prepared_cache[signature]

            print(f"running {run_name} ...")
            if model_family == "ridge":
                row, extras = run_ridge_variant(variant_cfg, data, layer_names)
            else:
                row, extras = run_decoder_variant(
                    replace(variant_cfg, random_seed=seed),
                    data,
                    layer_names,
                    device,
                    seed,
                )
            # end if the variant is the linear reference

            row["mean_fraction_of_ceiling"] = float(
                np.nanmean(
                    [
                        extras["ceiling"][layer_name]["fraction_of_ceiling"]
                        for layer_name in layer_names
                    ]
                )
            )
            row["model"] = run_name
            result_rows[run_name] = row
            extras_by_variant[run_name] = extras
            with open(record_path, "w") as record_file:
                json.dump(
                    {"row": row, "extras": extras, "overrides": VARIANTS[name]},
                    record_file,
                    indent=2,
                )
            # end with saved variant record
            print(
                f"  heldout mean R2 {row['heldout']['mean_r2']:.4f} | "
                f"mean component r {row['heldout']['mean_component_r']:.4f} | "
                f"frac ceiling {row['mean_fraction_of_ceiling']:.3f} | "
                f"{row['train_seconds']:.0f}s"
            )
        # end for seed
    # end for requested variant

    # The neural time axis is fixed by the window and the bin width, so a run
    # whose variants all came back from disk -- nothing prepared, nothing
    # cached -- still knows where its bins are.
    bin_edges_ms = np.linspace(
        cfg.window_start_ms,
        cfg.window_end_ms,
        int(round((cfg.window_end_ms - cfg.window_start_ms) / cfg.timebin_ms)) + 1,
    )
    if prepared_cache:
        bin_edges_ms = list(prepared_cache.values())[0]["bin_edges_ms"]
    # end if a preparation is still in hand
    # A panel per variant is unreadable past a handful, and the curves worth
    # comparing are the ones that actually decode; the rest are kept in
    # results.json for anyone who wants them.
    attention_runs = [
        run_name
        for run_name, extras in extras_by_variant.items()
        if extras.get("attention") is not None
    ]
    attention_runs.sort(
        key=lambda run_name: result_rows[run_name]["heldout"]["mean_r2"],
        reverse=True,
    )
    attention_by_variant = {
        run_name: extras_by_variant[run_name]["attention"]
        for run_name in attention_runs[: cfg.max_attention_panels]
    }
    per_bin_r2 = next(
        (
            extras["per_bin_r2"]
            for extras in extras_by_variant.values()
            if extras.get("per_bin_r2")
        ),
        None,
    )
    plot_attention_curves(
        attention_by_variant,
        per_bin_r2,
        np.asarray(bin_edges_ms),
        layer_names,
        output_dir / "attention_over_time.png",
    )

    with open(output_dir / "config.json", "w") as config_file:
        json.dump(
            {
                **asdict(cfg),
                "variant_overrides": VARIANTS,
                "ridge_alphas": [float(alpha) for alpha in RIDGE_ALPHAS],
                "bin_edges_ms": np.asarray(bin_edges_ms).tolist(),
            },
            config_file,
            indent=2,
        )
    # end with saved configuration
    with open(output_dir / "results.json", "w") as results_file:
        json.dump(
            {"rows": result_rows, "extras": extras_by_variant},
            results_file,
            indent=2,
        )
    # end with saved metrics

    # The depth-ordering claim in one table: where each depth's attention sits
    # on the time axis, against where the linear reference says that depth is
    # actually decodable. An attention centroid that does not track the
    # reference centroid is a pattern the model invented, not one it found.
    bin_centers_ms = 0.5 * (
        np.asarray(bin_edges_ms)[:-1] + np.asarray(bin_edges_ms)[1:]
    )
    if per_bin_r2 is not None:
        reference_centroids = [
            temporal_centroid(
                np.clip([scores[layer_name]["r2"] for scores in per_bin_r2], 0, None),
                bin_centers_ms,
            )
            for layer_name in layer_names
        ]
        print("\n" + f"{'temporal centroid (ms)':<24}" + "".join(
            f"{name:>10}" for name in layer_names
        ))
        print("-" * (24 + 10 * len(layer_names)))
        print(
            f"{'single-bin ridge R2':<24}"
            + "".join(f"{value:>10.1f}" for value in reference_centroids)
        )
        for run_name, extras in sorted(attention_by_variant.items()):
            centroids = temporal_centroid(np.asarray(extras), bin_centers_ms)
            print(
                f"{run_name:<24}"
                + "".join(f"{value:>10.1f}" for value in centroids)
            )
        # end for attention variant
        for run_name, row in sorted(result_rows.items()):
            if row.get("attention_entropy"):
                print(
                    f"{run_name + ' entropy':<24}"
                    + "".join(
                        f"{value:>10.2f}" for value in row["attention_entropy"]
                    )
                    + f"   (uniform = {np.log(len(bin_centers_ms)):.2f})"
                )
            # end if the variant recorded attention entropy
        # end for scored variant
    # end if the linear reference was run

    header = (
        f"{'variant':<24}{'mean R2':>9}{'mean r':>9}{'frac ceil':>10}  "
        + " ".join(f"{name:>8}" for name in layer_names)
        + f"{'params':>11}"
    )
    print("\n" + header)
    print("-" * len(header))
    ordered = sorted(
        result_rows.items(),
        key=lambda item: item[1]["heldout"]["mean_r2"],
        reverse=True,
    )
    for run_name, row in ordered:
        print(format_row(run_name, row, layer_names))
    # end for scored variant
    if any(
        VARIANTS[name.split("_seed")[0]].get("target_ranks")
        for name in result_rows
        if name.split("_seed")[0] in VARIANTS
    ):
        print(
            "\nNote: variants that override target_ranks are scored on a "
            "different set of retained PCs, so their R2 is not comparable to "
            "the rest of this table."
        )
    # end if a variant changed the target space it is scored on
    if any(row["alpha_at_grid_edge"] for row in result_rows.values()):
        print("\n* ridge penalty selected at the edge of the alpha grid.")
    # end if a ridge penalty is untrustworthy
    print(f"\nsaved to {output_dir}")
# EOF


if __name__ == "__main__":
    main()
