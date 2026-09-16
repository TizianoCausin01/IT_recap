"""
Beat the published three0 ridge reference on held-out image responses.

The reference is the `ridge_cv` row of run_temporal_architecture_experiments.py:
a concatenated four-layer DINOv3 RidgeCV over alphas 1e-6 to 1e3, fitted on the
466 training images of the seed-0 holdout and scored on 155 test images.

This script rebuilds that number exactly, then adds one change at a time up to
the reduced-rank multi-backbone decoder in IT_recap.three0_decoding, and repeats
the whole comparison over many independent image splits. Because the targets are
trial averages of a finite number of repeats, raw MSE is reported next to the
noise floor measured from the single-trial rasters: only about 14% of the test
MSE is reducible at all, so the reducible-error cut is the number that says how
much of the model gap was actually closed.
"""

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402


ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_SRC = PROJECT_ROOT / "python_scripts" / "src"
sys.path.insert(0, str(PROJECT_SRC))

with open(PROJECT_ROOT / "config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)
# end with project configuration

from IT_recap.three0_decoding import (  # noqa: E402
    THREE0_STIMULI,
    fit_reduced_rank_decoder,
    load_three0_targets,
    load_three0_trial_noise,
    make_image_splits,
    normalized_linear_kernel,
    score_predictions,
)


# The four DINOv3 layers the published reference concatenates, and where they
# sit inside the cached 24-layer dino_v3_l block.
REFERENCE_LAYERS = (3, 13, 16, 20)
REFERENCE_LAYER_WIDTH = 1024
REFERENCE_ALPHAS = np.logspace(-6, 3, 10)

# One change per step, each keeping everything before it.
STEP_NAMES = (
    "published ridge",
    "+wide alpha grid",
    "+train on train+val",
    "+rank-20 reduced rank",
    "+inverse-noise weights",
    "+backbone-ensemble kernel",
)


@dataclass
class Cfg:
    env: str | None = None
    feature_archive: str = "three0_backbone_features.npz"
    stimuli_folder: str = THREE0_STIMULI
    brain_area: str = "AIT"
    target_fs: int = 100
    time_start_ms: float = 0.0
    time_end_ms: float = 300.0
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    n_splits: int = 20
    rank: int = 20
    n_folds: int = 5
    output_dir: str | None = None
    smoke_test: bool = False


"""
parse_args
Parse command-line overrides into the experiment configuration.

OUTPUT:
    - cfg: Cfg -> data, split, and decoder settings
"""
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare the published three0 ridge with a reduced-rank decoder."
    )
    for field_name, field in Cfg.__dataclass_fields__.items():
        default = field.default
        if isinstance(default, bool):
            parser.add_argument(f"--{field_name}", action="store_true")
        elif default is None:
            parser.add_argument(f"--{field_name}", default=default)
        else:
            parser.add_argument(f"--{field_name}", type=type(default), default=default)
        # end if boolean, optional, or typed argument
    # end for configuration field
    return Cfg(**vars(parser.parse_args()))
# EOF


"""
run_one_split
Score every ablation step on a single held-out image split.

INPUT:
    - cfg: Cfg -> decoder settings
    - seed: int -> split seed
    - targets: np.ndarray -> [images, time, units] trial-averaged responses
    - ann_index: np.ndarray -> ANN index of every target row
    - reference_features: np.ndarray -> [images, dims] four-layer DINOv3 block
    - reference_kernel: np.ndarray -> kernel of the same DINOv3 block
    - ensemble_kernel: np.ndarray -> mean kernel over every backbone block
    - output_weights: np.ndarray -> inverse trial-mean noise SD per output
    - noise_variance: np.ndarray -> [images, time, units] variance of the mean

OUTPUT:
    - record: dict -> per-step scores plus the split's noise floor
"""
def run_one_split(
    cfg,
    seed,
    targets,
    ann_index,
    reference_features,
    reference_kernel,
    ensemble_kernel,
    output_weights,
    noise_variance,
):
    n_time, n_units = targets.shape[1], targets.shape[2]
    flat_targets = targets.reshape(len(targets), -1).astype(np.float64)
    splits = make_image_splits(
        ann_index, seed, cfg.validation_fraction, cfg.test_fraction
    )
    train, test = splits["train"], splits["test"]
    development = np.concatenate([train, splits["validation"]])
    train_mean = targets[train].mean(0)
    development_mean = targets[development].mean(0)
    noise_floor = float(noise_variance[test].mean())
    uniform_weights = np.ones(flat_targets.shape[1])
    full_rank = flat_targets.shape[1]

    # Step 0 is the published recipe verbatim: narrow alpha grid, train only.
    reference = RidgeCV(alphas=REFERENCE_ALPHAS)
    reference.fit(reference_features[train], flat_targets[train])
    step_predictions = [
        (reference.predict(reference_features[test]), train_mean),
    ]
    # Steps 1-2 keep the plain full-rank ridge and only widen the alpha search,
    # then fold the unused validation images back into the fit.
    for rows, mean in [(train, train_mean), (development, development_mean)]:
        predictions, _ = fit_reduced_rank_decoder(
            reference_kernel, flat_targets, rows, test,
            rank=full_rank, output_weights=uniform_weights, n_folds=cfg.n_folds,
        )
        step_predictions.append((predictions, mean))
    # end for fitting-row set
    # Steps 3-5 add the reduced rank, the noise weighting, and the zoo kernel.
    for kernel, weights in [
        (reference_kernel, uniform_weights),
        (reference_kernel, output_weights),
        (ensemble_kernel, output_weights),
    ]:
        predictions, _ = fit_reduced_rank_decoder(
            kernel, flat_targets, development, test,
            rank=cfg.rank, output_weights=weights, n_folds=cfg.n_folds,
        )
        step_predictions.append((predictions, development_mean))
    # end for decoder variant

    record = {"seed": seed, "noise_floor": noise_floor, "steps": {}}
    for name, (predictions, mean) in zip(STEP_NAMES, step_predictions):
        record["steps"][name] = score_predictions(
            predictions.reshape(-1, n_time, n_units),
            targets[test],
            mean,
            noise_floor,
        )
    # end for ablation step
    return record
# EOF


"""
summarize
Average every step over splits and express the gain against the noise floor.

INPUT:
    - records: list[dict] -> per-split scores from run_one_split

OUTPUT:
    - summary: list[dict] -> one row per ablation step, in step order
"""
def summarize(records):
    baseline_mse = np.array(
        [record["steps"][STEP_NAMES[0]]["mse"] for record in records]
    )
    floor = np.array([record["noise_floor"] for record in records])
    summary = []
    for name in STEP_NAMES:
        mse = np.array([record["steps"][name]["mse"] for record in records])
        # The reducible error is what sits above the trial-mean noise floor.
        reducible_cut = (baseline_mse - mse) / (baseline_mse - floor) * 100.0
        summary.append(
            {
                "step": name,
                "test_mse": float(mse.mean()),
                "test_r2": float(
                    np.mean([record["steps"][name]["r2"] for record in records])
                ),
                "fraction_of_ceiling": float(
                    np.mean(
                        [
                            record["steps"][name]["fraction_of_ceiling"]
                            for record in records
                        ]
                    )
                ),
                "raw_mse_reduction_pct": float(
                    ((baseline_mse - mse) / baseline_mse * 100.0).mean()
                ),
                "reducible_cut_pct": float(reducible_cut.mean()),
                "reducible_cut_sem": float(
                    reducible_cut.std(ddof=1) / np.sqrt(len(records))
                )
                if len(records) > 1
                else 0.0,
                "splits_improved": int((mse < baseline_mse).sum()),
            }
        )
    # end for ablation step
    return summary
# EOF


def main():
    cfg = parse_args()
    if cfg.smoke_test:
        cfg.n_splits = min(cfg.n_splits, 2)
    # end if a short run was requested
    env = cfg.env or ENV
    paths = dict(config[env]["paths"])
    sys.path.append(paths["useful_stuff_path"])
    output_dir = Path(
        cfg.output_dir or PROJECT_ROOT / "results" / "three0_beat_ridge"
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    targets, ann_index, _ = load_three0_targets(
        paths,
        brain_area=cfg.brain_area,
        target_fs=cfg.target_fs,
        time_start_ms=cfg.time_start_ms,
        time_end_ms=cfg.time_end_ms,
        stimuli_folder=cfg.stimuli_folder,
    )
    print(f"targets {targets.shape}")

    noise_cache = (
        Path(paths["data_path"]) / "data"
        / f"three0_{cfg.brain_area}_{cfg.target_fs}Hz_trial_noise.npz"
    )
    noise_variance, _ = load_three0_trial_noise(
        paths,
        brain_area=cfg.brain_area,
        target_fs=cfg.target_fs,
        time_start_ms=cfg.time_start_ms,
        time_end_ms=cfg.time_end_ms,
        cache_path=noise_cache,
    )
    # One weight per flattened output: reliable outputs get more of the rank.
    output_weights = 1.0 / np.sqrt(
        noise_variance.mean(0).reshape(-1)
    )
    print(
        f"trial-mean noise floor {noise_variance.mean():.5f} "
        f"(response variance {targets.var():.5f})"
    )

    archive_path = Path(paths["data_path"]) / "models" / cfg.feature_archive
    if not archive_path.is_file():
        raise SystemExit(
            f"Missing {archive_path}. Run extract_three0_backbone_features.py."
        )
    # end if the feature archive was never extracted
    with np.load(archive_path, allow_pickle=True) as archive:
        block_names = [
            key for key in archive.files if key not in {"ann_index", "image_paths"}
        ]
        kernels = {name: normalized_linear_kernel(archive[name]) for name in block_names}
        dino_block = archive["dino_v3_l"]
    # end with backbone feature archive
    print(f"backbone blocks: {', '.join(block_names)}")

    # The published reference uses only four DINOv3 layers of that block.
    reference_columns = np.concatenate(
        [
            np.arange(layer * REFERENCE_LAYER_WIDTH, (layer + 1) * REFERENCE_LAYER_WIDTH)
            for layer in REFERENCE_LAYERS
        ]
    )
    reference_features = dino_block[:, reference_columns].astype(np.float64)
    reference_kernel = normalized_linear_kernel(reference_features)
    ensemble_kernel = np.mean([kernels[name] for name in block_names], axis=0)

    records = []
    for seed in range(cfg.n_splits):
        records.append(
            run_one_split(
                cfg, seed, targets, ann_index, reference_features,
                reference_kernel, ensemble_kernel, output_weights, noise_variance,
            )
        )
        first = records[-1]["steps"][STEP_NAMES[0]]
        last = records[-1]["steps"][STEP_NAMES[-1]]
        print(
            f"  seed {seed:2d} | published {first['mse']:.5f} (R2 {first['r2']:.4f})"
            f" | decoder {last['mse']:.5f} (R2 {last['r2']:.4f})",
            flush=True,
        )
    # end for image split

    summary = summarize(records)
    print(
        f"\n{'step':28s} {'test_mse':>9s} {'test_R2':>8s} {'raw dMSE':>9s} "
        f"{'reducible cut':>16s} {'% ceiling':>10s} {'won':>5s}"
    )
    for row in summary:
        print(
            f"{row['step']:28s} {row['test_mse']:9.5f} {row['test_r2']:8.4f} "
            f"{row['raw_mse_reduction_pct']:8.2f}% "
            f"{row['reducible_cut_pct']:9.1f}% +- {row['reducible_cut_sem']:.1f} "
            f"{row['fraction_of_ceiling'] * 100:9.1f}% "
            f"{row['splits_improved']:3d}/{len(records)}"
        )
    # end for summary row

    with open(output_dir / "config.json", "w") as config_out:
        json.dump(asdict(cfg) | {"blocks": block_names}, config_out, indent=2)
    # end with configuration file
    with open(output_dir / "results.json", "w") as results_out:
        json.dump({"summary": summary, "per_split": records}, results_out, indent=2)
    # end with results file

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    positions = np.arange(len(summary))
    axes[0].barh(positions, [row["reducible_cut_pct"] for row in summary],
                 xerr=[row["reducible_cut_sem"] for row in summary],
                 color="#eb6834")
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels([row["step"] for row in summary], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("cut of the reducible test error (%)")
    axes[0].set_title(f"three0 {cfg.brain_area}, {len(records)} image splits")

    baseline = np.array([r["steps"][STEP_NAMES[0]]["mse"] for r in records])
    decoder = np.array([r["steps"][STEP_NAMES[-1]]["mse"] for r in records])
    axes[1].scatter(baseline, decoder, color="#2a78d6", s=22)
    limits = [min(baseline.min(), decoder.min()), max(baseline.max(), decoder.max())]
    axes[1].plot(limits, limits, color="#666666", linewidth=1)
    axes[1].set_xlabel("published ridge test MSE")
    axes[1].set_ylabel("reduced-rank decoder test MSE")
    axes[1].set_title("per split (below the line is better)")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    # end for panel
    figure.tight_layout()
    figure.savefig(output_dir / "three0_beat_ridge.png", dpi=160)
    plt.close(figure)
    print(f"\nwrote {output_dir}")
# EOF


if __name__ == "__main__":
    main()
