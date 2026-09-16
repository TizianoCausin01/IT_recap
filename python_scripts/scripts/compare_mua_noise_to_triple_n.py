"""
Quantify the TVSD MUA noise floor and compare it against Triple-N sorted units.

The decoder experiments in this project all plateau around half of the TVSD
noise ceiling, which leaves two candidate explanations: the model is too weak,
or the MUA targets are too noisy to do better. This script measures the second
one directly. It computes per-site split-half reliability of the monkey F MUA
from the 100 repeated test images, then, when Triple-N Processed files are
present, puts that distribution next to the per-unit reliability of
BombCell-sorted Neuropixels units recorded under a comparable protocol.

Two reliability conventions are reported, because the two datasets use
different ones and mixing them up would make MUA look worse than it is:

    r_sb        Spearman-Brown corrected split-half correlation of the
                window-averaged response. This is what Triple-N stores in
                reliability_basic/best and what its noise-ceiling demo returns.
    ceiling     sqrt(r_sb), the ceiling on the stim_r scale, which is the
                convention used by this project's existing results.json files.

Triple-N Processed files are small (one per session, no rasters), so this
comparison needs a few hundred MB rather than the multi-GB GoodUnit files.
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


ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_SRC = PROJECT_ROOT / "python_scripts" / "src"
sys.path.insert(0, str(PROJECT_SRC))

with open(PROJECT_ROOT / "config.yaml", "r") as f:
    config = yaml.safe_load(f)

from IT_recap.neural_prediction_training import stimulus_correlation  # noqa: E402
from IT_recap.triple_n import (  # noqa: E402
    TRIPLE_N_UNIT_TYPES,
    load_triple_n_processed,
)
from IT_recap.tvsd import load_tvsd_metadata  # noqa: E402


# Colours are assigned by entity so a population keeps its colour in every panel.
POPULATION_COLORS = {
    "tvsd_mua": "#eb6834",
    "triple_n_single_unit": "#1baf7a",
    "triple_n_mua": "#2a78d6",
}
POPULATION_LABELS = {
    "tvsd_mua": "TVSD monkey F MUA (Utah)",
    "triple_n_single_unit": "Triple-N single units (Npx)",
    "triple_n_mua": "Triple-N sorted MUA (Npx)",
}


@dataclass
class Cfg:
    # Environment and cache locations. None resolves through config.yaml.
    env: str = ENV
    mua_file_name: str = "f_THINGS_MUA_trials.mat"
    triple_n_dir: str | None = None
    output_dir: str | None = None

    # Which prepared TVSD target cache to score.
    area: str = "IT"
    target_fs: int = 100
    time_start_ms: float = 0.0
    time_end_ms: float = 200.0

    # Window treated as driven response when averaging over time.
    response_onset_ms: float = 50.0

    # Split-half estimation.
    n_resamples: int = 100
    random_seed: int = 0

    # Triple-N comparison population.
    triple_n_reliability_field: str = "reliability_best"


"""
parse_args
Parse command-line overrides into the noise-comparison configuration.

OUTPUT:
    - cfg: Cfg -> cache selection, response window, and resampling settings
"""
def parse_args() -> Cfg:
    parser = argparse.ArgumentParser(
        description=(
            "Compare TVSD MUA split-half reliability against Triple-N sorted "
            "single units."
        )
    )
    parser.add_argument("--env", default=Cfg.env, choices=config)
    parser.add_argument("--mua_file_name", default=Cfg.mua_file_name)
    parser.add_argument("--triple_n_dir", default=Cfg.triple_n_dir)
    parser.add_argument("--output_dir", default=Cfg.output_dir)
    parser.add_argument("--area", default=Cfg.area)
    parser.add_argument("--target_fs", type=int, default=Cfg.target_fs)
    parser.add_argument("--time_start_ms", type=float, default=Cfg.time_start_ms)
    parser.add_argument("--time_end_ms", type=float, default=Cfg.time_end_ms)
    parser.add_argument(
        "--response_onset_ms", type=float, default=Cfg.response_onset_ms
    )
    parser.add_argument("--n_resamples", type=int, default=Cfg.n_resamples)
    parser.add_argument("--random_seed", type=int, default=Cfg.random_seed)
    parser.add_argument(
        "--triple_n_reliability_field",
        default=Cfg.triple_n_reliability_field,
        choices=("reliability_best", "reliability_basic"),
    )
    return Cfg(**vars(parser.parse_args()))
# EOF


"""
window_mean_split_half_reliability
Split-half reliability of each channel's window-averaged response.

This is the Triple-N convention: collapse the response window to one number per
trial, split each image's repetitions in half, correlate the two halves across
images, then apply the Spearman-Brown correction. Returning the uncorrected
half-split value as well makes the correction's size visible.

INPUT:
    - trial_responses: np.ndarray -> [presentations, channels] window means
    - image_ids: np.ndarray -> zero-based image identifier per presentation
    - n_images: int -> number of distinct repeated images
    - n_resamples: int -> random half-splits averaged over
    - seed: int -> split reproducibility

OUTPUT:
    - reliability_sb: np.ndarray -> [channels] Spearman-Brown corrected r
    - reliability_half: np.ndarray -> [channels] raw half-split r
"""
def window_mean_split_half_reliability(
    trial_responses, image_ids, n_images, n_resamples=100, seed=0
):
    rng = np.random.default_rng(seed)
    rows_by_image = [
        np.flatnonzero(image_ids == image_id) for image_id in range(n_images)
    ]

    half_correlations = []
    for _ in range(n_resamples):
        first_half, second_half = [], []
        for rows in rows_by_image:
            shuffled_rows = rng.permutation(rows)
            midpoint = len(shuffled_rows) // 2
            first_half.append(trial_responses[shuffled_rows[:midpoint]].mean(axis=0))
            second_half.append(trial_responses[shuffled_rows[midpoint:]].mean(axis=0))
        # end for repeated image
        half_correlations.append(
            stimulus_correlation(np.stack(first_half), np.stack(second_half))
        )
    # end for half-split resample

    reliability_half = np.nanmean(half_correlations, axis=0)
    # Spearman-Brown extrapolates a half-set correlation to the full set. It is
    # only meaningful for averaging, which is what both halves do here.
    reliability_sb = 2 * reliability_half / (1 + reliability_half)
    return np.clip(reliability_sb, 0.0, 1.0), reliability_half
# EOF


"""
summarize_reliability
Reduce one population's per-site reliability to reportable statistics.

INPUT:
    - reliability: np.ndarray -> [sites] Spearman-Brown corrected r
    - name: str -> population identifier

OUTPUT:
    - summary: dict -> counts, quantiles, and usable-site fractions
"""
def summarize_reliability(reliability, name):
    finite = reliability[np.isfinite(reliability)]
    if len(finite) == 0:
        raise ValueError(f"{name} has no finite reliability values.")
    # end if the population is empty
    return {
        "population": name,
        "n_sites": int(len(reliability)),
        "n_finite": int(len(finite)),
        "median_r_sb": round(float(np.median(finite)), 4),
        "mean_r_sb": round(float(np.mean(finite)), 4),
        "q25_r_sb": round(float(np.quantile(finite, 0.25)), 4),
        "q75_r_sb": round(float(np.quantile(finite, 0.75)), 4),
        # sqrt(r_sb) is the ceiling on the stim_r scale used by results.json.
        "median_stim_r_ceiling": round(float(np.median(np.sqrt(finite))), 4),
        "fraction_above_0.4": round(float(np.mean(finite > 0.4)), 4),
        "fraction_above_0.7": round(float(np.mean(finite > 0.7)), 4),
        "fraction_below_0.1": round(float(np.mean(finite < 0.1)), 4),
    }
# EOF


"""
load_tvsd_test_responses
Load the repeated-test presentations of a prepared TVSD target cache.

INPUT:
    - cfg: Cfg -> cache selection and response window
    - paths: dict -> active project paths

OUTPUT:
    - trial_responses: np.ndarray -> [presentations, channels] window means
    - trial_targets: np.ndarray -> [presentations, time, channels] responses
    - image_ids: np.ndarray -> zero-based test-image id per presentation
"""
def load_tvsd_test_responses(cfg, paths):
    data_root = Path(paths["data_path"])
    target_cache_path = data_root / "data" / (
        f"tvsd_monkeyF_{cfg.area}_{cfg.time_start_ms:g}-"
        f"{cfg.time_end_ms:g}ms_{cfg.target_fs}Hz_baseline_corrected.npy"
    )
    mua_path = data_root / "data" / cfg.mua_file_name
    for required_path in (target_cache_path, mua_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"Missing cache {required_path}.")
        # end if a required cache is absent
    # end for required cache

    targets = np.load(target_cache_path, mmap_mode="r")
    allmat, _ = load_tvsd_metadata(mua_path)
    test_rows = np.flatnonzero(allmat[:, 2] > 0)
    image_ids = allmat[test_rows, 2] - 1

    # Only the 3,000 repeated-test presentations are materialized; the rest of
    # the cache stays on disk.
    trial_targets = np.asarray(targets[test_rows], dtype=np.float32)

    # Average the driven part of the window so each trial is one number per
    # site, matching how Triple-N computes its per-unit reliability.
    onset_bin = int(
        round((cfg.response_onset_ms - cfg.time_start_ms) * cfg.target_fs / 1000)
    )
    if not 0 <= onset_bin < trial_targets.shape[1]:
        raise ValueError(
            f"response_onset_ms {cfg.response_onset_ms} falls outside the "
            f"{trial_targets.shape[1]}-bin cache."
        )
    # end if the onset lies outside the prepared window
    trial_responses = trial_targets[:, onset_bin:, :].mean(axis=1)
    return trial_responses, trial_targets, image_ids
# EOF


"""
load_triple_n_reliabilities
Collect per-unit reliability from every Triple-N Processed file that is local.

INPUT:
    - triple_n_dir: Path -> directory holding Processed_ses*.mat files
    - reliability_field: str -> which reliability column to read

OUTPUT:
    - populations: dict -> unit-type label mapped to a reliability vector
    - session_names: list[str] -> Processed files that were read
"""
def load_triple_n_reliabilities(triple_n_dir, reliability_field):
    processed_paths = sorted(Path(triple_n_dir).glob("Processed_ses*.mat"))
    reliability_by_type = {}
    session_names = []

    for processed_path in processed_paths:
        summary = load_triple_n_processed(processed_path)
        if reliability_field not in summary:
            print(f"  {processed_path.name}: no {reliability_field}, skipped")
            continue
        # end if this session lacks the requested column

        reliability = np.ravel(summary[reliability_field]).astype(float)
        unit_codes = np.ravel(summary["UnitType"]).astype(int)
        if len(reliability) != len(unit_codes):
            print(f"  {processed_path.name}: misaligned columns, skipped")
            continue
        # end if the summary columns disagree

        for code, label in TRIPLE_N_UNIT_TYPES.items():
            selected = reliability[unit_codes == code]
            if len(selected):
                reliability_by_type.setdefault(label, []).append(selected)
            # end if this session contributes units of this class
        # end for BombCell class
        session_names.append(processed_path.name)
    # end for Processed file

    populations = {
        label: np.concatenate(chunks)
        for label, chunks in reliability_by_type.items()
    }
    return populations, session_names
# EOF


"""
plot_reliability_comparison
Draw the reliability distributions and the time course of the MUA ceiling.

INPUT:
    - populations: dict -> population name mapped to per-site reliability
    - time_ceiling: np.ndarray -> [time bins] median MUA ceiling per bin
    - time_axis_ms: np.ndarray -> [time bins] bin centre times
    - output_path: Path -> destination .png

OUTPUT:
    - None
"""
def plot_reliability_comparison(
    populations, time_ceiling, time_axis_ms, output_path
):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    bin_edges = np.linspace(0.0, 1.0, 41)
    for name, reliability in populations.items():
        finite = reliability[np.isfinite(reliability)]
        axes[0].hist(
            finite,
            bins=bin_edges,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=POPULATION_COLORS.get(name, "#666666"),
            label=f"{POPULATION_LABELS.get(name, name)} (n={len(finite)})",
        )
        axes[0].axvline(
            np.median(finite),
            color=POPULATION_COLORS.get(name, "#666666"),
            linestyle=":",
            linewidth=1.2,
        )
    # end for compared population
    axes[0].set_xlabel("split-half reliability (Spearman-Brown corrected)")
    axes[0].set_ylabel("density")
    axes[0].set_title("Per-site response reliability")
    axes[0].legend(fontsize=7, frameon=False)

    axes[1].plot(time_axis_ms, time_ceiling, color=POPULATION_COLORS["tvsd_mua"])
    axes[1].set_xlabel("time from stimulus onset (ms)")
    axes[1].set_ylabel("median ceiling on stim_r scale")
    axes[1].set_title("TVSD MUA ceiling over the response window")
    axes[1].set_ylim(0.0, 1.0)

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    # end for panel
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
# EOF


def main():
    cfg = parse_args()
    paths = config[cfg.env]["paths"]
    output_dir = Path(
        cfg.output_dir or PROJECT_ROOT / "results" / "mua_noise_vs_triple_n"
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"scoring TVSD monkey F {cfg.area} MUA")
    trial_responses, trial_targets, image_ids = load_tvsd_test_responses(
        cfg, paths
    )
    n_images = int(image_ids.max()) + 1
    print(
        f"  {len(trial_responses):,} repeated-test presentations | "
        f"{n_images} images | {trial_responses.shape[1]} sites"
    )

    mua_reliability, mua_half = window_mean_split_half_reliability(
        trial_responses,
        image_ids,
        n_images,
        n_resamples=cfg.n_resamples,
        seed=cfg.random_seed,
    )
    populations = {"tvsd_mua": mua_reliability}

    # Per-bin ceiling shows whether the noise is uniform across the window or
    # concentrated before the response has developed.
    time_ceiling = []
    for time_bin in range(trial_targets.shape[1]):
        bin_reliability, _ = window_mean_split_half_reliability(
            trial_targets[:, time_bin, :],
            image_ids,
            n_images,
            n_resamples=max(4, cfg.n_resamples // 10),
            seed=cfg.random_seed,
        )
        time_ceiling.append(np.nanmedian(np.sqrt(bin_reliability)))
    # end for target time bin
    time_ceiling = np.asarray(time_ceiling)
    time_axis_ms = cfg.time_start_ms + (
        np.arange(trial_targets.shape[1]) + 0.5
    ) * (1000.0 / cfg.target_fs)

    triple_n_dir = Path(
        cfg.triple_n_dir or Path(paths["data_path"]) / "data" / "triple_n"
    ).expanduser()
    session_names = []
    if triple_n_dir.is_dir():
        print(f"reading Triple-N Processed files from {triple_n_dir}")
        triple_n_populations, session_names = load_triple_n_reliabilities(
            triple_n_dir, cfg.triple_n_reliability_field
        )
        for label, reliability in triple_n_populations.items():
            if label in {"single_unit", "mua"}:
                populations[f"triple_n_{label}"] = reliability
            # end if this class is one of the compared populations
        # end for Triple-N unit class
    # end if Triple-N summaries are available locally

    if len(session_names) == 0:
        print(
            "no Triple-N Processed files found; reporting the TVSD MUA side "
            "only. Run download_triple_n_sessions.py to add the comparison."
        )
    # end if the comparison half is missing

    summaries = [
        summarize_reliability(reliability, name)
        for name, reliability in populations.items()
    ]
    for summary in summaries:
        print(
            f"  {summary['population']:>26} | n={summary['n_finite']:>5} | "
            f"median r_sb {summary['median_r_sb']:.3f} | "
            f"median ceiling {summary['median_stim_r_ceiling']:.3f} | "
            f"frac>0.4 {summary['fraction_above_0.4']:.2f} | "
            f"frac<0.1 {summary['fraction_below_0.1']:.2f}"
        )
    # end for scored population

    plot_reliability_comparison(
        populations,
        time_ceiling,
        time_axis_ms,
        output_dir / "reliability_comparison.png",
    )
    np.savez(
        output_dir / "reliability.npz",
        tvsd_mua_reliability_sb=mua_reliability,
        tvsd_mua_reliability_half=mua_half,
        tvsd_mua_time_ceiling=time_ceiling,
        time_axis_ms=time_axis_ms,
        **{
            f"{name}_reliability_sb": reliability
            for name, reliability in populations.items()
            if name != "tvsd_mua"
        },
    )
    with open(output_dir / "summary.json", "w") as summary_file:
        json.dump(
            {
                "config": asdict(cfg),
                "triple_n_sessions": session_names,
                "populations": summaries,
            },
            summary_file,
            indent=2,
        )
    # end with summary file
    print(f"\nwrote {output_dir}")
# EOF


if __name__ == "__main__":
    main()
