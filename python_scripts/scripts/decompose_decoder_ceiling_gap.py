"""
Decide whether the TVSD decoder plateau is noise-limited or model-limited.

Every decoder variant tried so far reaches roughly 55% of the MUA noise
ceiling. That single number cannot distinguish two very different situations:
the targets could be too noisy to do better, or the model and its DINOv3
features could be missing structure that the recordings actually contain.

The two cases make opposite predictions once the shortfall is broken down by
site reliability. stim_r is already normalized by the ceiling, so if noise were
what binds, clean sites would show a higher fraction of ceiling than noisy ones
and excluding dead sites would raise the average. If instead the model is what
binds, the fraction stays flat across reliability strata.

This script reads the per-(time, site) stim_r arrays that the decoder
experiments already saved, so it retrains nothing.
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
sys.path.insert(0, str(PROJECT_ROOT / "python_scripts" / "src"))

with open(PROJECT_ROOT / "config.yaml", "r") as f:
    config = yaml.safe_load(f)


VARIANT_COLORS = {
    "ridge": "#2a78d6",
    "baseline": "#eb6834",
    "noise_layer": "#1baf7a",
    "temporal_noise": "#4a3aa7",
}

# Site-reliability strata used for the decomposition, on the stim_r scale.
CEILING_BINS = ((0.0, 0.3), (0.3, 0.6), (0.6, 0.8), (0.8, 0.9), (0.9, 1.01))


@dataclass
class Cfg:
    # Saved time courses from run_tvsd_noise_regularization.py.
    time_course_path: str | None = None
    output_dir: str | None = None

    # Target window the arrays were produced with.
    time_start_ms: float = 0.0
    target_fs: int = 100

    # Bins from which the ceiling has plateaued; earlier bins are onset ramp.
    steady_onset_ms: float = 85.0
    response_onset_ms: float = 50.0

    # Variant whose per-site behaviour is decomposed.
    variant: str = "baseline"


"""
parse_args
Parse command-line overrides into the ceiling-decomposition configuration.

OUTPUT:
    - cfg: Cfg -> input arrays, response windows, and variant selection
"""
def parse_args() -> Cfg:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose the decoder's fraction of noise ceiling by time bin and "
            "by site reliability."
        )
    )
    parser.add_argument("--time_course_path", default=Cfg.time_course_path)
    parser.add_argument("--output_dir", default=Cfg.output_dir)
    parser.add_argument("--time_start_ms", type=float, default=Cfg.time_start_ms)
    parser.add_argument("--target_fs", type=int, default=Cfg.target_fs)
    parser.add_argument(
        "--steady_onset_ms", type=float, default=Cfg.steady_onset_ms
    )
    parser.add_argument(
        "--response_onset_ms", type=float, default=Cfg.response_onset_ms
    )
    parser.add_argument("--variant", default=Cfg.variant)
    return Cfg(**vars(parser.parse_args()))
# EOF


"""
decompose_by_reliability
Group sites by their noise ceiling and report the fraction achieved in each.

INPUT:
    - site_ceiling: np.ndarray -> [sites] ceiling averaged over the window
    - site_stim_r: np.ndarray -> [sites] decoder stim_r over the same window

OUTPUT:
    - strata: list[dict] -> per-stratum counts, means, and achieved fraction
"""
def decompose_by_reliability(site_ceiling, site_stim_r):
    finite = np.isfinite(site_ceiling) & np.isfinite(site_stim_r)
    strata = []
    for lower, upper in CEILING_BINS:
        in_bin = finite & (site_ceiling >= lower) & (site_ceiling < upper)
        if not in_bin.any():
            continue
        # end if no site falls in this stratum
        mean_ceiling = float(site_ceiling[in_bin].mean())
        mean_stim_r = float(site_stim_r[in_bin].mean())
        strata.append(
            {
                "ceiling_range": [lower, min(upper, 1.0)],
                "n_sites": int(in_bin.sum()),
                "mean_ceiling": round(mean_ceiling, 4),
                "mean_stim_r": round(mean_stim_r, 4),
                # The diagnostic quantity: flat across strata means the
                # shortfall does not scale with how noisy a site is.
                "fraction_of_ceiling": round(mean_stim_r / mean_ceiling, 4),
            }
        )
    # end for reliability stratum
    return strata
# EOF


def main():
    cfg = parse_args()
    time_course_path = Path(
        cfg.time_course_path
        or PROJECT_ROOT / "results" / "tvsd_noise_regularization"
        / "stim_r_time_courses.npz"
    ).expanduser()
    if not time_course_path.is_file():
        raise FileNotFoundError(
            f"Missing {time_course_path}. Run run_tvsd_noise_regularization.py "
            "first; it saves the per-(time, site) stim_r arrays."
        )
    # end if the saved time courses are absent

    output_dir = Path(
        cfg.output_dir or PROJECT_ROOT / "results" / "decoder_ceiling_gap"
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = np.load(time_course_path)
    ceiling = arrays["ceiling_mean"]
    if f"{cfg.variant}_mean" not in arrays:
        raise KeyError(
            f"{cfg.variant!r} is absent; found "
            f"{[name for name in arrays.files if name.endswith('_mean')]}."
        )
    # end if the requested variant was not saved

    n_time = ceiling.shape[0]
    time_axis_ms = cfg.time_start_ms + (np.arange(n_time) + 0.5) * (
        1000.0 / cfg.target_fs
    )
    steady_bin = int(
        round((cfg.steady_onset_ms - cfg.time_start_ms) * cfg.target_fs / 1000)
    )
    response_bin = int(
        round((cfg.response_onset_ms - cfg.time_start_ms) * cfg.target_fs / 1000)
    )
    steady = slice(steady_bin, n_time)

    variant_names = [
        name[: -len("_mean")] for name in arrays.files
        if name.endswith("_mean") and name != "ceiling_mean"
    ]

    # --- fraction of ceiling over time, for every saved variant ---
    windows = {}
    for name in variant_names:
        stim_r = arrays[f"{name}_mean"]
        windows[name] = {
            "response_window_fraction": round(
                float(
                    np.nanmean(stim_r[response_bin:])
                    / np.nanmean(ceiling[response_bin:])
                ),
                4,
            ),
            "steady_window_fraction": round(
                float(np.nanmean(stim_r[steady]) / np.nanmean(ceiling[steady])),
                4,
            ),
            "steady_stim_r": round(float(np.nanmean(stim_r[steady])), 4),
        }
    # end for saved variant
    print(f"ceiling over {cfg.steady_onset_ms:g}-200 ms: "
          f"{np.nanmean(ceiling[steady]):.4f}")
    for name, scores in windows.items():
        print(
            f"  {name:>16} | stim_r {scores['steady_stim_r']:.4f} | fraction "
            f"{scores['steady_window_fraction']:.3f}"
        )
    # end for scored variant

    # --- the decisive breakdown: fraction against site reliability ---
    site_ceiling = np.nanmean(ceiling[steady], axis=0)
    site_stim_r = np.nanmean(arrays[f"{cfg.variant}_mean"][steady], axis=0)
    strata = decompose_by_reliability(site_ceiling, site_stim_r)
    print(f"\nper-site decomposition for {cfg.variant}:")
    for stratum in strata:
        low, high = stratum["ceiling_range"]
        print(
            f"  ceiling {low:.1f}-{high:.1f} | n={stratum['n_sites']:3d} | "
            f"stim_r {stratum['mean_stim_r']:.3f} | fraction "
            f"{stratum['fraction_of_ceiling']:.3f}"
        )
    # end for reliability stratum

    # --- and the same question asked by excluding poor sites ---
    exclusions = []
    finite = np.isfinite(site_ceiling) & np.isfinite(site_stim_r)
    for threshold in (0.0, 0.3, 0.5, 0.7):
        kept = finite & (site_ceiling > threshold)
        exclusions.append(
            {
                "min_site_ceiling": threshold,
                "n_sites": int(kept.sum()),
                "mean_stim_r": round(float(site_stim_r[kept].mean()), 4),
                "fraction_of_ceiling": round(
                    float(site_stim_r[kept].mean() / site_ceiling[kept].mean()), 4
                ),
            }
        )
    # end for exclusion threshold

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(
        time_axis_ms, np.nanmean(ceiling, axis=1), color="#666666",
        label="noise ceiling", linewidth=2,
    )
    for name in variant_names:
        axes[0].plot(
            time_axis_ms,
            np.nanmean(arrays[f"{name}_mean"], axis=1),
            color=VARIANT_COLORS.get(name, "#999999"),
            label=name,
            linewidth=1.4,
        )
    # end for saved variant
    axes[0].axvline(cfg.steady_onset_ms, color="#bbbbbb", linestyle=":", linewidth=1)
    axes[0].set_xlabel("time from stimulus onset (ms)")
    axes[0].set_ylabel("mean stim_r over sites")
    axes[0].set_title("Prediction and ceiling over time")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend(frameon=False, fontsize=7)

    stratum_centers = [
        0.5 * (stratum["ceiling_range"][0] + stratum["ceiling_range"][1])
        for stratum in strata
    ]
    axes[1].plot(
        stratum_centers,
        [stratum["fraction_of_ceiling"] for stratum in strata],
        "o-",
        color=VARIANT_COLORS.get(cfg.variant, "#eb6834"),
    )
    for center, stratum in zip(stratum_centers, strata):
        axes[1].annotate(
            f"n={stratum['n_sites']}",
            (center, stratum["fraction_of_ceiling"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=7,
            color="#666666",
        )
    # end for annotated stratum
    axes[1].set_xlabel("site noise ceiling (stim_r scale)")
    axes[1].set_ylabel("fraction of ceiling achieved")
    axes[1].set_title(
        f"{cfg.variant}: shortfall vs site reliability\n"
        f"({cfg.steady_onset_ms:g}-200 ms)"
    )
    axes[1].set_ylim(0.0, 1.0)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    # end for panel
    figure.tight_layout()
    figure.savefig(output_dir / "decoder_ceiling_gap.png", dpi=160)
    plt.close(figure)

    with open(output_dir / "decomposition.json", "w") as summary_file:
        json.dump(
            {
                "config": asdict(cfg),
                "steady_ceiling": round(float(np.nanmean(ceiling[steady])), 4),
                "variants": windows,
                "reliability_strata": strata,
                "site_exclusions": exclusions,
            },
            summary_file,
            indent=2,
        )
    # end with summary file
    print(f"\nwrote {output_dir}")
# EOF


if __name__ == "__main__":
    main()
