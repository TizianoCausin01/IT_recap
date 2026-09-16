"""
Ask whether the TVSD train and test splits are the same distribution, and
whether the difference costs the decoder anything.

Every experiment in this project fits on 22,248 singly-presented images and
reports a score on 100 images shown 30 times, so a gap between validation and
test has three candidate explanations that the usual numbers cannot separate:
the test images are different images, they were recorded at different times, or
their targets are 30-repetition averages rather than single trials. This script
runs one check per candidate and then the matched comparison that prices them:

    1. protocol -- repetitions, recording days, position within the session;
    2. stimuli -- a domain classifier on encoder features, against a null built
       by splitting the training images against themselves;
    3. targets -- standardized response offsets, raw and day-matched, plus the
       spread the repetition averaging removes;
    4. matched hold-out -- one ridge map scored on 100 held-out *training*
       images and on the 100 test images, both single-trial. That difference is
       the shift; the repetition-averaged score sits next to it for reference.
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python_scripts" / "src"))

from IT_recap.distribution_shift import (  # noqa: E402
    matched_holdout_stim_r,
    neural_target_shift,
    single_trial_reliability,
    stimulus_feature_shift,
    summarize_presentation_protocol,
)
from IT_recap.tvsd_experiments import (  # noqa: E402
    N_TEST_IMAGES,
    average_targets_over_bin_groups,
    average_targets_over_window,
    gather_presentation_features,
    load_cached_data,
    load_project_paths,
    select_window_bin_indices,
)


@dataclass
class Cfg:
    env: str | None = None
    mua_file_name: str = "f_THINGS_MUA_trials.mat"
    feature_archive_name: str = "tvsd_monkeyF_ijepa_vith14_1k_224_features.npz"
    model_name: str = "ijepa_vith14_1k"
    # All six cached depths: a thinner stack understates ridge, and ridge is the
    # reference the matched hold-out comparison is read off.
    layer_names: list[str] = field(
        default_factory=lambda: [
            "encoder.layer.3.output.dense",
            "encoder.layer.4.output.dense",
            "encoder.layer.13.output.dense",
            "encoder.layer.17.output.dense",
            "encoder.layer.20.output.dense",
            "encoder.layer.27.output.dense",
        ]
    )
    output_dir: str | None = None

    area: str = "IT"
    target_fs: int = 100
    time_start_ms: float = 0.0
    time_end_ms: float = 200.0
    window_start_ms: float = 80.0
    window_end_ms: float = 180.0
    timebin_ms: float = 20.0

    # Feature-shift test: components retained from the reference pool, and how
    # many probe groups the observed and null AUCs average over.
    n_components: int = 64
    n_feature_resamples: int = 10

    # Matched hold-out test.
    n_holdout_resamples: int = 5
    validation_fraction: float = 0.1
    reliability_resamples: int = 20

    random_seed: int = 0
    smoke_test: bool = False


"""
parse_args
Parse command-line overrides into a configuration object.

OUTPUT:
    - cfg: Cfg -> data, test, and resampling settings
"""
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    default_layers = Cfg.__dataclass_fields__["layer_names"].default_factory()
    for field_name, field_definition in Cfg.__dataclass_fields__.items():
        if field_name == "layer_names":
            parser.add_argument("--layer_names", default=",".join(default_layers))
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
            parser.add_argument(argument_name, type=type(default), default=default)
        # end if boolean, optional string, or typed argument
    # end for configuration field
    arguments = vars(parser.parse_args())
    arguments["layer_names"] = [
        name.strip() for name in arguments["layer_names"].split(",") if name.strip()
    ]
    return Cfg(**arguments)
# EOF


"""
main
Run the four checks and print them as one table per check.
"""
def main():
    cfg = parse_args()
    if cfg.smoke_test:
        cfg.n_feature_resamples = 2
        cfg.n_holdout_resamples = 1
        cfg.reliability_resamples = 3
    # end if a smoke run was requested

    paths = load_project_paths(cfg)
    output_dir = Path(
        cfg.output_dir
        or PROJECT_ROOT / "results" / f"tvsd_distribution_shift_{cfg.model_name}"
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    targets_cache, train_features, test_features, allmat = load_cached_data(
        cfg, paths
    )
    bin_indices, covered_ms = select_window_bin_indices(
        targets_cache.shape[1],
        cfg.target_fs,
        cfg.time_start_ms,
        cfg.window_start_ms,
        cfg.window_end_ms,
    )
    print(
        f"{cfg.area} | window {covered_ms[0]:g}-{covered_ms[1]:g} ms | "
        f"{targets_cache.shape[2]} sites | {cfg.model_name} features "
        f"{train_features.shape[1]} x {train_features.shape[2]}"
    )

    # --- 1. protocol ---
    protocol = summarize_presentation_protocol(allmat)
    print("\n1. Protocol")
    print(
        f"   train {protocol['n_train_images']:,} images x "
        f"{protocol['train_reps_per_image']:.0f} rep | test "
        f"{protocol['n_test_images']} images x "
        f"{protocol['test_reps_per_image']:.0f} reps"
    )
    print(
        f"   days: {protocol['n_days']} total, "
        f"{protocol['n_days_with_train']} with train, "
        f"{protocol['n_days_with_test']} with test | "
        f"day-histogram total variation {protocol['day_tv_distance']:.4f}"
    )
    print(
        "   median within-day position: train "
        f"{protocol['median_within_day_position_train']:.3f} | test "
        f"{protocol['median_within_day_position_test']:.3f}"
    )

    # --- 2. stimuli ---
    features = stimulus_feature_shift(
        train_features,
        test_features,
        n_components=cfg.n_components,
        n_resamples=cfg.n_feature_resamples,
        seed=cfg.random_seed,
    )
    print(
        f"\n2. Stimulus features ({features['n_components']} components, "
        f"{features['variance_retained']:.2f} of the training variance)"
    )
    print(
        f"   domain-classifier AUC: test {features['observed_auc_mean']:.3f} "
        f"+/- {features['observed_auc_sd']:.3f} | null (train vs train) "
        f"{features['null_auc_mean']:.3f} +/- {features['null_auc_sd']:.3f} "
        f"| gap {features['auc_gap']:+.3f}"
    )
    print(
        "   nearest training image: test "
        f"{features['median_nn_distance_test']:.2f} vs held-out train "
        f"{features['median_nn_distance_heldout_train']:.2f} "
        f"(ratio {features['nn_distance_ratio']:.3f})"
    )
    print(
        "   variance outside the training subspace: test "
        f"{features['out_of_subspace_test']:.3f} vs held-out train "
        f"{features['out_of_subspace_heldout_train']:.3f}"
    )

    # --- 3. targets ---
    window_responses = average_targets_over_window(targets_cache, bin_indices)[
        :, 0, :
    ]
    target_report = neural_target_shift(window_responses, allmat, N_TEST_IMAGES)
    print("\n3. Neural targets (window mean, in training-SD units)")
    print(
        f"   raw test-minus-train offset: mean |d| "
        f"{target_report['mean_abs_raw_offset']:.3f}, 95th pct "
        f"{target_report['p95_abs_raw_offset']:.3f}"
    )
    print(
        f"   day-matched offset ({target_report['n_days_compared']} days): "
        f"mean |d| {target_report['mean_abs_day_matched_offset']:.3f}, 95th pct "
        f"{target_report['p95_abs_day_matched_offset']:.3f}"
    )
    print(
        "   across-stimulus spread: single trial "
        f"{target_report['mean_test_single_trial_sd']:.4f} -> "
        f"30-rep average {target_report['mean_test_averaged_sd']:.4f} "
        f"(ratio {target_report['averaged_over_single_trial_sd']:.3f})"
    )

    # --- 4. matched hold-out ---
    cached_bin_ms = 1000.0 / cfg.target_fs
    group_size = int(round(cfg.timebin_ms / cached_bin_ms))
    timebin_targets = average_targets_over_bin_groups(
        targets_cache, bin_indices, group_size
    )
    train_rows = np.flatnonzero(allmat[:, 1] > 0)
    test_rows = np.flatnonzero(allmat[:, 2] > 0)
    # Ridge reads one flat vector per presentation, all encoder depths stacked.
    flatten = lambda block: block.reshape(len(block), -1)
    train_design = flatten(
        gather_presentation_features(
            train_features, test_features, allmat, train_rows
        )
    )
    test_design = flatten(
        gather_presentation_features(
            train_features, test_features, allmat, test_rows
        )
    )
    test_image_ids = allmat[test_rows, 2] - 1

    reliability = single_trial_reliability(
        timebin_targets[test_rows],
        test_image_ids,
        N_TEST_IMAGES,
        n_resamples=cfg.reliability_resamples,
        seed=cfg.random_seed,
    )
    holdout = matched_holdout_stim_r(
        train_design,
        timebin_targets[train_rows],
        test_design,
        timebin_targets[test_rows],
        test_image_ids,
        n_images=N_TEST_IMAGES,
        n_resamples=cfg.n_holdout_resamples,
        validation_fraction=cfg.validation_fraction,
        seed=cfg.random_seed,
    )
    summary = holdout["summary"]
    print(
        f"\n4. Matched hold-out ({cfg.n_holdout_resamples} ridge fits, "
        f"{N_TEST_IMAGES} images each, {timebin_targets.shape[1]} bins of "
        f"{cfg.timebin_ms:g} ms)"
    )
    # A prediction correlated against one noisy trial is capped at the ratio of
    # signal to single-trial spread, which is the square root of the rep-to-rep
    # reliability. Both single-trial numbers below are read against that.
    single_trial_ceiling = float(np.sqrt(max(reliability, 0.0)))
    print(
        f"   test-target reliability rep-to-rep {reliability:.4f} -> "
        f"single-trial ceiling {single_trial_ceiling:.4f}"
    )
    print(
        "   stim_r on 100 held-out TRAIN images, single trial: "
        f"{summary['heldout_train_single_trial']:.4f} +/- "
        f"{summary['sd_heldout_train_single_trial']:.4f} "
        f"({summary['heldout_train_single_trial'] / single_trial_ceiling:.0%} "
        "of ceiling)"
    )
    print(
        "   stim_r on 100 TEST images,       single trial: "
        f"{summary['test_single_trial']:.4f} +/- "
        f"{summary['sd_test_single_trial']:.4f} "
        f"({summary['test_single_trial'] / single_trial_ceiling:.0%} of ceiling)"
    )
    print(f"   shift cost {summary['shift_cost_single_trial']:+.4f}")
    print(
        "   stim_r on the same TEST images, 30 reps averaged: "
        f"{summary['test_repetition_averaged']:.4f}  "
        "(the averaging, not the shift)"
    )
    penalties = sorted({row["alpha"] for row in holdout["rows"]})
    at_edge = any(row["alpha_at_grid_edge"] for row in holdout["rows"])
    print(
        f"   selected alphas {penalties}"
        + ("  WARNING: a fit sits at a grid edge" if at_edge else "")
    )

    report = {
        "config": asdict(cfg),
        "window_ms": list(covered_ms),
        "protocol": protocol,
        "stimulus_features": features,
        "neural_targets": target_report,
        "single_trial_reliability": reliability,
        "single_trial_ceiling": single_trial_ceiling,
        "matched_holdout": holdout,
    }
    with open(output_dir / "distribution_shift.json", "w") as results_file:
        json.dump(report, results_file, indent=2)
    # end with saved results
    print(f"\nsaved to {output_dir}")
# EOF


if __name__ == "__main__":
    main()
# EOC
