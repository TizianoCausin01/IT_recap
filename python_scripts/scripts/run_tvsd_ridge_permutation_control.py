"""
Fit the joint ridge on a broken feature-response coupling and score it.

Every number this project reports is a distance from a ceiling, never from a
floor: the reported MSE of ~0.095 means little without knowing what the same
ridge scores when its design matrix cannot possibly explain the response. This
script supplies that floor. The fit split's feature rows are permuted against
their targets, so a presentation is fitted to another image's response, while
validation and test keep their true pairing -- the map is learned from noise and
then asked to predict real data.

Everything else is the protocol of run_tvsd_ridge_stacking.py: the same window,
the same seeded split, the same train-only standardization, and the same
image-averaged scoring, so the permuted rows drop straight into that table.

Two references are printed beside the permutations: the true-coupling ridge, and
the constant model that predicts the fit split's mean response for every image,
which is where a permuted fit is expected to land.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "python_scripts" / "src")
)

from IT_recap.tvsd_experiments import (  # noqa: E402
    PROJECT_ROOT,
    fit_ridge_from_presentations,
    load_cached_data,
    load_project_paths,
    prepare_timebin_data,
    resolve_device,
    score_and_decompose,
)


REDUCER = "mean"


@dataclass
class Cfg:
    env: str | None = None
    mua_file_name: str = "f_THINGS_MUA_trials.mat"
    feature_archive_name: str = "tvsd_monkeyF_ijepa_vith14_1k_224_features.npz"
    output_dir: str | None = None

    # Identical to the stacking experiment, so the rows are comparable.
    area: str = "IT"
    target_fs: int = 100
    time_start_ms: float = 0.0
    time_end_ms: float = 200.0
    window_start_ms: float = 80.0
    window_end_ms: float = 180.0
    timebin_ms: float = 20.0
    model_name: str = "ijepa_vith14_1k"
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
    validation_fraction: float = 0.1
    random_seed: int = 0

    # One permutation is enough for the floor; more of them show its spread.
    n_permutations: int = 3

    # Loader settings prepare_timebin_data expects; nothing here trains.
    batch_size: int = 256
    num_workers: int = 0
    noise_ceiling_resamples: int = 40
    device: str = "cpu"
    smoke_test: bool = False


"""
parse_args
Parse command-line overrides into a configuration object.

OUTPUT:
    - cfg: Cfg -> caches, window, and permutation settings
"""
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    default_layer_names = Cfg.__dataclass_fields__["layer_names"].default_factory()
    for field_name, field_definition in Cfg.__dataclass_fields__.items():
        if field_name == "layer_names":
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
        name.strip() for name in arguments["layer_names"].split(",") if name.strip()
    ]
    return Cfg(**arguments)
# EOF


"""
fit_and_score_ridge
Fit the joint ridge on the given fit-split inputs and score it on the test pool.

INPUT:
    - subset_inputs: dict -> split name to [presentations, features]
    - subset_targets: dict -> split name to [presentations, time, neurons]
    - scoring: dict -> test image ids, ceiling, and response slice
    - name: str -> label for the scored row

OUTPUT:
    - row: dict -> metrics, alpha, and fit time
"""
def fit_and_score_ridge(subset_inputs, subset_targets, scoring, name):
    fit_start = time.perf_counter()
    ridge_fit = fit_ridge_from_presentations(subset_inputs, subset_targets)
    row, _ = score_and_decompose(
        ridge_fit["trial_predictions"],
        subset_targets["test"],
        scoring,
        name,
        REDUCER,
    )
    row.update(
        {
            "ridge_alpha": ridge_fit["alpha"],
            "validation_mse": round(ridge_fit["validation_mse"], 5),
            "fit_seconds": round(time.perf_counter() - fit_start, 1),
        }
    )
    return row
# EOF


def main():
    cfg = parse_args()
    paths = load_project_paths(cfg)
    device = resolve_device(cfg.device)
    output_dir = Path(
        cfg.output_dir
        or PROJECT_ROOT / "results" / f"tvsd_ridge_permutation_{cfg.model_name}"
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    targets, train_features, test_features, allmat = load_cached_data(cfg, paths)
    data = prepare_timebin_data(
        cfg, targets, train_features, test_features, allmat, device
    )
    subset_targets, scoring = data["subset_targets"], data["scoring"]

    # One design matrix per split, all six depths concatenated, exactly as the
    # joint ridge of the stacking experiment sees them.
    subset_inputs = {
        split_name: split_features.reshape(len(split_features), -1)
        for split_name, split_features in data["subset_features"].items()
    }

    rows = []
    print("fitting the true-coupling ridge")
    rows.append(
        fit_and_score_ridge(
            subset_inputs, subset_targets, scoring, "ridge (true coupling)"
        )
    )
    print(f"  test MSE {rows[-1]['test_mse']:.5f}")

    # The constant model: every test image gets the fit split's mean response.
    # A permuted fit has no reason to do better than this, and its distance from
    # it is the only thing a permutation can accidentally gain.
    mean_predictions = np.repeat(
        scoring["fit_target_mean"][None], len(subset_targets["test"]), axis=0
    )
    mean_row, _ = score_and_decompose(
        mean_predictions, subset_targets["test"], scoring, "fit-split mean", REDUCER
    )
    rows.append(mean_row)

    for permutation in range(cfg.n_permutations):
        # The fit split's feature rows are permuted against their targets, so
        # each presentation is fitted to another image's response. Validation and
        # test keep the true pairing: the broken map meets real data.
        permutation_rng = np.random.default_rng(cfg.random_seed + permutation)
        shuffled_rows = permutation_rng.permutation(len(subset_inputs["train"]))
        shuffled_inputs = dict(subset_inputs)
        shuffled_inputs["train"] = subset_inputs["train"][shuffled_rows]
        print(f"fitting permutation {permutation + 1}/{cfg.n_permutations}")
        rows.append(
            fit_and_score_ridge(
                shuffled_inputs,
                subset_targets,
                scoring,
                f"ridge (shuffled coupling {permutation})",
            )
        )
        print(f"  test MSE {rows[-1]['test_mse']:.5f}")
    # end for permutation

    with open(output_dir / "config.json", "w") as config_file:
        json.dump(asdict(cfg), config_file, indent=2)
    # end with saved configuration
    with open(output_dir / "results.json", "w") as results_file:
        json.dump({"rows": rows}, results_file, indent=2)
    # end with saved metrics

    header = (
        f"{'model':<36}{'test MSE':>10}{'stim_r':>9}{'var expl':>10}"
        f"{'alpha':>12}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        alpha = row.get("ridge_alpha")
        print(
            f"{row['model']:<36}{row['test_mse']:>10.5f}"
            f"{row['mean_stim_r_response']:>9.4f}"
            f"{row['variance_explained_vs_mean']:>10.4f}"
            f"{'' if alpha is None else f'{alpha:.3g}':>12}"
        )
    # end for scored row
    print(
        "\nvar expl is 1 - MSE / (MSE of predicting the fit-split mean), so a "
        "permuted fit scoring zero there has learned nothing at all."
    )
    print(f"\nsaved to {output_dir}")
# EOF


if __name__ == "__main__":
    main()
# EOC
