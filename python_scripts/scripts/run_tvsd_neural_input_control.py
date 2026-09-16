"""
Ask what a held-out IT site's image response is predictable from at all.

The image-computable decoders plateau near 0.68 of the noise ceiling. Two very
different things would explain that: the ANN features may fail to span the
subspace IT's image tuning lives in, or each site may carry private tuning that
nothing outside itself predicts. Those are separated by swapping the *input*
and keeping everything else: predict twenty held-out sites from the other three
hundred, from V4, or from I-JEPA features, and score all of them the same way.

The one thing that has to be right is the repetition bookkeeping. Feeding a
site's neighbours from the same trials as the target lets a decoder read shared
trial noise instead of shared tuning, which is a real effect but not one any
image model could ever match, and it inflates stim_r badly. So the neural input
is averaged over one half of each image's repetitions and the target over the
disjoint other half; the same-half version is run alongside only to measure how
large that leak is.

Every model here is a ridge. That is deliberate: the question is what the input
carries, not what a decoder can extract, and holding the mapping fixed and
linear is what makes the three inputs comparable.
"""

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python_scripts" / "src"))

from IT_recap.neural_prediction_training import (  # noqa: E402
    aggregate_trials_by_image,
    split_half_reliability,
    stimulus_correlation,
)
from IT_recap.tvsd_experiments import (  # noqa: E402
    N_TEST_IMAGES,
    load_cached_targets,
    load_project_paths,
    prepare_search_data,
    prepare_timebin_data,
    resolve_device,
)


# Column of ALLMAT holding the repetition index of a presentation.
REPETITION_COLUMN = 3
ALPHAS = np.logspace(1, 9, 33)


@dataclass
class Cfg:
    env: str | None = None
    mua_file_name: str = "f_THINGS_MUA_trials.mat"
    feature_archive_name: str = "tvsd_monkeyF_ijepa_vith14_1k_224_features.npz"
    output_dir: str | None = None

    # The protocol of the decoder experiments this control explains.
    area: str = "IT"
    auxiliary_area: str = "V4"
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

    # Held-out site folds and the repetition split.
    fold_size: int = 20
    n_folds: int = 16
    noise_ceiling_resamples: int = 40

    batch_size: int = 256
    num_workers: int = 0
    device: str = "auto"
    smoke_test: bool = False


"""
parse_args
Parse command-line overrides into the control's configuration.

OUTPUT:
    - cfg: Cfg -> window, fold, and cache settings
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
            parser.add_argument(argument_name, action="store_true")
        elif default is None:
            parser.add_argument(argument_name, default=default)
        else:
            parser.add_argument(argument_name, type=type(default), default=default)
        # end if boolean, optional, or typed argument
    # end for configuration field
    arguments = vars(parser.parse_args())
    arguments["layer_names"] = [
        name.strip() for name in arguments["layer_names"].split(",") if name.strip()
    ]
    return Cfg(**arguments)
# EOF


"""
split_test_repetitions
Assign every test presentation to one of two disjoint repetition halves.

INPUT:
    - repetition_ids: np.ndarray -> repetition index of each test presentation
    - image_ids: np.ndarray -> zero-based image identity of each presentation
    - seed: int -> seed of the per-image repetition shuffle

OUTPUT:
    - first_half: np.ndarray -> boolean mask of the input half
    - second_half: np.ndarray -> boolean mask of the target half
"""
def split_test_repetitions(repetition_ids, image_ids, seed=0):
    generator = np.random.default_rng(seed)
    first_half = np.zeros(len(image_ids), dtype=bool)
    for image_id in np.unique(image_ids):
        rows = np.flatnonzero(image_ids == image_id)
        # Shuffle within the image so the halves are not "early vs late" reps,
        # which would confound the split with adaptation over the session.
        shuffled = generator.permutation(rows)
        first_half[shuffled[: len(shuffled) // 2]] = True
    # end for repeated-test image
    return first_half, ~first_half
# EOF


"""
prepare_ridge
Centre and eigendecompose one input space, once, for reuse across site folds.

Sixteen folds share the same image features and the same V4 input, and the
expensive part of a ridge -- the Gram matrix and its eigendecomposition -- does
not depend on the targets at all. Doing it once turns a search over folds and
penalties into a few matrix products each.

INPUT:
    - fit_features: np.ndarray -> [samples, dimensions] fitting inputs

OUTPUT:
    - prepared: dict -> the centred inputs, their eigenbasis, and the mean
"""
def prepare_ridge(fit_features):
    feature_mean = fit_features.mean(0)
    centered = (fit_features - feature_mean).astype(np.float32)
    gram = (centered.T @ centered).astype(np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    return {
        "feature_mean": feature_mean,
        "centered": centered,
        "eigenvalues": np.maximum(eigenvalues, 0.0),
        "eigenvectors": eigenvectors,
    }
# EOF


"""
ridge_path_from
Predict held-out rows for a grid of penalties from a prepared input space.

INPUT:
    - prepared: dict -> output of prepare_ridge
    - fit_targets: np.ndarray -> [samples, outputs] fitting targets
    - eval_features: np.ndarray -> [samples, dimensions] inputs to predict
    - alphas: np.ndarray -> ridge penalties

OUTPUT:
    - predictions: np.ndarray -> [alphas, samples, outputs]
"""
def ridge_path_from(prepared, fit_targets, eval_features, alphas):
    target_mean = fit_targets.mean(0)
    cross = prepared["centered"].T @ (fit_targets - target_mean).astype(np.float32)
    projected = prepared["eigenvectors"].T @ cross.astype(np.float64)
    centered_eval = eval_features - prepared["feature_mean"]
    return np.stack(
        [
            centered_eval
            @ (
                prepared["eigenvectors"]
                @ (projected / (prepared["eigenvalues"] + alpha)[:, None])
            )
            + target_mean
            for alpha in alphas
        ]
    )
# EOF


"""
fit_and_score_input
Fit one input space to one fold of held-out sites and score it on the target half.

Hyperparameter choice uses the validation presentations of the training images,
never the repeated test images, so the reported stim_r is untouched.

INPUT:
    - prepared: dict -> output of prepare_ridge for this input space
    - fit_targets: np.ndarray -> [presentations, outputs] training targets
    - validation_features: np.ndarray -> validation inputs
    - validation_targets: np.ndarray -> validation targets
    - test_features: np.ndarray -> [images, dimensions] test-half inputs
    - test_targets: np.ndarray -> [images, time, sites] target-half response

OUTPUT:
    - site_correlations: np.ndarray -> [time, sites] stim_r of the fold
    - alpha: float -> the selected penalty
"""
def fit_and_score_input(
    prepared,
    fit_targets,
    validation_features,
    validation_targets,
    test_features,
    test_targets,
):
    n_time, n_sites = test_targets.shape[1], test_targets.shape[2]
    validation_path = ridge_path_from(
        prepared, fit_targets, validation_features, ALPHAS
    )
    errors = ((validation_path - validation_targets) ** 2).mean(axis=(1, 2))
    best = int(errors.argmin())
    predictions = ridge_path_from(
        prepared, fit_targets, test_features, [ALPHAS[best]]
    )[0]
    return (
        stimulus_correlation(
            predictions.reshape(-1, n_time, n_sites), test_targets
        ),
        float(ALPHAS[best]),
    )
# EOF


def main():
    cfg = parse_args()
    if cfg.smoke_test:
        cfg.n_folds = 2
        cfg.noise_ceiling_resamples = 4
    # end if a short run was requested

    paths = load_project_paths(cfg)
    device = resolve_device(cfg.device)
    output_dir = Path(
        cfg.output_dir or PROJECT_ROOT / "results" / "tvsd_neural_input_control"
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = prepare_search_data(cfg, paths, device)
    shapes, indices, allmat = data["shapes"], data["indices"], data["allmat"]
    n_time, n_sites = shapes["n_timepoints"], shapes["n_neurons"]
    subset_targets, subset_features = data["subset_targets"], data["subset_features"]

    # --- the repetition split that keeps input and target independent ---
    test_image_ids = data["scoring"]["test_image_ids"]
    repetition_ids = allmat[indices["test"], REPETITION_COLUMN]
    input_half, target_half = split_test_repetitions(
        repetition_ids, test_image_ids, cfg.random_seed
    )
    test_response = subset_targets["test"]
    input_average = aggregate_trials_by_image(
        test_response[input_half], test_image_ids[input_half], N_TEST_IMAGES
    )
    target_average = aggregate_trials_by_image(
        test_response[target_half], test_image_ids[target_half], N_TEST_IMAGES
    )
    # The ceiling has to describe the half it is used on: reliability of a
    # half-sized average, not of the full 30 repetitions.
    ceiling = split_half_reliability(
        test_response[target_half],
        test_image_ids[target_half],
        N_TEST_IMAGES,
        reducer="mean",
        n_resamples=cfg.noise_ceiling_resamples,
        seed=cfg.random_seed,
    )
    print(
        f"{int(input_half.sum())} input / {int(target_half.sum())} target "
        f"presentations | half-average ceiling {np.nanmean(ceiling):.4f}"
    )

    # --- the auxiliary area, on the identical split ---
    auxiliary_cfg = replace(cfg, area=cfg.auxiliary_area)
    auxiliary_targets, auxiliary_allmat = load_cached_targets(auxiliary_cfg, paths)
    auxiliary = prepare_timebin_data(
        auxiliary_cfg,
        auxiliary_targets,
        data["train_features"],
        data["test_features"],
        auxiliary_allmat,
        device,
    )
    auxiliary_response = auxiliary["subset_targets"]
    auxiliary_input_average = aggregate_trials_by_image(
        auxiliary_response["test"][input_half], test_image_ids[input_half], N_TEST_IMAGES
    )

    flatten = lambda array: array.reshape(len(array), -1)
    image_fit = flatten(subset_features["train"])
    image_validation = flatten(subset_features["validation"])
    image_test = flatten(data["test_features"])
    print(
        f"inputs: image {image_fit.shape[1]} dims | IT neighbours "
        f"{(n_sites - cfg.fold_size) * n_time} | {cfg.auxiliary_area} "
        f"{auxiliary_response['train'].shape[-1] * n_time}"
    )

    generator = np.random.default_rng(cfg.random_seed)
    site_order = generator.permutation(n_sites)
    folds = [
        site_order[start : start + cfg.fold_size]
        for start in range(0, n_sites, cfg.fold_size)
    ][: cfg.n_folds]

    collected = {name: [] for name in
                 ("image", "it_disjoint", "it_same_half", "v4_disjoint")}
    fold_sites, alphas = [], {name: [] for name in collected}
    # The image and V4 inputs are the same for every fold, so their Gram
    # decomposition is done once; only the IT neighbourhood changes per fold.
    shared_ridge = {
        "image": prepare_ridge(image_fit),
        "v4_disjoint": prepare_ridge(flatten(auxiliary_response["train"])),
    }
    for fold_index, held_out in enumerate(folds):
        neighbours = np.setdiff1d(np.arange(n_sites), held_out)
        target_test = target_average[:, :, held_out]
        specifications = {
            "image": (image_validation, image_test),
            "it_disjoint": (
                flatten(subset_targets["validation"][:, :, neighbours]),
                flatten(input_average[:, :, neighbours]),
            ),
            # The leak this control exists to avoid, measured on purpose.
            "it_same_half": (
                flatten(subset_targets["validation"][:, :, neighbours]),
                flatten(target_average[:, :, neighbours]),
            ),
            "v4_disjoint": (
                flatten(auxiliary_response["validation"]),
                flatten(auxiliary_input_average),
            ),
        }
        neighbour_ridge = prepare_ridge(
            flatten(subset_targets["train"][:, :, neighbours])
        )
        for name, (validation, test) in specifications.items():
            correlations, alpha = fit_and_score_input(
                shared_ridge.get(name, neighbour_ridge),
                flatten(subset_targets["train"][:, :, held_out]),
                validation,
                flatten(subset_targets["validation"][:, :, held_out]),
                test,
                target_test,
            )
            collected[name].append(correlations)
            alphas[name].append(alpha)
        # end for input space
        fold_sites.append(held_out)
        print(
            f"  fold {fold_index + 1:2d}/{len(folds)} | "
            + " | ".join(
                f"{name} {np.nanmean(collected[name][-1]):.4f}" for name in collected
            ),
            flush=True,
        )
    # end for held-out site fold

    all_sites = np.concatenate(fold_sites)
    fold_ceiling = ceiling[:, all_sites]
    summary = {}
    for name, per_fold in collected.items():
        correlations = np.concatenate(per_fold, axis=1)
        summary[name] = {
            "mean_stim_r": round(float(np.nanmean(correlations)), 4),
            "fraction_of_ceiling": round(
                float(np.nanmean(correlations) / np.nanmean(fold_ceiling)), 4
            ),
            "median_alpha": float(np.median(alphas[name])),
        }
    # end for input space
    summary["ceiling"] = {"mean_stim_r": round(float(np.nanmean(fold_ceiling)), 4)}

    with open(output_dir / "config.json", "w") as config_file:
        json.dump(asdict(cfg), config_file, indent=2)
    # end with saved configuration
    with open(output_dir / "results.json", "w") as results_file:
        json.dump(summary, results_file, indent=2)
    # end with saved metrics
    np.savez_compressed(
        output_dir / "site_stim_r.npz",
        sites=all_sites,
        ceiling=fold_ceiling,
        **{name: np.concatenate(per_fold, axis=1) for name, per_fold in collected.items()},
    )

    print(f"\n{'input':<28}{'stim_r':>9}{'% of ceiling':>15}")
    print("-" * 52)
    for name in ("image", "v4_disjoint", "it_disjoint", "it_same_half"):
        row = summary[name]
        print(
            f"{name:<28}{row['mean_stim_r']:>9.4f}"
            f"{row['fraction_of_ceiling'] * 100:>14.1f}%"
        )
    # end for input space
    print(f"{'ceiling (half average)':<28}{summary['ceiling']['mean_stim_r']:>9.4f}")
    print(f"\nsaved to {output_dir}")
# EOF


if __name__ == "__main__":
    main()
