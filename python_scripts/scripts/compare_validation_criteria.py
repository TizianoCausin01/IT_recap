"""
Score ridge, the LDS and the GRU under alternative validation criteria.

The criterion these experiments select on -- validation stim_r, a correlation
across the 2,225 single-trial unique-image presentations -- resolves the gaps
between architectures but not the gaps within one, so a natural question is
whether a per-image criterion does better. This script fits the three current
best image-only models and scores their *validation* predictions four ways:

    - stim_r, the incumbent: correlate across images for each (bin, site);
    - peak population pattern: correlate the 320-site pattern within each image
      and bin, keep each image's best bin, average over images;
    - mean population pattern: the same, averaging the bins instead;
    - whole-pattern: one correlation per image over the full bin x site vector.

Each pattern criterion is computed twice, with and without removing every
(bin, site) mean over images first. Pearson centres along the axis it
correlates over, which for a per-image pattern is the site axis, so the per-site
offsets survive unless they are taken out explicitly; the gap between the two
columns is how much of an uncentred score is that nuisance term.

The point of the table is not the absolute values but whether a criterion
reproduces, on validation alone, the ordering the repetition-averaged test
metrics give -- which is what a selection criterion is for.
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch


sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "python_scripts" / "src")
)

from IT_recap.neural_prediction_training import (  # noqa: E402
    disjoint_half_stimulus_correlation,
    optimally_rescaled_mse,
    population_pattern_correlation,
    stimulus_correlation,
)
from IT_recap.tvsd_experiments import (  # noqa: E402
    N_TEST_IMAGES,
    PROJECT_ROOT,
    evaluate_cached_feature_decoder,
    fit_ridge_from_presentations,
    load_cached_targets,
    load_pooled_spatial_features,
    load_project_paths,
    make_multi_input_loader,
    prepare_timebin_data,
    resolve_device,
    score_and_decompose,
    standardize_features_on_fit_split,
    train_cached_feature_decoder,
)
from model_classes.early_response_models import (  # noqa: E402
    build_early_response_model,
)


# The three current best image-only models, at the settings the search and the
# penalty sweep selected. Ridge carries no architecture arguments.
MODEL_SPECS = {
    "lds": {
        "state_dim": 256,
        "dropout": 0.4,
        "input_noise_std": 0.5,
        "learning_rate": 1e-4,
        "weight_decay": 1e-1,
        "l2_lambda": 1e-3,
    },
    "gru": {
        "hidden_dim": 1024,
        "time_embedding_dim": 64,
        "dropout": 0.3,
        "input_noise_std": 0.0,
        "learning_rate": 3e-4,
        "weight_decay": 3e-1,
        "l2_lambda": 1e-3,
    },
}
MODEL_LABELS = {
    "ridge": "Ridge",
    "lds": "LDS + penalty",
    "gru": "GRU + penalty",
}

# Criteria minimized rather than maximized, so the table picks their winner
# by the smallest value.
LOWER_IS_BETTER = ("mse", "mse_rescaled", "best_bin_mse")

# Settings the trainer reads off cfg rather than the model constructor.
OPTIMIZER_HYPERPARAMETERS = ("learning_rate", "weight_decay", "l2_lambda")


@dataclass
class Cfg:
    env: str | None = None
    mua_file_name: str = "f_THINGS_MUA_trials.mat"
    spatial_stem: str = "tvsd_monkeyF_alexnet_features_11_spatial"
    model_name: str = "alexnet_conv5"
    output_dir: str | None = None

    area: str = "IT"
    target_fs: int = 100
    time_start_ms: float = 0.0
    time_end_ms: float = 200.0
    window_start_ms: float = 80.0
    window_end_ms: float = 180.0
    timebin_ms: float = 20.0
    spatial_pool_size: int = 2

    validation_fraction: float = 0.1
    random_seed: int = 0
    init_seed: int = -1

    # Penalty scope is shared by both decoders; the weight is per model.
    l2_scope: str = "readout"
    l2_lambda: float = 0.0
    temporal_noise_std: float = 0.0
    early_dim: int = 64

    batch_size: int = 256
    num_workers: int = 0
    epochs: int = 100
    minimum_epochs: int = 15
    patience: int = 12
    gradient_clip: float = 1.0
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    selection_metric: str = "stim_r"

    noise_ceiling_resamples: int = 40
    device: str = "auto"
    smoke_test: bool = False


"""
parse_args
Parse command-line overrides into a configuration object.

OUTPUT:
    - cfg: Cfg -> data, model, and optimization settings
"""
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for field_name, field_definition in Cfg.__dataclass_fields__.items():
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
    return Cfg(**vars(parser.parse_args()))
# EOF


"""
validation_criteria
Score one model's validation predictions under every compared criterion.

INPUT:
    - predictions: np.ndarray -> [presentations, time, channels]
    - targets: np.ndarray -> [presentations, time, channels]

OUTPUT:
    - criteria: dict -> criterion name to its scalar value
"""
def validation_criteria(predictions, targets):
    criteria = {
        "stim_r": float(
            np.nanmean(stimulus_correlation(predictions, targets))
        ),
        "mse": float(np.mean((predictions - targets) ** 2)),
        "mse_rescaled": optimally_rescaled_mse(predictions, targets),
        # The MSE counterpart of the peak-pattern criterion: score each image
        # at the bin it is predicted best in, then average over images.
        "best_bin_mse": float(
            np.mean(np.min(((predictions - targets) ** 2).mean(axis=2), axis=1))
        ),
    }
    for centered, tag in ((True, "centred"), (False, "raw")):
        pattern = population_pattern_correlation(
            predictions, targets, center_across_images=centered
        )
        criteria[f"peak_pattern_{tag}"] = float(
            np.nanmean(np.nanmax(pattern, axis=1))
        )
        criteria[f"mean_pattern_{tag}"] = float(np.nanmean(pattern))
        # The whole-pattern variant correlates one flattened vector per image,
        # which is the same computation with the time axis folded in.
        flat_predictions = predictions.reshape(len(predictions), 1, -1)
        flat_targets = targets.reshape(len(targets), 1, -1)
        criteria[f"whole_pattern_{tag}"] = float(
            np.nanmean(
                population_pattern_correlation(
                    flat_predictions, flat_targets, center_across_images=centered
                )
            )
        )
    # end for centring choice
    return criteria
# EOF


"""
test_metrics
Score one model's test predictions the two ways this project reports.

INPUT:
    - predictions: np.ndarray -> [presentations, time, channels]
    - targets: np.ndarray -> [presentations, time, channels]
    - scoring: dict -> test image ids, ceiling, response slice
    - name: str -> label written into the row
    - cfg: Cfg -> resample settings of the cross-half metric

OUTPUT:
    - row: dict -> test stim_r, its decomposition, and the cross-half score
"""
def test_metrics(predictions, targets, scoring, name, cfg):
    row = score_and_decompose(predictions, targets, scoring, name)[0]
    correlations = disjoint_half_stimulus_correlation(
        predictions,
        targets,
        scoring["test_image_ids"],
        N_TEST_IMAGES,
        n_resamples=cfg.noise_ceiling_resamples,
        seed=cfg.random_seed,
    )
    row["disjoint_half_stim_r"] = round(
        float(np.nanmean(correlations[scoring["response_slice"]])), 4
    )
    return row
# EOF


"""
fit_decoder
Train one decoder at its selected configuration and return its predictions.

INPUT:
    - architecture: str -> "lds" or "gru"
    - loaders: dict -> train, validation, and test loaders
    - shapes: dict -> n_layers, feature_dim, n_timepoints, and n_neurons
    - cfg: Cfg -> settings the model specification does not carry
    - device: torch.device -> compute device

OUTPUT:
    - predictions: dict -> "validation" and "test" prediction arrays
    - targets: dict -> the matching target arrays
"""
def fit_decoder(architecture, loaders, shapes, cfg, device):
    specification = MODEL_SPECS[architecture]
    model_kwargs = {
        name: value
        for name, value in specification.items()
        if name not in OPTIMIZER_HYPERPARAMETERS
    }
    torch.manual_seed(
        cfg.random_seed if cfg.init_seed < 0 else cfg.init_seed
    )
    model = build_early_response_model(
        architecture,
        **shapes,
        **model_kwargs,
        n_early_bins=1,
        early_dim=cfg.early_dim,
        use_early_response=False,
        temporal_noise_std=cfg.temporal_noise_std,
    ).to(device)

    run_cfg = replace(
        cfg,
        **{
            name: specification[name]
            for name in OPTIMIZER_HYPERPARAMETERS
            if name in specification
        },
    )
    train_cached_feature_decoder(model, loaders, run_cfg, device, verbose=False)

    predictions, targets = {}, {}
    for subset_name in ("validation", "test"):
        predictions[subset_name], targets[subset_name] = (
            evaluate_cached_feature_decoder(model, loaders[subset_name], device)
        )
    # end for scored subset
    return predictions, targets
# EOF


def main():
    cfg = parse_args()
    if cfg.smoke_test:
        cfg.epochs = min(cfg.epochs, 3)
        cfg.minimum_epochs = 1
        cfg.patience = 1
        cfg.noise_ceiling_resamples = 4
    # end if a smoke run was requested

    paths = load_project_paths(cfg)
    device = resolve_device(cfg.device)
    output_dir = Path(
        cfg.output_dir
        or PROJECT_ROOT / "results" / f"tvsd_validation_criteria_{cfg.model_name}"
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    targets_cache, allmat = load_cached_targets(cfg, paths)
    pooled_features, _ = load_pooled_spatial_features(
        cfg, paths, cfg.spatial_pool_size
    )
    data = prepare_timebin_data(
        cfg,
        targets_cache,
        pooled_features["train"],
        pooled_features["test"],
        allmat,
        device,
    )
    shapes, scoring = data["shapes"], data["scoring"]
    subset_targets = data["subset_targets"]
    subset_features = standardize_features_on_fit_split(data["subset_features"])

    # Every decoder here is image-only, so the early-response input exists only
    # to satisfy the shared signature and is never read.
    zero_early = {
        subset_name: np.zeros(
            (len(features), 1, shapes["n_neurons"]), dtype=np.float32
        )
        for subset_name, features in subset_features.items()
    }
    loaders = {
        subset_name: make_multi_input_loader(
            [subset_features[subset_name], zero_early[subset_name]],
            subset_targets[subset_name],
            cfg,
            shuffle=subset_name == "train",
        )
        for subset_name in ("train", "validation", "test")
    }

    rows = {}
    print("\nfitting ridge")
    design = {
        subset_name: features.reshape(len(features), -1)
        for subset_name, features in subset_features.items()
    }
    ridge_fit = fit_ridge_from_presentations(design, subset_targets)
    rows["ridge"] = {
        "validation": validation_criteria(
            ridge_fit["validation_predictions"], subset_targets["validation"]
        ),
        "test": test_metrics(
            ridge_fit["trial_predictions"],
            subset_targets["test"],
            scoring,
            "ridge",
            cfg,
        ),
    }
    for architecture in MODEL_SPECS:
        print(f"fitting {architecture}")
        predictions, model_targets = fit_decoder(
            architecture, loaders, shapes, cfg, device
        )
        rows[architecture] = {
            "validation": validation_criteria(
                predictions["validation"], model_targets["validation"]
            ),
            "test": test_metrics(
                predictions["test"],
                model_targets["test"],
                scoring,
                architecture,
                cfg,
            ),
        }
    # end for fitted decoder

    with open(output_dir / "validation_criteria.json", "w") as results_file:
        json.dump({"config": asdict(cfg), "rows": rows}, results_file, indent=2)
    # end with saved results

    # --- the tables ---
    criterion_names = list(rows["ridge"]["validation"])
    models = list(rows)
    header = f"{'criterion':<24}" + "".join(
        f"{MODEL_LABELS[name]:>16}" for name in models
    ) + f"{'winner':>16}{'matches test':>14}"
    print("\nValidation criteria (2,225 single-trial unique-image presentations)")
    print(header)
    print("-" * len(header))

    test_order = sorted(
        models, key=lambda name: -rows[name]["test"]["mean_stim_r_response"]
    )
    for criterion in criterion_names:
        values = {name: rows[name]["validation"][criterion] for name in models}
        sign = 1.0 if criterion in LOWER_IS_BETTER else -1.0
        order = sorted(models, key=lambda name: sign * values[name])
        print(
            f"{criterion:<24}"
            + "".join(f"{values[name]:>16.4f}" for name in models)
            + f"{MODEL_LABELS[order[0]]:>16}"
            + f"{'yes' if order == test_order else 'no':>14}"
        )
    # end for criterion

    print("\nTest metrics, for reference (100 images x 30 repetitions)")
    print(
        f"{'model':<18}{'stim_r':>10}{'cross-half':>13}{'MSE':>10}"
        f"{'MSE rescaled':>15}"
    )
    print("-" * 66)
    for name in test_order:
        row = rows[name]["test"]
        print(
            f"{MODEL_LABELS[name]:<18}{row['mean_stim_r_response']:>10.4f}"
            f"{row['disjoint_half_stim_r']:>13.4f}{row['test_mse']:>10.5f}"
            f"{row['mse_after_rescaling']:>15.5f}"
        )
    # end for model
    print(f"\nsaved to {output_dir}")
# EOF


if __name__ == "__main__":
    main()
# EOC
