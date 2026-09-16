"""
Stack a decoder on per-depth ridge maps and ask whether that beats ridge.

The architecture search hands every decoder the same concatenated ANN features
and lets it find its own linear map. This script asks the opposite question:
give the linear map away for free -- one ridge per depth, each already
predicting the complete [time, sites] response -- and let the decoder work in
the space of those predictions instead. Its input axis then holds L guesses at
the target rather than L feature vectors, so all it has left to learn is which
depth to believe in which time bin, plus whatever nonlinear correction the
ridge maps cannot express. The learned time code is jittered per image, which
is the regularizer that helped this decoder family before.

The level-one maps are cross-fitted over the fit split: every fit presentation
is predicted by a ridge that never saw its image. Skipping that would train the
stack on level-one predictions far better than the ones it meets at test.

Five controls decide what any win means:
    - the joint ridge over all depths at once, the model to beat;
    - the best single depth map, since depths that agree leave nothing to stack;
    - the plain mean of the level-one maps, which costs no parameters;
    - the same decoder on the raw ANN features, so a win is attributable to the
      stacking rather than to the decoder;
    - the decoder trained on what a map leaves over rather than replacing it,
      applied both to the mean of the depth maps and to the joint ridge itself.

Every run is scored on the untouched 100 repeated test images and repeated over
several seeded splits, because this project's validation split is far too small
to separate decoders on its own. Finished runs are written under ``runs/`` and
reloaded rather than repeated, so an interrupted sweep resumes.
"""

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "python_scripts" / "src")
)

from IT_recap.tvsd_experiments import (  # noqa: E402
    PROJECT_ROOT,
    evaluate_cached_feature_decoder,
    fit_ridge_from_presentations,
    load_cached_data,
    load_project_paths,
    make_tensor_loader,
    prepare_timebin_data,
    resolve_device,
    score_and_decompose,
    train_cached_feature_decoder,
)
from IT_recap.tvsd_stacking import (  # noqa: E402
    average_stacked_predictions,
    build_ridge_stack,
    cross_fit_layer_ridge,
)
from model_classes.timebin_models import build_timebin_model  # noqa: E402


# Only the repetition mean has a defined Spearman-Brown noise ceiling.
REDUCER = "mean"

# The parameter-free references, in the order their per-site correlations are
# stacked into one array on disk.
REFERENCE_RUN_IDS = ("ridge_joint", "ridge_stack_mean", "ridge_best_layer")


@dataclass
class Cfg:
    # Caches.
    env: str | None = None
    mua_file_name: str = "f_THINGS_MUA_trials.mat"
    feature_archive_name: str = "tvsd_monkeyF_ijepa_vith14_1k_224_features.npz"
    output_dir: str | None = None

    # Target identity and window, identical to the architecture search.
    area: str = "IT"
    target_fs: int = 100
    time_start_ms: float = 0.0
    time_end_ms: float = 200.0
    window_start_ms: float = 80.0
    window_end_ms: float = 180.0
    timebin_ms: float = 20.0
    model_name: str = "ijepa_vith14_1k"
    # All six cached depths: the more level-one maps disagree, the more there is
    # for a stack to arbitrate between.
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

    # Replication. Each seed reshuffles the fit/validation split as well as the
    # weights, so a variant's spread across seeds is the honest error bar.
    random_seed: int = 0
    n_seeds: int = 3
    cross_fit_folds: int = 3

    # The stacked decoder. "baseline" is the notebook's layer-attention model,
    # the one whose time code the temporal noise perturbs.
    architecture: str = "baseline"
    time_embedding_dim: int = 64
    value_dim: int = 256
    mlp_hidden_dim: int = 256
    dropout: float = 0.3
    input_noise_std: float = 0.0
    temporal_noise_levels: str = "0.0,0.1,0.25,0.5"
    # The noise level the controls are run at, so they are matched to the
    # middle of the sweep rather than to its best outcome.
    reference_temporal_noise: float = 0.25
    # Correcting the joint ridge costs one extra cross-fitted joint ridge per
    # seed, which is the most expensive fit in the script.
    include_joint_residual: bool = True

    # Optimization, shared by every trained variant.
    batch_size: int = 256
    num_workers: int = 0
    epochs: int = 50
    minimum_epochs: int = 10
    patience: int = 10
    gradient_clip: float = 1.0
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    l2_lambda: float = 0.0
    l2_scope: str = "readout"
    selection_metric: str = "stim_r"

    # Evaluation and bookkeeping.
    noise_ceiling_resamples: int = 40
    device: str = "auto"
    smoke_test: bool = False


"""
parse_args
Parse command-line overrides into a configuration object.

OUTPUT:
    - cfg: Cfg -> caches, run matrix, and optimization settings
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
build_run_matrix
List the trained variants, each one a dict the training function reads.

INPUT:
    - cfg: Cfg -> noise sweep and the level the controls are matched to

OUTPUT:
    - runs: list[dict] -> name, input source, target mode, and model settings
"""
def build_run_matrix(cfg):
    noise_levels = [
        float(level)
        for level in cfg.temporal_noise_levels.split(",")
        if level.strip()
    ]
    runs = [
        {
            "name": f"stack tnoise {level:g}",
            "run_id": f"stack_tnoise_{level:g}",
            "source": "stack",
            "target_mode": "replace",
            "temporal_noise_std": level,
            "normalize_features": False,
        }
        for level in noise_levels
    ]
    # The level-one maps are already in the target's units, so the decoder's
    # input LayerNorm is off above. This control turns it back on to show what
    # normalizing a prediction away costs.
    runs.append(
        {
            "name": f"stack normalized tnoise {cfg.reference_temporal_noise:g}",
            "run_id": "stack_normalized",
            "source": "stack",
            "target_mode": "replace",
            "temporal_noise_std": cfg.reference_temporal_noise,
            "normalize_features": True,
        }
    )
    # The same decoder on the raw features: without this a win over ridge says
    # nothing about the stacking itself.
    runs.append(
        {
            "name": f"features tnoise {cfg.reference_temporal_noise:g}",
            "run_id": "features",
            "source": "features",
            "target_mode": "replace",
            "temporal_noise_std": cfg.reference_temporal_noise,
            "normalize_features": True,
        }
    )
    # Correcting the mean level-one map instead of replacing it.
    runs.append(
        {
            "name": f"stack residual tnoise {cfg.reference_temporal_noise:g}",
            "run_id": "stack_residual",
            "source": "stack",
            "target_mode": "residual",
            "temporal_noise_std": cfg.reference_temporal_noise,
            "normalize_features": False,
        }
    )
    # The same correction applied to the joint ridge, which is the model to
    # beat: its test map is the reference row itself, so this run can only lose
    # to ridge by as much as its own correction is wrong.
    if cfg.include_joint_residual:
        runs.append(
            {
                "name": f"ridge residual tnoise {cfg.reference_temporal_noise:g}",
                "run_id": "ridge_residual",
                "source": "stack",
                "target_mode": "residual_joint",
                "temporal_noise_std": cfg.reference_temporal_noise,
                "normalize_features": False,
            }
        )
    # end if the joint ridge is corrected as well
    return runs
# EOF


"""
release_device_memory
Return a finished run's cached accelerator memory to the driver.

INPUT:
    - device: torch.device -> compute device

OUTPUT:
    - None
"""
def release_device_memory(device):
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    # end if the device caches its allocations
# EOF


"""
train_one_variant
Train one decoder on one input source and score it on the repeated test images.

A residual run is trained on what its base map -- the mean of the level-one
maps, or the joint ridge -- leaves over, and predicts that map plus the
decoder's output, so it can only depart from the map by as much as it learns.

INPUT:
    - variant: dict -> one row of the run matrix
    - cfg: Cfg -> optimization settings
    - inputs: dict -> split name to [presentations, groups, embedding]
    - subset_targets: dict -> split name to [presentations, time, neurons]
    - base_predictions: dict | None -> split name to the map this run corrects,
      or None when the decoder predicts the response outright
    - scoring: dict -> test image ids, ceiling, and response slice
    - device: torch.device -> compute device

OUTPUT:
    - row: dict -> metrics of this run
    - site_correlations: np.ndarray -> [time, channels] test stim_r
    - history: list[dict] -> per-epoch optimization record
"""
def train_one_variant(
    variant, cfg, inputs, subset_targets, base_predictions, scoring, device
):
    # A residual run fits the part of the response its base map misses.
    if base_predictions is None:
        training_targets = subset_targets
    else:
        training_targets = {
            split_name: split_targets - base_predictions[split_name]
            for split_name, split_targets in subset_targets.items()
        }
    # end if this run corrects a map rather than replacing it

    loaders = {
        split_name: make_tensor_loader(
            inputs[split_name],
            training_targets[split_name],
            cfg,
            shuffle=split_name == "train",
        )
        for split_name in ("train", "validation", "test")
    }
    n_timepoints, n_neurons = subset_targets["train"].shape[1:]

    # Every variant starts from the same initialization stream within a seed.
    torch.manual_seed(cfg.random_seed)
    model = build_timebin_model(
        cfg.architecture,
        n_layers=inputs["train"].shape[1],
        feature_dim=inputs["train"].shape[2],
        n_timepoints=n_timepoints,
        n_neurons=n_neurons,
        time_embedding_dim=cfg.time_embedding_dim,
        value_dim=cfg.value_dim,
        mlp_hidden_dim=cfg.mlp_hidden_dim,
        dropout=cfg.dropout,
        input_noise_std=cfg.input_noise_std,
        temporal_noise_std=variant["temporal_noise_std"],
        normalize_features=variant["normalize_features"],
    ).to(device)

    run_start = time.perf_counter()
    (
        history,
        best_epoch,
        best_validation_mse,
        best_validation_stim_r,
    ) = train_cached_feature_decoder(model, loaders, cfg, device, verbose=False)
    train_seconds = time.perf_counter() - run_start

    test_predictions, _ = evaluate_cached_feature_decoder(
        model, loaders["test"], device
    )
    if base_predictions is not None:
        test_predictions = test_predictions + base_predictions["test"]
    # end if the corrected map has to be added back

    row, site_correlations = score_and_decompose(
        test_predictions, subset_targets["test"], scoring, variant["name"], REDUCER
    )
    row.update(
        {
            "run_id": variant["run_id"],
            "seed": cfg.random_seed,
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "temporal_noise_std": variant["temporal_noise_std"],
            "normalize_features": variant["normalize_features"],
            "target_mode": variant["target_mode"],
            "best_epoch": best_epoch,
            # Selection ran on the residual for a residual run, so this number
            # is comparable within a run's own family, not across them.
            "validation_mse": round(best_validation_mse, 5),
            "validation_stim_r": round(best_validation_stim_r, 4),
            "train_seconds": round(train_seconds, 1),
        }
    )
    del model
    release_device_memory(device)
    return row, site_correlations, history
# EOF


"""
save_finished_run
Store one finished run so a rerun reuses it instead of repeating it.

INPUT:
    - run_dir: Path -> directory holding one file per finished run
    - run_key: str -> identifier of the run, seed included
    - row: dict -> the run's metrics
    - site_correlations: np.ndarray -> [time, channels] test stim_r
    - history: list[dict] | None -> per-epoch record, absent for closed forms

OUTPUT:
    - None: writes the run's record and its per-site correlations
"""
def save_finished_run(run_dir, run_key, row, site_correlations, history=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / f"{run_key}_site_stim_r.npy", site_correlations)
    with open(run_dir / f"{run_key}.json", "w") as run_file:
        json.dump({"row": row, "history": history}, run_file, indent=2)
    # end with saved run record
# EOF


"""
load_finished_run
Read one stored run back, or report that it has not been run yet.

INPUT:
    - run_dir: Path -> directory holding one file per finished run
    - run_key: str -> identifier of the requested run

OUTPUT:
    - finished: tuple | None -> (row, site_correlations, history), or None
"""
def load_finished_run(run_dir, run_key):
    record_path = run_dir / f"{run_key}.json"
    site_path = run_dir / f"{run_key}_site_stim_r.npy"
    if not (record_path.is_file() and site_path.is_file()):
        return None
    # end if this run has not finished before
    with open(record_path, "r") as run_file:
        record = json.load(run_file)
    # end with stored run record
    return record["row"], np.load(site_path), record["history"]
# EOF


"""
score_closed_form_references
Score the parameter-free references: the joint ridge and the level-one maps.

INPUT:
    - cfg: Cfg -> nothing beyond the seed reported on each row
    - subset_features: dict -> split name to [presentations, layers, embedding]
    - subset_targets: dict -> split name to [presentations, time, neurons]
    - stacked: dict -> split name to [presentations, layers, time * neurons]
    - layer_diagnostics: list[dict] -> per-depth alphas and validation scores
    - scoring: dict -> test image ids, ceiling, and response slice

OUTPUT:
    - rows: list[dict] -> one row per reference
    - site_correlations: dict -> run id to [time, channels] test stim_r
"""
def score_closed_form_references(
    cfg, subset_features, subset_targets, stacked, layer_diagnostics, scoring
):
    n_timepoints, n_neurons = subset_targets["train"].shape[1:]
    rows, site_correlations = [], {}

    # The joint ridge: every depth concatenated into one design matrix. With an
    # image-only input this is exactly the notebook's ridge reference.
    joint_start = time.perf_counter()
    joint_ridge = fit_ridge_from_presentations(
        {
            split_name: split_features.reshape(len(split_features), -1)
            for split_name, split_features in subset_features.items()
        },
        subset_targets,
    )
    joint_row, joint_site_r = score_and_decompose(
        joint_ridge["trial_predictions"],
        subset_targets["test"],
        scoring,
        "ridge (all depths jointly)",
        REDUCER,
    )
    joint_row.update(
        {
            "run_id": "ridge_joint",
            "seed": cfg.random_seed,
            "trainable_parameters": joint_ridge["n_coefficients"],
            "validation_mse": round(joint_ridge["validation_mse"], 5),
            "ridge_alpha": joint_ridge["alpha"],
            "train_seconds": round(time.perf_counter() - joint_start, 1),
        }
    )
    rows.append(joint_row)
    site_correlations["ridge_joint"] = joint_site_r

    # The mean of the level-one maps: stacking with no model on top.
    mean_row, mean_site_r = score_and_decompose(
        average_stacked_predictions(stacked["test"], n_timepoints, n_neurons),
        subset_targets["test"],
        scoring,
        "mean of the depth maps",
        REDUCER,
    )
    mean_row.update({"run_id": "ridge_stack_mean", "seed": cfg.random_seed})
    rows.append(mean_row)
    site_correlations["ridge_stack_mean"] = mean_site_r

    # The single best depth, chosen on validation stim_r. If the joint ridge
    # does not beat this, the depths carry no complementary information and a
    # stack has nothing to arbitrate.
    best_layer = max(
        layer_diagnostics, key=lambda diagnostics: diagnostics["validation_stim_r"]
    )["layer"]
    best_layer_row, best_layer_site_r = score_and_decompose(
        stacked["test"][:, best_layer].reshape(-1, n_timepoints, n_neurons),
        subset_targets["test"],
        scoring,
        f"best single depth (index {best_layer})",
        REDUCER,
    )
    best_layer_row.update(
        {
            "run_id": "ridge_best_layer",
            "seed": cfg.random_seed,
            "best_layer": int(best_layer),
        }
    )
    rows.append(best_layer_row)
    site_correlations["ridge_best_layer"] = best_layer_site_r
    return rows, site_correlations
# EOF


"""
summarize_over_seeds
Average every run id over the seeds it finished on.

INPUT:
    - rows: list[dict] -> every scored run, seeds mixed

OUTPUT:
    - summary: list[dict] -> one row per run id, ordered by mean test stim_r
"""
def summarize_over_seeds(rows):
    by_run_id = {}
    for row in rows:
        by_run_id.setdefault(row["run_id"], []).append(row)
    # end for scored run
    summary = []
    for run_id, run_rows in by_run_id.items():
        stim_r = np.array([row["mean_stim_r_response"] for row in run_rows])
        summary.append(
            {
                "run_id": run_id,
                "model": run_rows[0]["model"],
                "n_seeds": len(run_rows),
                "mean_stim_r": round(float(stim_r.mean()), 4),
                "std_stim_r": round(float(stim_r.std(ddof=0)), 4),
                "min_stim_r": round(float(stim_r.min()), 4),
                "max_stim_r": round(float(stim_r.max()), 4),
                "mean_fraction_of_ceiling": round(
                    float(
                        np.mean([row["fraction_of_ceiling"] for row in run_rows])
                    ),
                    3,
                ),
                "mean_test_mse": round(
                    float(np.mean([row["test_mse"] for row in run_rows])), 5
                ),
            }
        )
    # end for run id
    return sorted(summary, key=lambda row: -row["mean_stim_r"])
# EOF


"""
plot_stacking_summary
Show every variant's per-seed test stim_r against the joint ridge reference.

INPUT:
    - rows: list[dict] -> every scored run, seeds mixed
    - summary: list[dict] -> per-run-id aggregate, ordered by mean stim_r
    - output_path: Path -> figure destination

OUTPUT:
    - None: writes the figure
"""
def plot_stacking_summary(rows, summary, output_path):
    ordered_ids = [row["run_id"] for row in summary]
    figure, axis = plt.subplots(figsize=(9, 0.5 + 0.55 * len(ordered_ids)))
    for position, run_id in enumerate(ordered_ids):
        seed_values = [
            row["mean_stim_r_response"] for row in rows if row["run_id"] == run_id
        ]
        # Red marks the closed-form references, blue the trained decoders.
        color = "#c0392b" if run_id in REFERENCE_RUN_IDS else "#2c6fbb"
        axis.scatter(
            seed_values,
            np.full(len(seed_values), position),
            color=color,
            alpha=0.55,
            s=28,
        )
        axis.scatter(
            [np.mean(seed_values)], [position], color=color, marker="|", s=420
        )
    # end for plotted run id

    ridge_values = [
        row["mean_stim_r_response"] for row in rows if row["run_id"] == "ridge_joint"
    ]
    if ridge_values:
        axis.axvline(
            float(np.mean(ridge_values)),
            color="#c0392b",
            linestyle="--",
            linewidth=1,
        )
    # end if the joint ridge was scored

    axis.set_yticks(range(len(ordered_ids)))
    axis.set_yticklabels(
        [row["model"] for row in summary], fontsize=8
    )
    axis.invert_yaxis()
    axis.set_xlabel("test stim_r (mean over sites and bins), one point per seed")
    axis.set_title("Decoders stacked on per-depth ridge maps")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
# EOF


def main():
    cfg = parse_args()
    if cfg.smoke_test:
        cfg.epochs = min(cfg.epochs, 2)
        cfg.minimum_epochs = 1
        cfg.patience = 1
        cfg.n_seeds = 1
        cfg.cross_fit_folds = 2
        cfg.noise_ceiling_resamples = 4
        cfg.temporal_noise_levels = "0.25"
    # end if a smoke run was requested

    paths = load_project_paths(cfg)
    device = resolve_device(cfg.device)
    output_dir = Path(
        cfg.output_dir
        or PROJECT_ROOT / "results" / f"tvsd_ridge_stacking_{cfg.model_name}"
    ).expanduser()
    run_dir = output_dir / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # The archive is read once; only the seeded split changes between seeds.
    targets, train_features, test_features, allmat = load_cached_data(cfg, paths)
    run_matrix = build_run_matrix(cfg)

    all_rows, site_correlations, histories = [], {}, {}
    for seed_offset in range(cfg.n_seeds):
        seed_cfg = replace(cfg, random_seed=cfg.random_seed + seed_offset)
        print(f"\n=== seed {seed_cfg.random_seed} ===")
        data = prepare_timebin_data(
            seed_cfg, targets, train_features, test_features, allmat, device
        )
        subset_targets = data["subset_targets"]
        subset_features = data["subset_features"]
        scoring = data["scoring"]
        n_timepoints, n_neurons = subset_targets["train"].shape[1:]

        # --- level one: one ridge per depth, cross-fitted on the fit split ---
        print(
            f"fitting {len(cfg.layer_names)} depth maps, "
            f"{seed_cfg.cross_fit_folds}-fold cross-fitted"
        )
        ridge_start = time.perf_counter()
        stacked, layer_diagnostics = build_ridge_stack(
            subset_features,
            subset_targets,
            seed_cfg.cross_fit_folds,
            seed_cfg.random_seed,
        )
        print(f"  {time.perf_counter() - ridge_start:.0f}s")
        base_predictions = {
            split_name: average_stacked_predictions(
                split_stack, n_timepoints, n_neurons
            )
            for split_name, split_stack in stacked.items()
        }

        # --- the parameter-free references ---
        reference_key = f"references_seed{seed_cfg.random_seed}"
        finished = load_finished_run(run_dir, reference_key)
        if finished is None:
            reference_rows, reference_site_r = score_closed_form_references(
                seed_cfg,
                subset_features,
                subset_targets,
                stacked,
                layer_diagnostics,
                scoring,
            )
            save_finished_run(
                run_dir,
                reference_key,
                {"rows": reference_rows, "layers": layer_diagnostics},
                np.stack(
                    [reference_site_r[name] for name in REFERENCE_RUN_IDS]
                ),
            )
        else:
            reference_rows = finished[0]["rows"]
            reference_site_r = {
                name: finished[1][position]
                for position, name in enumerate(REFERENCE_RUN_IDS)
            }
            print("  references reloaded from a previous run")
        # end if the references had already been scored
        all_rows.extend(reference_rows)
        for name, correlations in reference_site_r.items():
            site_correlations[f"{name}_seed{seed_cfg.random_seed}"] = correlations
        # end for reference

        # --- the joint ridge's own out-of-fold map, for the run that corrects it ---
        base_by_mode = {"replace": None, "residual": base_predictions}
        pending_joint_residual = [
            variant
            for variant in run_matrix
            if variant["target_mode"] == "residual_joint"
            and load_finished_run(
                run_dir, f"{variant['run_id']}_seed{seed_cfg.random_seed}"
            )
            is None
        ]
        if pending_joint_residual:
            print("cross-fitting the joint ridge over all depths at once")
            joint_start = time.perf_counter()
            joint_maps, joint_diagnostics = cross_fit_layer_ridge(
                {
                    split_name: split_features.reshape(len(split_features), -1)
                    for split_name, split_features in subset_features.items()
                },
                subset_targets,
                seed_cfg.cross_fit_folds,
                seed_cfg.random_seed,
            )
            base_by_mode["residual_joint"] = joint_maps
            print(
                f"  alpha {joint_diagnostics['alpha']:.3g} | validation stim_r "
                f"{joint_diagnostics['validation_stim_r']:.4f} | "
                f"{time.perf_counter() - joint_start:.0f}s"
            )
        # end if a run needs the joint ridge's residual

        # --- the trained variants ---
        inputs_by_source = {"stack": stacked, "features": subset_features}
        for variant in run_matrix:
            run_key = f"{variant['run_id']}_seed{seed_cfg.random_seed}"
            finished = load_finished_run(run_dir, run_key)
            if finished is not None:
                row, correlations, history = finished
                print(f"{variant['name']}: reloaded")
            else:
                print(f"{variant['name']}: training")
                row, correlations, history = train_one_variant(
                    variant,
                    seed_cfg,
                    inputs_by_source[variant["source"]],
                    subset_targets,
                    base_by_mode[variant["target_mode"]],
                    scoring,
                    device,
                )
                save_finished_run(run_dir, run_key, row, correlations, history)
                print(
                    f"  test stim_r {row['mean_stim_r_response']:.4f} | "
                    f"epoch {row['best_epoch']:02d} | {row['train_seconds']:.0f}s"
                )
            # end if this run had already finished
            all_rows.append(row)
            site_correlations[run_key] = correlations
            histories[run_key] = history
        # end for trained variant

        del data, stacked, subset_features, subset_targets
        del base_predictions, base_by_mode
        release_device_memory(device)
    # end for seed

    summary = summarize_over_seeds(all_rows)
    with open(output_dir / "config.json", "w") as config_file:
        json.dump(asdict(cfg), config_file, indent=2)
    # end with saved configuration
    with open(output_dir / "results.json", "w") as results_file:
        json.dump(
            {"rows": all_rows, "summary": summary, "histories": histories},
            results_file,
            indent=2,
        )
    # end with saved metrics
    np.savez_compressed(output_dir / "site_stim_r.npz", **site_correlations)
    plot_stacking_summary(
        all_rows, summary, output_dir / "ridge_stacking.png"
    )

    header = (
        f"{'model':<40}{'seeds':>7}{'test r':>9}{'sd':>8}{'range':>16}"
        f"{'frac ceil':>11}{'test MSE':>10}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in summary:
        print(
            f"{row['model']:<40}{row['n_seeds']:>7}{row['mean_stim_r']:>9.4f}"
            f"{row['std_stim_r']:>8.4f}"
            f"{row['min_stim_r']:>8.4f}-{row['max_stim_r']:<7.4f}"
            f"{row['mean_fraction_of_ceiling']:>11.3f}"
            f"{row['mean_test_mse']:>10.5f}"
        )
    # end for summarized run id
    print(
        "\nA variant beats ridge only if its whole seed range clears the joint "
        "ridge row; this split is too small for a single seed to decide."
    )
    print(f"\nsaved to {output_dir}")
# EOF


if __name__ == "__main__":
    main()
# EOC
