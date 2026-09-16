"""
Predict 300 IT sites from 20 other sites, from AlexNet conv5, or from both.

The neural-input control asked what a held-out site is predictable *from*, with
ridge holding the mapping fixed. This flips the direction and frees the mapping:
twenty input sites drawn from a single Utah array predict the remaining three
hundred, under ridge and under the two trained decoders from the timebin search.

Two things make the comparison mean something.

The twenty input sites all come from one array, so the three hundred targets
split into the forty-four that share that array and the two hundred and
fifty-six that do not. Reported separately, that says whether an input site
carries information about distant cortex or only about the tissue under the
same electrodes -- the confound the matched-count array control raised.

Selection never touches single-trial neural input. Feeding a decoder its
neighbours from the target's own trial lets it read shared trial noise, and the
validation presentations of the training images are single trials, so selecting
there would reward exactly that. Instead thirty of the hundred repeated test
images are set aside for early stopping and alpha choice under the same
disjoint-repetition rule used for scoring, and the remaining seventy are scored.
"""

import argparse
import copy
import json
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python_scripts" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "python_scripts" / "scripts"))

from IT_recap.neural_prediction_training import (  # noqa: E402
    aggregate_trials_by_image,
    split_half_reliability,
    stimulus_correlation,
)
from IT_recap.tvsd_experiments import (  # noqa: E402
    N_TEST_IMAGES,
    load_project_paths,
    prepare_search_data,
    resolve_device,
)
from model_classes.timebin_models import build_timebin_model  # noqa: E402
from run_tvsd_neural_input_control import (  # noqa: E402
    REPETITION_COLUMN,
    prepare_ridge,
    ridge_path_from,
    split_test_repetitions,
)


# Sites per Utah array in monkey F's IT block, in physical channel order.
SITES_PER_ARRAY = 64
# Every input group is this wide, so the models' per-group LayerNorm puts the
# AlexNet block and the neural block on the same scale without any rescaling.
GROUP_WIDTH = 100
# Blocks are scaled to unit total variance, so the useful penalties sit low;
# the grid has to reach well below one or the search pins at its own floor.
RIDGE_ALPHAS = np.logspace(-4, 9, 53)

# The configurations these two architectures won the timebin search with.
MODEL_HYPERPARAMETERS = {
    "baseline": {
        "time_embedding_dim": 128, "value_dim": 256, "mlp_hidden_dim": 512,
        "dropout": 0.5, "input_noise_std": 0.1, "temporal_noise_std": 0.1,
    },
    "gru": {
        "hidden_dim": 512, "time_embedding_dim": 16, "dropout": 0.1,
        "input_noise_std": 0.25, "temporal_noise_std": 0.5,
    },
}


@dataclass
class Cfg:
    env: str | None = None
    mua_file_name: str = "f_THINGS_MUA_trials.mat"
    feature_archive_name: str = "tvsd_monkeyF_ijepa_vith14_1k_224_features.npz"
    spatial_stem: str = "tvsd_monkeyF_alexnet_features_11_spatial"
    output_dir: str | None = None

    area: str = "IT"
    target_fs: int = 100
    time_start_ms: float = 0.0
    time_end_ms: float = 200.0
    window_start_ms: float = 80.0
    window_end_ms: float = 180.0
    timebin_ms: float = 20.0
    model_name: str = "ijepa_vith14_1k"
    layer_names: list[str] = field(
        default_factory=lambda: ["encoder.layer.17.output.dense"]
    )
    validation_fraction: float = 0.1
    random_seed: int = 0

    # Input construction.
    n_input_sites: int = 20
    n_alexnet_components: int = 500
    pca_sample: int = 4000
    n_arrays: int = 5

    # The repeated-test images set aside for selection.
    n_selection_images: int = 30

    batch_size: int = 256
    num_workers: int = 0
    epochs: int = 40
    minimum_epochs: int = 8
    patience: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    gradient_clip: float = 1.0
    noise_ceiling_resamples: int = 40
    device: str = "auto"
    models: str = "ridge,baseline,gru"
    smoke_test: bool = False


"""
parse_args
Parse command-line overrides into the experiment configuration.

OUTPUT:
    - cfg: Cfg -> input construction, split, and optimization settings
"""
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for field_name, field_definition in Cfg.__dataclass_fields__.items():
        if field_name == "layer_names":
            continue
        # end if the unused pooled-feature selection is skipped
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
    return Cfg(**vars(parser.parse_args()))
# EOF


"""
load_alexnet_components
Project the cached conv5 map onto its leading training principal components.

The map is 256 x 13 x 13 per image and cannot go into a ridge whole, so it is
reduced once, on a sample of training stimuli, and the projection is reused for
every fold. This keeps only global structure and therefore understates what the
map can do under a spatial readout -- it is a floor for AlexNet, not a verdict.

INPUT:
    - cfg: Cfg -> component count and sample size
    - paths: dict -> active project paths

OUTPUT:
    - train_components: np.ndarray -> [train stimuli, components]
    - test_components: np.ndarray -> [test stimuli, components]
    - explained: float -> fraction of map variance retained
"""
def load_alexnet_components(cfg, paths):
    cache_dir = Path(paths["data_path"]) / "models"
    train_map = np.load(cache_dir / f"{cfg.spatial_stem}_train.npy", mmap_mode="r")
    test_map = np.load(cache_dir / f"{cfg.spatial_stem}_test.npy", mmap_mode="r")
    sample = np.sort(
        np.random.default_rng(cfg.random_seed).permutation(len(train_map))[
            : cfg.pca_sample
        ]
    )
    pca = PCA(
        n_components=cfg.n_alexnet_components,
        svd_solver="randomized",
        random_state=cfg.random_seed,
    )
    pca.fit(np.asarray(train_map[sample], dtype=np.float32).reshape(len(sample), -1))
    train_components = np.empty(
        (len(train_map), cfg.n_alexnet_components), np.float32
    )
    for start in range(0, len(train_map), 2000):
        chunk = np.asarray(train_map[start : start + 2000], dtype=np.float32)
        train_components[start : start + 2000] = pca.transform(
            chunk.reshape(len(chunk), -1)
        )
    # end for stimulus chunk
    test_components = pca.transform(
        np.asarray(test_map, dtype=np.float32).reshape(len(test_map), -1)
    ).astype(np.float32)
    return train_components, test_components, float(pca.explained_variance_ratio_.sum())
# EOF


"""
unit_variance
Scale a block so its training total variance is one.

Concatenated blocks share one ridge penalty, so a block with a larger norm is
effectively penalized less. The trained decoders LayerNorm each group and do not
need this, but the ridge does, and applying it to both keeps the comparison fair.

INPUT:
    - block: dict -> subset name to [samples, dimensions] array

OUTPUT:
    - scaled: dict -> the same blocks divided by the training scale
"""
def unit_variance(block):
    scale = np.sqrt(block["train"].var(0).sum())
    return {name: (value / scale).astype(np.float32) for name, value in block.items()}
# EOF


"""
as_groups
Reshape a flat block into equal-width groups for the cached-feature decoders.

INPUT:
    - block: dict -> subset name to [samples, dimensions] array
    - width: int -> width of every group

OUTPUT:
    - grouped: dict -> subset name to [samples, groups, width] array
"""
def as_groups(block, width=GROUP_WIDTH):
    return {
        name: value.reshape(len(value), -1, width) for name, value in block.items()
    }
# EOF


"""
score_stim_r
Correlate predictions with the target half across the scored images.

INPUT:
    - predictions: np.ndarray -> [images, time, sites]
    - targets: np.ndarray -> [images, time, sites]

OUTPUT:
    - correlations: np.ndarray -> [time, sites] stim_r
"""
def score_stim_r(predictions, targets):
    return stimulus_correlation(predictions, targets)
# EOF


"""
train_decoder
Optimize one cached-feature decoder, selecting on disjoint-repetition stim_r.

INPUT:
    - cfg: Cfg -> optimization settings
    - architecture: str -> key in TIMEBIN_MODEL_CLASSES
    - grouped: dict -> subset name to [samples, groups, width] inputs
    - train_targets: np.ndarray -> [presentations, time, sites]
    - selection: tuple -> (inputs, targets) of the selection images
    - device: torch.device -> compute device

OUTPUT:
    - model: nn.Module -> the restored best checkpoint
    - best_epoch: int -> epoch it came from
    - best_score: float -> selection stim_r at that epoch
"""
def train_decoder(cfg, architecture, grouped, train_targets, selection, device):
    n_time, n_sites = train_targets.shape[1], train_targets.shape[2]
    torch.manual_seed(cfg.random_seed)
    model = build_timebin_model(
        architecture,
        n_layers=grouped["train"].shape[1],
        feature_dim=grouped["train"].shape[2],
        n_timepoints=n_time,
        n_neurons=n_sites,
        **MODEL_HYPERPARAMETERS[architecture],
    ).to(device)

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(grouped["train"]), torch.from_numpy(train_targets)
        ),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        generator=torch.Generator().manual_seed(cfg.random_seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(2, cfg.patience // 3)
    )
    selection_inputs = torch.from_numpy(selection[0]).to(device)
    selection_targets = selection[1]

    best_state = copy.deepcopy(model.state_dict())
    best_score, best_epoch, stalled = -np.inf, 0, 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for features, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            predictions, _ = model(features.to(device))
            loss = torch.nn.functional.mse_loss(predictions, targets.to(device))
            loss.backward()
            if cfg.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            # end if gradient clipping is enabled
            optimizer.step()
        # end for training batch

        model.eval()
        with torch.no_grad():
            predictions = model(selection_inputs)[0].cpu().numpy()
        # end with no gradient tracking
        score = float(np.nanmean(score_stim_r(predictions, selection_targets)))
        scheduler.step(-score)
        if score > best_score + 1e-8:
            best_score, best_epoch, stalled = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stalled += 1
        # end if this epoch improved the selection score
        if epoch >= cfg.minimum_epochs and stalled >= cfg.patience:
            break
        # end if patience is exhausted
    # end for optimization epoch
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch, best_score
# EOF


def main():
    cfg = parse_args()
    if cfg.smoke_test:
        cfg.epochs, cfg.minimum_epochs, cfg.patience = 2, 1, 1
        cfg.n_arrays, cfg.noise_ceiling_resamples, cfg.pca_sample = 1, 4, 500
    # end if a short run was requested

    paths = load_project_paths(cfg)
    device = resolve_device(cfg.device)
    output_dir = Path(
        cfg.output_dir or PROJECT_ROOT / "results" / "tvsd_population_readout"
    ).expanduser()
    (output_dir / "runs").mkdir(parents=True, exist_ok=True)

    data = prepare_search_data(cfg, paths, device)
    subset_targets, indices, allmat = (
        data["subset_targets"], data["indices"], data["allmat"]
    )
    n_sites = data["shapes"]["n_neurons"]
    image_ids = data["scoring"]["test_image_ids"]

    # --- disjoint repetition halves, then a disjoint image split ---
    input_half, target_half = split_test_repetitions(
        allmat[indices["test"], REPETITION_COLUMN], image_ids, cfg.random_seed
    )
    input_average = aggregate_trials_by_image(
        subset_targets["test"][input_half], image_ids[input_half], N_TEST_IMAGES
    )
    target_average = aggregate_trials_by_image(
        subset_targets["test"][target_half], image_ids[target_half], N_TEST_IMAGES
    )
    ceiling = split_half_reliability(
        subset_targets["test"][target_half],
        image_ids[target_half],
        N_TEST_IMAGES,
        "mean",
        cfg.noise_ceiling_resamples,
        cfg.random_seed,
    )
    shuffled_images = np.random.default_rng(cfg.random_seed).permutation(N_TEST_IMAGES)
    selection_images = shuffled_images[: cfg.n_selection_images]
    scored_images = shuffled_images[cfg.n_selection_images :]
    print(
        f"{int(input_half.sum())} input / {int(target_half.sum())} target "
        f"presentations | {len(selection_images)} selection / "
        f"{len(scored_images)} scored images"
    )

    alexnet_train, alexnet_test, explained = load_alexnet_components(cfg, paths)
    print(
        f"AlexNet PCA{cfg.n_alexnet_components} keeps {explained:.3f} of the "
        f"map variance", flush=True
    )
    stimulus_rows = {
        "train": allmat[indices["train"], 1] - 1,
        "test": allmat[indices["test"], 2] - 1,
    }
    alexnet_block = unit_variance(
        {
            "train": alexnet_train[stimulus_rows["train"]],
            "selection": alexnet_test[selection_images],
            "scored": alexnet_test[scored_images],
        }
    )

    array_of = np.arange(n_sites) // SITES_PER_ARRAY
    generator = np.random.default_rng(cfg.random_seed)
    flatten = lambda array: array.reshape(len(array), -1)
    records = []

    for array_index in range(cfg.n_arrays):
        members = np.flatnonzero(array_of == array_index)
        input_sites = np.sort(generator.permutation(members)[: cfg.n_input_sites])
        target_sites = np.setdiff1d(np.arange(n_sites), input_sites)
        same_array = np.isin(target_sites, members)
        print(
            f"\narray {array_index}: {len(input_sites)} input sites -> "
            f"{len(target_sites)} targets "
            f"({int(same_array.sum())} same array, {int((~same_array).sum())} other)"
        )

        neural_block = unit_variance(
            {
                # Training presentations carry the input sites' own single trial.
                "train": flatten(subset_targets["train"][:, :, input_sites]),
                # Evaluation reads them from the disjoint repetition half.
                "selection": flatten(input_average[selection_images][:, :, input_sites]),
                "scored": flatten(input_average[scored_images][:, :, input_sites]),
            }
        )
        inputs = {
            "alexnet": alexnet_block,
            "sites20": neural_block,
            "combined": {
                name: np.hstack([alexnet_block[name], neural_block[name]])
                for name in alexnet_block
            },
        }
        train_targets = subset_targets["train"][:, :, target_sites]
        selection_targets = target_average[selection_images][:, :, target_sites]
        scored_targets = target_average[scored_images][:, :, target_sites]

        requested_models = [m.strip() for m in cfg.models.split(",") if m.strip()]
        for input_name, block in inputs.items():
            if "ridge" not in requested_models:
                pass
            # end if ridge was not requested
            # --- ridge, alpha chosen on the selection images ---
            prepared = prepare_ridge(block["train"])
            flat_train = flatten(train_targets)
            selection_path = ridge_path_from(
                prepared, flat_train, block["selection"], RIDGE_ALPHAS
            )
            scores = [
                np.nanmean(
                    score_stim_r(
                        path.reshape(len(selection_targets), *selection_targets.shape[1:]),
                        selection_targets,
                    )
                )
                for path in selection_path
            ]
            best_alpha = float(RIDGE_ALPHAS[int(np.argmax(scores))])
            predictions = ridge_path_from(
                prepared, flat_train, block["scored"], [best_alpha]
            )[0].reshape(len(scored_targets), *scored_targets.shape[1:])
            records.append(
                {
                    "array": array_index, "input": input_name, "model": "ridge",
                    "alpha": best_alpha,
                    "alpha_at_grid_edge": bool(
                        best_alpha in (RIDGE_ALPHAS[0], RIDGE_ALPHAS[-1])
                    ),
                    **summarize(score_stim_r(predictions, scored_targets),
                                ceiling[:, target_sites], same_array),
                }
            )
            print(f"  ridge    {input_name:9s} {records[-1]['stim_r']:.4f}", flush=True)

            # --- the two trained decoders on the same input ---
            grouped = as_groups(block)
            for architecture in [
                name for name in MODEL_HYPERPARAMETERS if name in requested_models
            ]:
                started = time.perf_counter()
                model, best_epoch, best_score = train_decoder(
                    cfg, architecture, grouped, train_targets,
                    (grouped["selection"], selection_targets), device,
                )
                with torch.no_grad():
                    predictions = model(
                        torch.from_numpy(grouped["scored"]).to(device)
                    )[0].cpu().numpy()
                # end with no gradient tracking
                records.append(
                    {
                        "array": array_index, "input": input_name,
                        "model": architecture, "best_epoch": best_epoch,
                        "selection_stim_r": round(best_score, 4),
                        "train_seconds": round(time.perf_counter() - started, 1),
                        **summarize(score_stim_r(predictions, scored_targets),
                                    ceiling[:, target_sites], same_array),
                    }
                )
                print(
                    f"  {architecture:8s} {input_name:9s} "
                    f"{records[-1]['stim_r']:.4f} "
                    f"({records[-1]['train_seconds']:.0f}s)", flush=True
                )
                del model
                if device.type == "mps":
                    torch.mps.empty_cache()
                # end if the device caches its allocations
            # end for architecture
        # end for input space
        with open(output_dir / "results.json", "w") as results_file:
            json.dump({"config": asdict(cfg), "rows": records}, results_file, indent=2)
        # end with saved metrics
    # end for input array

    print(f"\n{'model':<10}{'input':<11}{'stim_r':>9}{'% ceil':>9}"
          f"{'same-array':>12}{'other-array':>13}")
    print("-" * 64)
    for model_name in ("ridge", "baseline", "gru"):
        for input_name in ("alexnet", "sites20", "combined"):
            chosen = [
                row for row in records
                if row["model"] == model_name and row["input"] == input_name
            ]
            if not chosen:
                continue
            # end if this cell was not run
            mean = lambda key: float(np.mean([row[key] for row in chosen]))
            print(
                f"{model_name:<10}{input_name:<11}{mean('stim_r'):>9.4f}"
                f"{mean('fraction_of_ceiling') * 100:>8.1f}%"
                f"{mean('stim_r_same_array'):>12.4f}"
                f"{mean('stim_r_other_array'):>13.4f}"
            )
        # end for input space
    # end for model
    print(f"\nsaved to {output_dir}")
# EOF


"""
summarize
Reduce one run's per-cell stim_r to the reported numbers.

INPUT:
    - correlations: np.ndarray -> [time, sites] stim_r
    - ceiling: np.ndarray -> [time, sites] noise ceiling of those sites
    - same_array: np.ndarray -> boolean mask of targets sharing the input array

OUTPUT:
    - summary: dict -> overall and per-array-membership means
"""
def summarize(correlations, ceiling, same_array):
    return {
        "stim_r": round(float(np.nanmean(correlations)), 4),
        "fraction_of_ceiling": round(
            float(np.nanmean(correlations) / np.nanmean(ceiling)), 4
        ),
        "stim_r_same_array": round(float(np.nanmean(correlations[:, same_array])), 4),
        "stim_r_other_array": round(float(np.nanmean(correlations[:, ~same_array])), 4),
        "ceiling": round(float(np.nanmean(ceiling)), 4),
    }
# EOF


if __name__ == "__main__":
    main()
