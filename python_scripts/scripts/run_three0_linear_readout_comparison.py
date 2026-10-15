"""
Compare three decoders of the trial-averaged three0 AIT response, all fed the
same mean-pooled DINOv3 layers:

    ridge           RidgeCV on the concatenated layers -> flattened [time x unit]
                    output; closed form, one alpha shared by every output
    timebin_linear  the same parameterization (one [D x units] map per time bin,
                    identical coefficient count) fitted by SGD on the squared
                    Frobenius norm of the population residual at each bin, with
                    an explicit L2 term. With lambda = alpha / n_fit its full-batch
                    optimum IS ridge at that alpha, so any gap is optimization
    baseline        the notebook's BaselineModel (feature attention)

The linear model is swept over its ridge-equivalent alpha (including the alpha
RidgeCV selected), and the best sweep entry is picked on validation R2. The
whole comparison is repeated over several 60/20/20 image splits. Test scores are
reported per time bin as R2 (with the trial-mean noise ceiling), stim_r, and
matched-time RSA. Every (split, model) run is written to runs/ before the next
starts, so rerunning the same command resumes after a kill.
"""

import argparse
import copy
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from sklearn.linear_model import Ridge, RidgeCV  # noqa: E402


ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python_scripts" / "src"))

with open(PROJECT_ROOT / "config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)
# end with project configuration

from IT_recap.dynamic_drsa import prepared_rdm_time_series  # noqa: E402
from IT_recap.neural_prediction_training import stimulus_correlation  # noqa: E402
from IT_recap.three0_decoding import (  # noqa: E402
    THREE0_STIMULI,
    load_three0_targets,
    load_three0_trial_noise,
    make_image_splits,
)
from IT_recap.tvsd_experiments import resolve_device  # noqa: E402
from model_classes.temporal_experiment_models import (  # noqa: E402
    TimebinLinearRegression,
)
from model_classes.temporal_models import BaselineModel  # noqa: E402


# Colorblind-validated slots shared with the TVSD comparison scripts.
MODEL_COLORS = {
    "ridge": "#2a78d6",
    "baseline": "#eb6834",
    "timebin_linear": "#1baf7a",
}
MODEL_LABELS = {
    "ridge": "RidgeCV (closed form)",
    "baseline": "BaselineModel",
    "timebin_linear": "Per-bin linear, SGD + Frobenius loss",
}


@dataclass
class Cfg:
    env: str | None = None
    output_dir: str | None = None

    # Neural target: trial-averaged natraster, as in the baseline notebooks.
    stimuli_folder: str = THREE0_STIMULI
    brain_area: str = "AIT"
    target_fs: int = 100
    time_start_ms: float = 0.0
    time_end_ms: float = 300.0

    # Frozen DINOv3 features, mean-pooled and cached.
    model_name: str = "dino_v3_l"
    model_source: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    img_size: int = 224
    pooling: str = "mean"
    layer_names: str = (
        "layer.3.mlp.down_proj,layer.13.mlp.down_proj,"
        "layer.16.mlp.down_proj,layer.20.mlp.down_proj"
    )

    # Splits: every seed is an independent 60/20/20 image split.
    seeds: str = "0,1,2,3,4"
    validation_fraction: float = 0.2
    test_fraction: float = 0.2

    # Ridge: z-scored features and a wide grid (the narrow one pins at its edge).
    ridge_alpha_min_exp: float = 0.0
    ridge_alpha_max_exp: float = 9.0
    ridge_n_alphas: int = 19

    # SGD linear sweep over ridge-equivalent alphas; "ridge" = RidgeCV's choice.
    linear_alphas: str = "0,1e2,1e3,1e4,1e5,1e6,ridge"
    # Plain SGD + momentum; lr = scale / top eigenvalue of X'X/N on the fit
    # rows, so the step adapts to each split's curvature (Adam's sign-like
    # steps overfit this 4096-d design within one epoch).
    linear_lr_scale: float = 0.05
    linear_momentum: float = 0.9

    # BaselineModel, as configured in noise_layer_baseline_model_dev.ipynb.
    temporal_embedding_dim: int = 256
    value_dim: int = 128
    n_attention_heads: int = 2
    temporal_compression_ratio: int = 3
    mlp_hidden_dim: int = 32
    dropout: float = 0.3
    attention_granularity: str = "feature"
    baseline_learning_rate: float = 1e-3
    baseline_weight_decay: float = 1e-4

    # Shared optimization; checkpoints are picked on validation R2.
    batch_size: int = 64
    epochs: int = 100
    patience: int = 100

    # Dynamic RSA settings of the notebooks.
    drsa_rdm_metric: str = "euclidean"
    drsa_rsa_metric: str = "spearman"

    device: str = "auto"
    smoke_test: bool = False


"""
parse_args
Parse command-line overrides into the experiment configuration.

OUTPUT:
    - cfg: Cfg -> data, split, sweep, and optimization settings
"""
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for field_name, field in Cfg.__dataclass_fields__.items():
        default = field.default
        if isinstance(default, bool):
            parser.add_argument(
                f"--{field_name}",
                action=argparse.BooleanOptionalAction,
                default=default,
            )
        elif default is None:
            parser.add_argument(f"--{field_name}", default=default)
        else:
            parser.add_argument(f"--{field_name}", type=type(default), default=default)
        # end if boolean, optional, or typed argument
    # end for configuration field
    return Cfg(**vars(parser.parse_args()))
# EOF


"""
r2_by_timebin
Coefficient of determination at every time bin, pooled over units, against the
fit-split mean (the predict-the-mean baseline scores exactly zero).

INPUT:
    - predictions: np.ndarray -> [images, time, units]
    - targets: np.ndarray -> [images, time, units]
    - fit_mean: np.ndarray -> [time, units] mean of the fitting rows

OUTPUT:
    - r2: np.ndarray -> [time] R2 per bin
"""
def r2_by_timebin(predictions, targets, fit_mean):
    residual = ((predictions - targets) ** 2).sum(axis=(0, 2))
    total = ((targets - fit_mean) ** 2).sum(axis=(0, 2))
    return 1.0 - residual / total
# EOF


"""
overall_r2
R2 over every image, bin, and unit at once (the number the three0 notes quote).

INPUT:
    - predictions, targets: np.ndarray -> [images, time, units]
    - fit_mean: np.ndarray -> [time, units] mean of the fitting rows

OUTPUT:
    - r2: float -> pooled R2
"""
def overall_r2(predictions, targets, fit_mean):
    residual = ((predictions - targets) ** 2).sum()
    total = ((targets - fit_mean) ** 2).sum()
    return float(1.0 - residual / total)
# EOF


"""
train_decoder
Minibatch-train one decoder on in-memory tensors and keep the epoch with the
best validation R2. Train and validation R2 are logged every epoch so both
families share one training-curve scale despite their different losses.

INPUT:
    - model: nn.Module -> decoder, already on the device
    - predict: callable -> features tensor -> [B, time, units] predictions
    - loss_fn: callable -> (predictions, targets, model) -> scalar loss
    - optimizer: torch.optim.Optimizer -> optimizer over trainable parameters
    - data: dict -> "train"/"validation" (features, targets) tensors on device
    - fit_mean: np.ndarray -> [time, units] fit-split mean for R2
    - cfg: Cfg -> epochs, patience, batch size, random seed source
    - seed: int -> shuffling seed

OUTPUT:
    - history: dict -> per-epoch train_loss, train_r2, validation_r2
    - best_epoch: int -> selected epoch (0 = initialization)
"""
def train_decoder(model, predict, loss_fn, optimizer, data, fit_mean, cfg, seed):
    train_x, train_y = data["train"]
    generator = torch.Generator().manual_seed(seed)

    def evaluate(split):
        model.eval()
        with torch.no_grad():
            predictions = predict(data[split][0]).cpu().numpy()
        # end with no gradient tracking
        return overall_r2(predictions, data[split][1].cpu().numpy(), fit_mean)
    # EOF

    history = {"train_loss": [], "train_r2": [], "validation_r2": []}
    best_r2, best_epoch = evaluate("validation"), 0
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        epoch_loss = 0.0
        for start in range(0, len(order), cfg.batch_size):
            batch = order[start : start + cfg.batch_size].to(train_x.device)
            loss = loss_fn(predict(train_x[batch]), train_y[batch], model)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        # end for minibatch
        history["train_loss"].append(epoch_loss / len(order))
        history["train_r2"].append(evaluate("train"))
        history["validation_r2"].append(evaluate("validation"))

        if history["validation_r2"][-1] > best_r2:
            best_r2, best_epoch = history["validation_r2"][-1], epoch
            best_state = copy.deepcopy(model.state_dict())
        # end if this epoch is the best checkpoint
        if epoch - best_epoch >= cfg.patience:
            break
        # end if validation R2 stopped improving
    # end for epoch
    model.load_state_dict(best_state)
    return history, best_epoch
# EOF


"""
run_linear_sgd
Fit the per-bin linear map by SGD. The loss is, per image, the squared
Frobenius norm of the [time, units] residual, plus lambda ||W||^2 with
lambda = alpha / n_fit, so that n_fit * loss is RidgeCV's objective exactly
(intercept unpenalized, as in sklearn).

INPUT:
    - alpha: float -> ridge-equivalent penalty
    - data, fit_mean, cfg, seed, device -> as in train_decoder
    - shapes: tuple -> (n_layers, feature_dim, n_time, n_units)

OUTPUT:
    - run: dict -> validation/test predictions, history, epoch, parameter count
"""
def run_linear_sgd(alpha, data, fit_mean, cfg, seed, device, shapes):
    torch.manual_seed(seed)
    model = TimebinLinearRegression(*shapes).to(device)
    l2_lambda = alpha / len(data["train"][0])

    def loss_fn(predictions, targets, net):
        # Sum over (time, unit) = squared Frobenius norm per image; mean over batch.
        frobenius = (predictions - targets).square().sum(dim=(1, 2)).mean()
        return frobenius + l2_lambda * net.weight.square().sum()
    # EOF

    predict = lambda features: model(features)[0]
    # Top curvature of the per-bin data term: largest eigenvalue of X'X/N
    # (taken from the N x N Gram matrix, which shares its nonzero spectrum).
    train_x = data["train"][0].flatten(1).cpu().double()
    top_eigenvalue = torch.linalg.eigvalsh(train_x @ train_x.T / len(train_x)).max()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.linear_lr_scale / float(top_eigenvalue),
        momentum=cfg.linear_momentum,
    )
    history, best_epoch = train_decoder(
        model, predict, loss_fn, optimizer, data, fit_mean, cfg, seed
    )
    return finish_run(model, predict, data, history, best_epoch)
# EOF


"""
run_baseline
Fit the notebook's BaselineModel on raw cached layer vectors with its usual
time-bin MSE loss and AdamW weight decay.

INPUT:
    - encoder: imgANN -> frozen DINOv3 wrapper the model is built around
    - data, fit_mean, cfg, seed, device -> as in train_decoder
    - shapes: tuple -> (n_layers, feature_dim, n_time, n_units)

OUTPUT:
    - run: dict -> validation/test predictions, history, epoch, parameter count
"""
def run_baseline(encoder, data, fit_mean, cfg, seed, device, shapes):
    torch.manual_seed(seed)
    model = BaselineModel(
        encoder,
        layers=cfg.layer_names.split(","),
        temporal_embedding_dim=cfg.temporal_embedding_dim,
        value_dim=cfg.value_dim,
        n_timepoints=shapes[2],
        temporal_compression_ratio=cfg.temporal_compression_ratio,
        n_neurons=shapes[3],
        mlp_hidden_dim=cfg.mlp_hidden_dim,
        dropout=cfg.dropout,
        attention_granularity=cfg.attention_granularity,
        n_attention_heads=cfg.n_attention_heads,
    ).to(device)
    loss_fn = lambda predictions, targets, _: (predictions - targets).square().mean()
    predict = lambda features: model(features, use_precomputed_features=True)[0]
    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(),
        lr=cfg.baseline_learning_rate,
        weight_decay=cfg.baseline_weight_decay,
    )
    history, best_epoch = train_decoder(
        model, predict, loss_fn, optimizer, data, fit_mean, cfg, seed
    )
    return finish_run(model, predict, data, history, best_epoch)
# EOF


"""
finish_run
Collect the restored checkpoint's validation and test predictions.

OUTPUT:
    - run: dict -> arrays and scalars written to the run file
"""
def finish_run(model, predict, data, history, best_epoch):
    model.eval()
    with torch.no_grad():
        validation_predictions = predict(data["validation"][0]).cpu().numpy()
        test_predictions = predict(data["test"][0]).cpu().numpy()
    # end with no gradient tracking
    trainable = [p for p in model.parameters() if p.requires_grad]
    return {
        "validation_predictions": validation_predictions,
        "test_predictions": test_predictions,
        "train_r2_curve": np.asarray(history["train_r2"]),
        "validation_r2_curve": np.asarray(history["validation_r2"]),
        "train_loss_curve": np.asarray(history["train_loss"]),
        "best_epoch": best_epoch,
        "n_parameters": int(sum(p.numel() for p in trainable)),
    }
# EOF


"""
run_split
Fit ridge, the SGD-linear sweep, and the BaselineModel on one image split,
writing every finished run before starting the next.

INPUT:
    - cfg: Cfg -> experiment settings
    - seed: int -> split seed
    - targets: np.ndarray -> [images, time, units] trial-averaged responses
    - ann_index: np.ndarray -> ANN index of every target row
    - features: np.ndarray -> [images, layers, embedding] aligned DINOv3 vectors
    - encoder: imgANN -> frozen encoder wrapper for BaselineModel
    - device: torch.device -> compute device
    - run_dir: Path -> per-run output directory
"""
def run_split(cfg, seed, targets, ann_index, features, encoder, device, run_dir):
    splits = make_image_splits(
        ann_index, seed, cfg.validation_fraction, cfg.test_fraction
    )
    n_images, n_layers, feature_dim = features.shape
    n_time, n_units = targets.shape[1:]
    shapes = (n_layers, feature_dim, n_time, n_units)

    # Center targets on the fit split, per (time, unit), as the notebooks do.
    fit_mean = targets[splits["train"]].mean(0)
    centered = targets - fit_mean
    zero_mean = np.zeros_like(fit_mean)

    # Linear models see the fit-split z-scored flat design; BaselineModel sees
    # the raw layer vectors and normalizes them internally.
    flat = features.reshape(n_images, -1)
    feature_mean = flat[splits["train"]].mean(0)
    feature_scale = flat[splits["train"]].std(0) + 1e-6
    standardized = ((flat - feature_mean) / feature_scale).reshape(features.shape)

    def tensors(design):
        return {
            split: (
                torch.as_tensor(design[rows], device=device),
                torch.as_tensor(centered[rows], device=device),
            )
            for split, rows in splits.items()
        }
    # EOF

    # --- ridge: closed form, LOO-selected alpha on the fit rows ---
    ridge_path = run_dir / f"seed{seed}_ridge.npz"
    if not ridge_path.is_file():
        flat_std = standardized.reshape(n_images, -1)
        alphas = np.logspace(
            cfg.ridge_alpha_min_exp, cfg.ridge_alpha_max_exp, cfg.ridge_n_alphas
        )
        ridge = RidgeCV(alphas=alphas).fit(
            flat_std[splits["train"]],
            centered[splits["train"]].reshape(len(splits["train"]), -1),
        )
        # Closed-form ridge at each swept alpha, for the SGD-vs-closed-form panel.
        sweep_alphas = [
            float(a) for a in cfg.linear_alphas.split(",") if a != "ridge"
        ]
        closed_form_test_r2 = []
        for alpha in sweep_alphas:
            fixed = Ridge(alpha=max(alpha, 1e-8)).fit(
                flat_std[splits["train"]],
                centered[splits["train"]].reshape(len(splits["train"]), -1),
            )
            closed_form_test_r2.append(
                overall_r2(
                    fixed.predict(flat_std[splits["test"]]).reshape(-1, n_time, n_units),
                    centered[splits["test"]],
                    zero_mean,
                )
            )
        # end for swept alpha
        np.savez_compressed(
            ridge_path,
            validation_predictions=ridge.predict(flat_std[splits["validation"]]).reshape(
                -1, n_time, n_units
            ),
            test_predictions=ridge.predict(flat_std[splits["test"]]).reshape(
                -1, n_time, n_units
            ),
            alpha=float(ridge.alpha_),
            n_parameters=int(ridge.coef_.size + np.size(ridge.intercept_)),
            sweep_alphas=np.asarray(sweep_alphas),
            closed_form_test_r2=np.asarray(closed_form_test_r2),
        )
        print(f"  seed {seed} | ridge alpha {ridge.alpha_:g}")
    # end if ridge is not on disk
    ridge_alpha = float(np.load(ridge_path)["alpha"])

    # --- SGD linear sweep over ridge-equivalent alphas ---
    linear_data = tensors(standardized.astype(np.float32))
    for alpha_text in cfg.linear_alphas.split(","):
        alpha = ridge_alpha if alpha_text == "ridge" else float(alpha_text)
        run_path = run_dir / f"seed{seed}_timebin_linear_{alpha_text}.npz"
        if run_path.is_file():
            continue
        # end if this sweep entry already finished
        run = run_linear_sgd(alpha, linear_data, zero_mean, cfg, seed, device, shapes)
        np.savez_compressed(run_path, alpha=alpha, **run)
        print(
            f"  seed {seed} | linear alpha {alpha_text} ({alpha:g}) | best epoch "
            f"{run['best_epoch']} | val R2 {run['validation_r2_curve'].max():.4f}"
        )
        if device.type == "mps":
            torch.mps.empty_cache()
        # end if MPS memory should be released between runs
    # end for sweep entry
    del linear_data

    # --- BaselineModel ---
    baseline_path = run_dir / f"seed{seed}_baseline.npz"
    if not baseline_path.is_file():
        run = run_baseline(
            encoder, tensors(features.astype(np.float32)), zero_mean, cfg, seed,
            device, shapes,
        )
        np.savez_compressed(baseline_path, **run)
        print(
            f"  seed {seed} | baseline | best epoch {run['best_epoch']} | "
            f"val R2 {run['validation_r2_curve'].max():.4f}"
        )
    # end if the baseline is not on disk
# EOF


"""
score_split
Score ridge, the validation-selected SGD-linear entry, the SGD-linear entry at
ridge's own alpha, and BaselineModel on one split's test images.

OUTPUT:
    - scores: dict -> per-model per-bin R2, stim_r, RSA, and pooled scalars
    - extras: dict -> ceiling per bin, sweep table, curves
"""
def score_split(cfg, seed, targets, ann_index, noise_variance, run_dir):
    splits = make_image_splits(
        ann_index, seed, cfg.validation_fraction, cfg.test_fraction
    )
    fit_mean = targets[splits["train"]].mean(0)
    test_targets = targets[splits["test"]] - fit_mean
    validation_targets = targets[splits["validation"]] - fit_mean
    zero_mean = np.zeros_like(fit_mean)

    # Noise ceiling per bin: the trial-mean variance no model can explain.
    total_by_bin = (test_targets**2).mean(axis=(0, 2))
    noise_by_bin = noise_variance[splits["test"]].mean(axis=(0, 2))
    ceiling_by_bin = 1.0 - noise_by_bin / total_by_bin
    ceiling_overall = 1.0 - noise_variance[splits["test"]].mean() / (test_targets**2).mean()

    neural_rdms = prepared_rdm_time_series(
        test_targets, cfg.drsa_rdm_metric, cfg.drsa_rsa_metric
    )

    ridge_run = np.load(run_dir / f"seed{seed}_ridge.npz")
    linear_runs = {
        alpha_text: np.load(run_dir / f"seed{seed}_timebin_linear_{alpha_text}.npz")
        for alpha_text in cfg.linear_alphas.split(",")
    }
    sweep = {
        alpha_text: {
            "alpha": float(run["alpha"]),
            "validation_r2": overall_r2(
                run["validation_predictions"], validation_targets, zero_mean
            ),
            "test_r2": overall_r2(run["test_predictions"], test_targets, zero_mean),
            "best_epoch": int(run["best_epoch"]),
        }
        for alpha_text, run in linear_runs.items()
    }
    selected_alpha = max(sweep, key=lambda key: sweep[key]["validation_r2"])

    runs = {
        "ridge": ridge_run,
        "timebin_linear": linear_runs[selected_alpha],
        "timebin_linear_at_ridge_alpha": linear_runs.get("ridge"),
        "baseline": np.load(run_dir / f"seed{seed}_baseline.npz"),
    }
    scores = {}
    for name, run in runs.items():
        if run is None:
            continue
        # end if the ridge-alpha entry was not part of the sweep
        predictions = run["test_predictions"]
        prediction_rdms = prepared_rdm_time_series(
            predictions, cfg.drsa_rdm_metric, cfg.drsa_rsa_metric
        )
        r2 = overall_r2(predictions, test_targets, zero_mean)
        scores[name] = {
            "r2_by_bin": r2_by_timebin(predictions, test_targets, zero_mean),
            "stim_r_by_bin": np.nanmean(
                stimulus_correlation(predictions, test_targets), axis=1
            ),
            # Matched-time RSA = diagonal of the dRSA matrix of the notebooks.
            "rsa_by_bin": (neural_rdms * prediction_rdms).sum(axis=1),
            "r2": r2,
            "fraction_of_ceiling": r2 / ceiling_overall,
            "validation_r2": overall_r2(
                run["validation_predictions"], validation_targets, zero_mean
            ),
            "n_parameters": int(run["n_parameters"]),
            "train_r2_curve": run["train_r2_curve"] if "train_r2_curve" in run else None,
            "validation_r2_curve": (
                run["validation_r2_curve"] if "validation_r2_curve" in run else None
            ),
        }
    # end for scored model
    extras = {
        "ceiling_by_bin": ceiling_by_bin,
        "ceiling_overall": ceiling_overall,
        "sweep": sweep,
        "selected_alpha": selected_alpha,
        "ridge_alpha": float(ridge_run["alpha"]),
        "closed_form_alphas": ridge_run["sweep_alphas"],
        "closed_form_test_r2": ridge_run["closed_form_test_r2"],
    }
    return scores, extras
# EOF


"""
pad_curves
Stack early-stopped curves of different lengths, padding with NaN.
"""
def pad_curves(curves):
    length = max(len(curve) for curve in curves)
    padded = np.full((len(curves), length), np.nan)
    for row, curve in enumerate(curves):
        padded[row, : len(curve)] = curve
    # end for curve
    return padded
# EOF


"""
plot_comparison
Training curves, per-bin test R2 / stim_r / RSA, the alpha sweep, and the
pooled test R2 per split.

INPUT:
    - cfg: Cfg -> time axis settings
    - all_scores: list[dict] -> score_split output per seed
    - all_extras: list[dict] -> score_split extras per seed
    - output_path: Path -> figure destination
"""
def plot_comparison(cfg, all_scores, all_extras, output_path):
    models = ("ridge", "timebin_linear", "baseline")
    n_time = len(all_extras[0]["ceiling_by_bin"])
    time_ms = cfg.time_start_ms + np.arange(n_time) * 1000.0 / cfg.target_fs
    figure, axes = plt.subplots(2, 3, figsize=(19, 10.5))
    axes = axes.ravel()

    # Panel 1: R2 learning curves, mean over splits (NaN after early stopping).
    for name in ("timebin_linear", "baseline"):
        for key, style in (("validation_r2_curve", "-"), ("train_r2_curve", "--")):
            curves = pad_curves([scores[name][key] for scores in all_scores])
            axes[0].plot(
                np.arange(1, curves.shape[1] + 1),
                np.nanmean(curves, axis=0),
                linestyle=style,
                linewidth=2,
                color=MODEL_COLORS[name],
                label=f"{MODEL_LABELS[name]} ({key.split('_')[0]})",
            )
        # end for train / validation curve
    # end for trained model
    ridge_validation = np.mean([scores["ridge"]["validation_r2"] for scores in all_scores])
    axes[0].axhline(
        ridge_validation, color=MODEL_COLORS["ridge"], linewidth=2, linestyle=":",
        label=f"{MODEL_LABELS['ridge']} validation ({ridge_validation:.3f})",
    )
    axes[0].set(
        xlabel="Epoch", ylabel="R2 (pooled over bins and units)",
        title="Training curves (mean over splits)", ylim=(-0.15, None),
    )
    axes[0].set_xscale("log")

    # Panels 2-4: per-bin test metrics, mean +- SEM over splits.
    panels = (
        ("r2_by_bin", "Test R2", "R2 per time bin"),
        ("stim_r_by_bin", "Mean stim_r over units", "Across-image correlation per bin"),
        ("rsa_by_bin", f"{cfg.drsa_rsa_metric} RDM correlation", "Matched-time RSA per bin"),
    )
    for axis, (key, ylabel, title) in zip(axes[1:4], panels):
        for name in models:
            values = np.stack([scores[name][key] for scores in all_scores])
            mean = np.nanmean(values, axis=0)
            sem = np.nanstd(values, axis=0) / np.sqrt(len(values))
            axis.plot(time_ms, mean, linewidth=2, color=MODEL_COLORS[name],
                      label=MODEL_LABELS[name])
            axis.fill_between(time_ms, mean - sem, mean + sem,
                              color=MODEL_COLORS[name], alpha=0.15, linewidth=0)
        # end for model
        if key == "r2_by_bin":
            ceiling = np.mean([extras["ceiling_by_bin"] for extras in all_extras], axis=0)
            axis.plot(time_ms, ceiling, color="black", linestyle=":", linewidth=2,
                      label="Trial-mean noise ceiling")
        # end if the ceiling applies
        axis.axhline(0, color="gray", linewidth=0.8)
        axis.set(xlabel="Time from image onset (ms)", ylabel=ylabel, title=title)

    # end for per-bin panel

    # Panel 5: SGD vs closed form at the same alpha (mean over splits).
    sweep_keys = [k for k in all_extras[0]["sweep"] if k != "ridge"]
    sweep_alphas = np.array([all_extras[0]["sweep"][k]["alpha"] for k in sweep_keys])
    plot_alphas = np.where(sweep_alphas == 0, 1e0, sweep_alphas)
    for field, style, label in (
        ("test_r2", "-o", "SGD linear, test"),
        ("validation_r2", "--o", "SGD linear, validation"),
    ):
        values = np.mean(
            [[extras["sweep"][k][field] for k in sweep_keys] for extras in all_extras], axis=0
        )
        axes[4].plot(plot_alphas, values, style, color=MODEL_COLORS["timebin_linear"],
                     linewidth=2, markersize=7, label=label)
    # end for sweep field
    closed_form = np.mean([extras["closed_form_test_r2"] for extras in all_extras], axis=0)
    axes[4].plot(plot_alphas, closed_form, "-s", color=MODEL_COLORS["ridge"],
                 linewidth=2, markersize=7, label="Closed-form ridge, test")
    for extras in all_extras:
        axes[4].axvline(extras["ridge_alpha"], color=MODEL_COLORS["ridge"],
                        linewidth=0.8, alpha=0.4)
    # end for split's RidgeCV alpha
    axes[4].set_xscale("log")
    axes[4].set(xlabel="Ridge-equivalent alpha (alpha=0 drawn at 1)",
                ylabel="R2", title="Same objective: SGD vs closed form",
                ylim=(-0.05, None))

    # Panel 6: pooled test R2 per split, as a fraction of the noise ceiling.
    for position, name in enumerate(models):
        fractions = [scores[name]["fraction_of_ceiling"] for scores in all_scores]
        axes[5].bar(position, np.mean(fractions), width=0.6,
                    color=MODEL_COLORS[name], alpha=0.85)
        axes[5].scatter(np.full(len(fractions), position), fractions,
                        color="black", s=20, zorder=3)
    # end for model
    axes[5].set_xticks(range(len(models)))
    axes[5].set_xticklabels([MODEL_LABELS[name] for name in models], fontsize=8)
    axes[5].set(ylabel="Test R2 / noise-ceiling R2", title="Pooled test R2 (dots = splits)")

    for axis in axes:
        axis.grid(alpha=0.25)
        if axis is not axes[5]:
            axis.legend(fontsize=8)
        # end if the panel carries a legend
    # end for axis
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
# EOF


def main():
    cfg = parse_args()
    if cfg.smoke_test:
        cfg.seeds = "0"
        cfg.epochs = 3
        cfg.linear_alphas = "0,ridge"
    # end if a smoke run was requested
    paths = config[cfg.env or ENV]["paths"]
    sys.path.insert(0, paths["useful_stuff_path"])
    from useful_stuff.image_processing.computational_models import imgANN

    device = resolve_device(cfg.device)
    output_dir = Path(
        cfg.output_dir
        or Path(paths["data_path"]) / "results" / "three0_linear_readout_comparison"
    ).expanduser()
    if cfg.smoke_test:
        output_dir = output_dir / "smoke"
    # end if smoke outputs must not mix with real runs
    run_dir = output_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    targets, ann_index, _ = load_three0_targets(
        paths, cfg.brain_area, cfg.target_fs, cfg.time_start_ms, cfg.time_end_ms,
        cfg.stimuli_folder,
    )
    noise_variance, _ = load_three0_trial_noise(
        paths, cfg.brain_area, cfg.target_fs, cfg.time_start_ms, cfg.time_end_ms,
        cache_path=Path(paths["data_path"]) / "data"
        / f"three0_{cfg.brain_area}_{cfg.target_fs}Hz_trial_noise.npz",
    )
    from IT_recap.hf_feature_extraction import load_hf_layer_features

    # Cached features are in ImageFolder order; ann_index aligns them to targets.
    features = load_hf_layer_features(
        output_dir=Path(paths["data_path"]) / "models",
        dataset_name=cfg.stimuli_folder,
        model_name=cfg.model_name,
        img_size=cfg.img_size,
        layer_names=cfg.layer_names.split(","),
        pooling=cfg.pooling,
        image_indices=ann_index,
    )
    print(
        f"device {device} | targets {targets.shape} | features {features.shape}"
    )

    encoder = imgANN(
        model_name=cfg.model_name, pkg="hf", img_size=cfg.img_size,
        pooling=cfg.pooling, dtype=torch.float32, attn_implementation="sdpa",
        repo_url=cfg.model_source, trust_remote_code=True,
    )
    seeds = [int(seed) for seed in cfg.seeds.split(",")]
    for seed in seeds:
        print(f"split seed {seed}")
        run_split(cfg, seed, targets, ann_index, features, encoder, device, run_dir)
    # end for split

    all_scores, all_extras = [], []
    for seed in seeds:
        scores, extras = score_split(cfg, seed, targets, ann_index, noise_variance, run_dir)
        all_scores.append(scores)
        all_extras.append(extras)
    # end for split
    plot_comparison(cfg, all_scores, all_extras, output_dir / "linear_readout_comparison.png")

    # Summary table: mean +- SEM over splits.
    rows = []
    for name in ("ridge", "timebin_linear", "timebin_linear_at_ridge_alpha", "baseline"):
        if name not in all_scores[0]:
            continue
        # end if this row was not computed
        row = {"model": name, "n_parameters": all_scores[0][name]["n_parameters"]}
        for key in ("validation_r2", "r2", "fraction_of_ceiling"):
            values = np.array([scores[name][key] for scores in all_scores])
            row[key] = float(values.mean())
            row[f"{key}_sem"] = float(values.std() / np.sqrt(len(values)))
        # end for scalar score
        for key in ("stim_r_by_bin", "rsa_by_bin"):
            row[f"mean_{key}"] = float(
                np.nanmean([np.nanmean(scores[name][key]) for scores in all_scores])
            )
        # end for per-bin score
        rows.append(row)
    # end for model
    summary = {
        "rows": rows,
        "per_split": [
            {
                "seed": seed,
                "ridge_alpha": extras["ridge_alpha"],
                "selected_linear_alpha": extras["selected_alpha"],
                "ceiling_r2": float(extras["ceiling_overall"]),
                "sweep": extras["sweep"],
                **{name: round(float(scores[name]["r2"]), 5) for name in scores},
            }
            for seed, scores, extras in zip(seeds, all_scores, all_extras)
        ],
    }
    with open(output_dir / "results.json", "w") as results_file:
        json.dump(summary, results_file, indent=2, default=float)
    # end with saved metrics
    with open(output_dir / "config.json", "w") as config_out:
        json.dump(asdict(cfg), config_out, indent=2)
    # end with saved configuration

    header = (
        f"{'model':<32}{'params':>11}{'val R2':>9}{'test R2':>14}"
        f"{'frac ceil':>11}{'stim_r':>9}{'RSA':>8}"
    )
    print("\n" + header + "\n" + "-" * len(header))
    for row in rows:
        print(
            f"{row['model']:<32}{row['n_parameters']:>11,}{row['validation_r2']:>9.4f}"
            f"{row['r2']:>8.4f}±{row['r2_sem']:.4f}{row['fraction_of_ceiling']:>11.3f}"
            f"{row['mean_stim_r_by_bin']:>9.4f}{row['mean_rsa_by_bin']:>8.4f}"
        )
    # end for summary row
    print(f"\nsaved to {output_dir}")
# EOF


if __name__ == "__main__":
    main()
# EOC
