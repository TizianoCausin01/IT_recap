"""
Compare four decoders of the trial-averaged AIT response (natraster) for Paul
and three0, all fed the same mean-pooled DINOv3 layers:

    ridge_site      closed-form ridge with one LOO-selected alpha per
                    (time bin, site): every output is its own ridge problem
    ridge_timebin   one linear map per time bin predicting all sites jointly,
                    a one-layer network fitted by SGD on the squared Frobenius
                    norm of the population residual at each bin, plus a ridge
                    penalty sum_t alpha_t / n_fit ||W_t||^2. alpha_t is RidgeCV's
                    choice for bin t with one alpha shared by its sites
    baseline        BaselineModel: learned temporal queries cross-attend to the
                    DINOv3 features, one population readout per time bin
    site_token      SiteTokenBaselineModel: the same decoder, but every site is
                    predicted separately from a query = time embedding + learned
                    site token, so sites share every weight except their token
    site_token_single (optional, add it to --models) the same model trained on
                    (image, site) pairs: each pass predicts one site's time
                    course, and at inference the per-site passes of an image are
                    stacked into the population vector. Sites never interact, so
                    the function is identical; an epoch covers every pair, which
                    gives N times more optimizer steps per epoch
    site_token_matched (optional) the joint site_token model with batch size and
                    epochs set so its number of optimizer updates and targets per
                    update match site_token_single

The SGD models train for a fixed number of epochs (100) and keep the epoch with
the best validation R2. Test scores are reported per time bin as R2, mean
across-image correlation over sites, and matched-time RSA. Every
(dataset, split, model) run is written to disk before the next starts, so a
killed run resumes.
"""

import argparse
import copy
import json
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from torch import nn  # noqa: E402


ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python_scripts" / "src"))

with open(PROJECT_ROOT / "config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)
# end with project configuration
sys.path.insert(0, config[ENV]["paths"]["useful_stuff_path"])

from IT_recap.dynamic_drsa import prepared_rdm_time_series  # noqa: E402
from IT_recap.hf_feature_extraction import (  # noqa: E402
    is_valid_image_file,
    load_hf_layer_features,
)
from IT_recap.neural_prediction_training import stimulus_correlation  # noqa: E402
from IT_recap.three0_decoding import make_image_splits  # noqa: E402
from IT_recap.tvsd_experiments import resolve_device  # noqa: E402
from model_classes.temporal_experiment_models import (  # noqa: E402
    TimebinLinearRegression,
)
from model_classes.temporal_models import (  # noqa: E402
    BaselineModel,
    SiteTokenBaselineModel,
)
from project_specific_utils.dataloader import (  # noqa: E402
    load_img_natraster,
    map_image_order_from_ann_to_monkey,
)
from run_three0_linear_readout_comparison import (  # noqa: E402
    overall_r2,
    pad_curves,
    r2_by_timebin,
)


# (monkey, recording date, stimulus folder) of each natraster dataset.
DATASETS = {
    "paul": ("paul", "230204", "fewer_occlusion"),
    "three0": ("three0", "250313", "talia_20each_tizi"),
}
MODELS = (
    "ridge_site", "ridge_timebin", "baseline", "site_token",
    "site_token_single", "site_token_matched",
)
MODEL_COLORS = {
    "ridge_site": "#2a78d6",
    "ridge_timebin": "#1baf7a",
    "baseline": "#eb6834",
    "site_token": "#8a4fd3",
    "site_token_single": "#e34948",
    "site_token_matched": "#6d6d6d",
}
MODEL_LABELS = {
    "ridge_site": "Ridge per bin & site",
    "ridge_timebin": "Ridge per bin (SGD, Frobenius)",
    "baseline": "Baseline attention",
    "site_token": "Baseline + site token",
    "site_token_single": "Site token, one site per pass",
    "site_token_matched": "Site token, joint, matched updates",
}


@dataclass
class Cfg:
    env: str | None = None
    output_dir: str | None = None
    datasets: str = "paul,three0"
    # Subset of MODELS to fit, score, and plot; ridge_site is the reference.
    models: str = "ridge_site,ridge_timebin,baseline,site_token"

    # Neural target: trial-averaged natraster, as in the baseline notebooks.
    brain_area: str = "AIT"
    target_fs: int = 100
    time_start_ms: float = 0.0
    time_end_ms: float = 300.0

    # Frozen DINOv3 features, mean-pooled and cached.
    model_name: str = "dino_v3_l"
    img_size: int = 224
    pooling: str = "mean"
    layer_names: str = (
        "layer.3.mlp.down_proj,layer.13.mlp.down_proj,layer.20.mlp.down_proj"
    )

    # Splits: every seed is an independent 60/20/20 image split.
    seeds: str = "0,1,2"
    validation_fraction: float = 0.2
    test_fraction: float = 0.2

    # Ridge alpha grid on z-scored features (wide: narrow grids pin at the edge).
    ridge_alpha_min_exp: float = 0.0
    ridge_alpha_max_exp: float = 9.0
    ridge_n_alphas: int = 19

    # Per-bin SGD ridge: plain SGD + momentum, lr = scale / top eigenvalue of
    # X'X/N (Adam's sign-like steps overfit the 3072-d design within an epoch).
    linear_lr_scale: float = 0.05
    linear_momentum: float = 0.9

    # Attention decoders, as configured in noise_layer_baseline_model_dev.ipynb.
    temporal_embedding_dim: int = 256
    value_dim: int = 128
    n_attention_heads: int = 2
    temporal_compression_ratio: int = 3
    mlp_hidden_dim: int = 32
    dropout: float = 0.3
    attention_granularity: str = "feature"
    baseline_learning_rate: float = 1e-3
    baseline_weight_decay: float = 1e-4

    # Shared optimization: fixed epoch budget, best validation R2 checkpoint.
    batch_size: int = 64
    epochs: int = 100
    # site_token_matched: three0 has 466 fit images x 25 sites, so single-site
    # training makes ceil(11650 / 64) * 100 = 18,200 updates of 64 targets.
    # Batches of 3 images (75 targets) give 156 updates per epoch, and 117
    # epochs give 18,252 updates.
    matched_batch_size: int = 3
    matched_epochs: int = 117
    # Evaluation chunk size, independent of the training batch.
    eval_batch_size: int = 64

    # Dynamic RSA settings of the notebooks.
    drsa_rdm_metric: str = "euclidean"
    drsa_rsa_metric: str = "spearman"

    device: str = "auto"
    smoke_test: bool = False


"""
parse_args
Parse command-line overrides into the experiment configuration.

OUTPUT:
    - cfg: Cfg -> data, split, and optimization settings
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


class CachedFeatureEncoder:
    """
    Minimal imgANN stand-in for BaselineModel when every forward pass uses
    cached activations: it supplies the layer width and a parameter-free
    backbone, so DINOv3 never has to be loaded.
    """

    def __init__(self, encoder_dim):
        self.model = nn.Identity()
        self.encoder_dim = encoder_dim
        self.features = {}

    def set_relevant_layers(self, relevant_layers):
        self.layer_names = list(relevant_layers)

    def get_layer_output_shape(self, layer_name):
        return (1, self.encoder_dim)

    def create_forward_hook(self, layer_names=None):
        return None
# EOC


"""
load_dataset
Load one natraster response and the cached DINOv3 layers in the same order.

INPUT:
    - paths: dict -> environment paths from config.yaml
    - dataset_name: str -> key of DATASETS
    - cfg: Cfg -> time window, area, and feature settings

OUTPUT:
    - targets: np.ndarray -> [images, time, sites] trial-averaged responses
    - ann_index: np.ndarray -> ImageFolder index of every target row
    - features: np.ndarray -> [images, layers, embedding] aligned features
"""
def load_dataset(paths, dataset_name, cfg):
    from torchvision.datasets import ImageFolder

    monkey_name, date, folder_name = DATASETS[dataset_name]
    image_dataset = ImageFolder(
        Path(paths["livingstone_lab"]) / "Stimuli" / folder_name,
        is_valid_file=is_valid_image_file,
        allow_empty=True,
    )
    # natraster rows follow the sorted unique image names; map them to ANN order.
    ann_index = np.asarray(
        map_image_order_from_ann_to_monkey(paths, monkey_name, date, image_dataset),
        dtype=int,
    )
    raster = load_img_natraster(
        paths=paths,
        monkey_name=monkey_name,
        date=date,
        original_fs=1000,
        new_fs=cfg.target_fs,
        time_start_ms=cfg.time_start_ms,
        time_end_ms=cfg.time_end_ms,
        brain_area=cfg.brain_area,
    )
    # TimeSeries stores [sites, time, images]; the decoders want image-major.
    targets = raster.get_array().transpose(2, 1, 0).astype(np.float32)
    features = load_hf_layer_features(
        output_dir=Path(paths["data_path"]) / "models",
        dataset_name=folder_name,
        model_name=cfg.model_name,
        img_size=cfg.img_size,
        layer_names=cfg.layer_names.split(","),
        pooling=cfg.pooling,
        image_indices=ann_index,
    )
    return targets, ann_index, features.astype(np.float32)
# EOF


"""
train_decoder
Minibatch-train one decoder for cfg.epochs on in-memory tensors and restore
the epoch with the best validation R2. The unpenalized train and validation
MSE are logged every epoch, so all models share one loss-curve scale.

INPUT:
    - model: nn.Module -> decoder, already on the device
    - predict: callable -> features tensor -> [B, time, sites] predictions
    - loss_fn: callable -> (predictions, targets, model) -> scalar loss
    - optimizer: torch.optim.Optimizer -> optimizer over trainable parameters
    - data: dict -> "train"/"validation"/"test" (features, targets) on device
    - cfg: Cfg -> epochs and batch size
    - seed: int -> shuffling seed
    - train_batch: callable | None -> item indices -> (predictions, targets);
      None trains on whole images, train_batch(b) = (predict(x[b]), y[b])
    - n_train_items: int | None -> items per epoch for train_batch (default:
      the number of training images)

OUTPUT:
    - history: dict -> per-epoch objective, train/validation MSE, validation R2
    - best_epoch: int -> restored epoch (0 = initialization)
"""
def train_decoder(
    model, predict, loss_fn, optimizer, data, cfg, seed, train_batch=None, n_train_items=None
):
    train_x, train_y = data["train"]
    if train_batch is None:
        train_batch = lambda batch: (predict(train_x[batch]), train_y[batch])
        n_train_items = len(train_x)
    # end if training items are whole images
    generator = torch.Generator().manual_seed(seed)
    zero_mean = np.zeros(train_y.shape[1:])  # targets are fit-split centered

    def predict_split(split):
        # Batched, since the site-token attention scales with images x queries.
        model.eval()
        with torch.no_grad():
            predictions = [
                predict(chunk).cpu().numpy()
                for chunk in data[split][0].split(cfg.eval_batch_size)
            ]
        # end with no gradient tracking
        return np.concatenate(predictions)
    # EOF

    def evaluate(split):
        predictions = predict_split(split)
        targets = data[split][1].cpu().numpy()
        mse = float(np.mean((predictions - targets) ** 2))
        return mse, overall_r2(predictions, targets, zero_mean)
    # EOF

    history = {"objective": [], "train_mse": [], "validation_mse": [], "validation_r2": []}
    best_r2, best_epoch = evaluate("validation")[1], 0
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = torch.randperm(n_train_items, generator=generator)
        epoch_loss = 0.0
        for start in range(0, len(order), cfg.batch_size):
            batch = order[start : start + cfg.batch_size].to(train_x.device)
            loss = loss_fn(*train_batch(batch), model)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        # end for minibatch
        validation_mse, validation_r2 = evaluate("validation")
        history["objective"].append(epoch_loss / len(order))
        history["train_mse"].append(evaluate("train")[0])
        history["validation_mse"].append(validation_mse)
        history["validation_r2"].append(validation_r2)
        if validation_r2 > best_r2:
            best_r2, best_epoch = validation_r2, epoch
            best_state = copy.deepcopy(model.state_dict())
        # end if this epoch is the best checkpoint
    # end for epoch
    model.load_state_dict(best_state)

    test_predictions = predict_split("test")
    trainable = [p for p in model.parameters() if p.requires_grad]
    return {
        "test_predictions": test_predictions,
        "best_epoch": best_epoch,
        "n_parameters": int(sum(p.numel() for p in trainable)),
        **{f"{key}_curve": np.asarray(values) for key, values in history.items()},
    }
# EOF


"""
fit_ridge_site
Closed-form ridge with a separate LOO-selected alpha for every (bin, site).

INPUT:
    - design: np.ndarray -> [images, layers * embedding] z-scored features
    - centered: np.ndarray -> [images, time, sites] fit-centered targets
    - splits: dict -> row indices per split
    - alphas: np.ndarray -> candidate penalties

OUTPUT:
    - run: dict -> test predictions, validation MSE, alphas, parameter count
"""
def fit_ridge_site(design, centered, splits, alphas):
    n_time, n_sites = centered.shape[1:]
    train_rows = splits["train"]
    # alpha_per_target: GCV picks one alpha for each of the T * N outputs.
    ridge = RidgeCV(alphas=alphas, alpha_per_target=True).fit(
        design[train_rows], centered[train_rows].reshape(len(train_rows), -1)
    )
    validation_predictions = ridge.predict(design[splits["validation"]])
    return {
        "test_predictions": ridge.predict(design[splits["test"]]).reshape(
            -1, n_time, n_sites
        ),
        "validation_mse": float(
            np.mean((validation_predictions.reshape(-1, n_time, n_sites)
                     - centered[splits["validation"]]) ** 2)
        ),
        "alphas": ridge.alpha_.reshape(n_time, n_sites),
        "n_parameters": int(ridge.coef_.size + np.size(ridge.intercept_)),
    }
# EOF


"""
fit_ridge_timebin
Per-bin ridge, all sites of a bin fitted together by SGD. alpha_t is picked by
RidgeCV on bin t with a single alpha shared by its sites; the loss is the
squared Frobenius norm of the [time, sites] residual per image plus
sum_t (alpha_t / n_fit) ||W_t||^2, so n_fit times the full-batch loss is the
per-bin ridge objective (intercept unpenalized, as in sklearn).

INPUT:
    - design: np.ndarray -> [images, layers * embedding] z-scored features
    - centered, splits, alphas -> as in fit_ridge_site
    - data: dict -> torch tensors of the z-scored layers per split
    - shapes: tuple -> (n_layers, feature_dim, n_time, n_sites)
    - cfg, seed, device -> optimization settings

OUTPUT:
    - run: dict -> train_decoder output plus the per-bin alphas
"""
def fit_ridge_timebin(design, centered, splits, alphas, data, shapes, cfg, seed, device):
    train_rows = splits["train"]
    timebin_alphas = np.array([
        RidgeCV(alphas=alphas).fit(design[train_rows], centered[train_rows, t]).alpha_
        for t in range(shapes[2])
    ])
    # One penalty per bin, broadcast over that bin's [inputs, sites] map.
    l2_lambda = torch.as_tensor(
        timebin_alphas / len(train_rows), dtype=torch.float32, device=device
    )

    torch.manual_seed(seed)
    model = TimebinLinearRegression(*shapes).to(device)

    def loss_fn(predictions, targets, net):
        # Sum over (time, site) = squared Frobenius norm per image; mean over batch.
        frobenius = (predictions - targets).square().sum(dim=(1, 2)).mean()
        penalty = (l2_lambda * net.weight.square().sum(dim=(1, 2))).sum()
        return frobenius + penalty
    # EOF

    # Top curvature of the data term: largest eigenvalue of X'X/N, read from
    # the N x N Gram matrix (it shares the nonzero spectrum).
    train_x = torch.as_tensor(design[train_rows], dtype=torch.float64)
    top_eigenvalue = float(torch.linalg.eigvalsh(train_x @ train_x.T / len(train_x)).max())
    # Bin t has curvature top_eigenvalue + lambda_t, and lambda_t spans orders
    # of magnitude (pre-onset bins choose huge alphas). Rescaling bin t's
    # gradient gives each bin its own stable step, scale / (eig + lambda_t).
    gradient_scale = (top_eigenvalue / (top_eigenvalue + l2_lambda))[:, None, None]
    model.weight.register_hook(lambda gradient: gradient * gradient_scale)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.linear_lr_scale / top_eigenvalue,
        momentum=cfg.linear_momentum,
    )
    run = train_decoder(
        model, lambda features: model(features)[0], loss_fn, optimizer, data, cfg, seed
    )
    return {**run, "alphas": timebin_alphas}
# EOF


"""
fit_attention_model
Train BaselineModel or SiteTokenBaselineModel on raw cached layer vectors with
the notebooks' time-bin MSE loss and AdamW weight decay.

INPUT:
    - model_class: type -> BaselineModel or SiteTokenBaselineModel
    - data: dict -> torch tensors of the raw layers per split
    - shapes: tuple -> (n_layers, feature_dim, n_time, n_sites)
    - cfg, seed, device -> optimization settings
    - single_site: bool -> train SiteTokenBaselineModel on (image, site) pairs
      and predict populations by stacking one pass per site token

OUTPUT:
    - run: dict -> train_decoder output
"""
def fit_attention_model(model_class, data, shapes, cfg, seed, device, single_site=False):
    torch.manual_seed(seed)
    model = model_class(
        CachedFeatureEncoder(shapes[1]),
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
    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(),
        lr=cfg.baseline_learning_rate,
        weight_decay=cfg.baseline_weight_decay,
    )
    loss_fn = lambda predictions, targets, _: (predictions - targets).square().mean()
    if not single_site:
        return train_decoder(
            model,
            lambda features: model(features, use_precomputed_features=True)[0],
            loss_fn,
            optimizer,
            data,
            cfg,
            seed,
        )
    # end if all sites are predicted in one pass

    train_x, train_y = data["train"]
    n_sites = shapes[3]

    def train_batch(pair_indices):
        # Pair k = (image k // N, site k % N): one site's [T] time course each.
        image_indices, site_indices = pair_indices // n_sites, pair_indices % n_sites
        predictions = model(
            train_x[image_indices], use_precomputed_features=True, site_indices=site_indices
        )[0]
        return predictions, train_y[image_indices, :, site_indices]
    # EOF

    def predict_population(features):
        # One pass per site token, stacked over sites: [B, T, N].
        return torch.stack(
            [
                model(
                    features,
                    use_precomputed_features=True,
                    site_indices=torch.full(
                        (len(features),), site, dtype=torch.long, device=features.device
                    ),
                )[0]
                for site in range(n_sites)
            ],
            dim=-1,
        )
    # EOF

    return train_decoder(
        model, predict_population, loss_fn, optimizer, data, cfg, seed,
        train_batch=train_batch, n_train_items=len(train_x) * n_sites,
    )
# EOF


"""
run_split
Fit the four models on one image split, writing every run before the next.

INPUT:
    - cfg: Cfg -> experiment settings
    - seed: int -> split seed
    - targets, ann_index, features -> load_dataset output
    - device: torch.device -> compute device
    - run_dir: Path -> per-run output directory
"""
def run_split(cfg, seed, targets, ann_index, features, device, run_dir):
    splits = make_image_splits(ann_index, seed, cfg.validation_fraction, cfg.test_fraction)
    n_images, n_layers, feature_dim = features.shape
    shapes = (n_layers, feature_dim, *targets.shape[1:])

    # Center targets on the fit split, per (time, site), as the notebooks do.
    centered = targets - targets[splits["train"]].mean(0)

    # Linear models see the fit-split z-scored flat design; the attention
    # models see raw layer vectors and normalize them internally.
    flat = features.reshape(n_images, -1)
    design = (flat - flat[splits["train"]].mean(0)) / (flat[splits["train"]].std(0) + 1e-6)
    alphas = np.logspace(cfg.ridge_alpha_min_exp, cfg.ridge_alpha_max_exp, cfg.ridge_n_alphas)

    def tensors(inputs):
        return {
            split: (
                torch.as_tensor(inputs[rows], device=device),
                torch.as_tensor(centered[rows], device=device),
            )
            for split, rows in splits.items()
        }
    # EOF

    for name in cfg.models.split(","):
        run_path = run_dir / f"seed{seed}_{name}.npz"
        if run_path.is_file():
            continue
        # end if this run already finished
        if name == "ridge_site":
            run = fit_ridge_site(design, centered, splits, alphas)
        elif name == "ridge_timebin":
            run = fit_ridge_timebin(
                design, centered, splits, alphas,
                tensors(design.reshape(features.shape).astype(np.float32)),
                shapes, cfg, seed, device,
            )
        else:
            model_class = BaselineModel if name == "baseline" else SiteTokenBaselineModel
            model_cfg = cfg
            if name == "site_token_matched":
                model_cfg = replace(
                    cfg, batch_size=cfg.matched_batch_size, epochs=cfg.matched_epochs
                )
            # end if the update count must match the single-site run
            run = fit_attention_model(
                model_class, tensors(features), shapes, model_cfg, seed, device,
                single_site=name == "site_token_single",
            )
        # end if closed-form ridge, SGD ridge, or attention model
        np.savez_compressed(run_path, **run)
        best = f" | best epoch {run['best_epoch']}" if "best_epoch" in run else ""
        print(f"  seed {seed} | {name}{best} | params {run['n_parameters']:,}", flush=True)
        if device.type == "mps":
            torch.mps.empty_cache()
        # end if MPS memory should be released between runs
    # end for model
# EOF


"""
score_split
Score every model on one split's test images, per time bin.

OUTPUT:
    - scores: dict -> per-model per-bin R2, correlation, RSA, pooled R2, curves
"""
def score_split(cfg, seed, targets, ann_index, run_dir):
    splits = make_image_splits(ann_index, seed, cfg.validation_fraction, cfg.test_fraction)
    test_targets = targets[splits["test"]] - targets[splits["train"]].mean(0)
    zero_mean = np.zeros(test_targets.shape[1:])
    # Prepared RDMs are rank-transformed and unit-normed: [time, image pairs].
    neural_rdms = prepared_rdm_time_series(
        test_targets, cfg.drsa_rdm_metric, cfg.drsa_rsa_metric
    )

    scores = {}
    for name in cfg.models.split(","):
        run = dict(np.load(run_dir / f"seed{seed}_{name}.npz"))
        predictions = run["test_predictions"]
        prediction_rdms = prepared_rdm_time_series(
            predictions, cfg.drsa_rdm_metric, cfg.drsa_rsa_metric
        )
        scores[name] = {
            "r2_by_bin": r2_by_timebin(predictions, test_targets, zero_mean),
            # Pearson across test images per (bin, site), averaged over sites.
            "corr_by_bin": np.nanmean(stimulus_correlation(predictions, test_targets), axis=1),
            # Matched-time RSA: the diagonal of the dRSA matrix.
            "rsa_by_bin": (neural_rdms * prediction_rdms).sum(axis=1),
            "r2": overall_r2(predictions, test_targets, zero_mean),
            **{key: value for key, value in run.items() if key != "test_predictions"},
        }
    # end for model
    return scores
# EOF


"""
plot_dataset
Loss curves plus per-bin test R2, correlation, and RSA for one dataset.

INPUT:
    - cfg: Cfg -> time axis settings
    - dataset_name: str -> panel title prefix
    - all_scores: list[dict] -> score_split output per seed
    - output_path: Path -> figure destination
"""
def plot_dataset(cfg, dataset_name, all_scores, output_path):
    n_time = len(all_scores[0]["ridge_site"]["r2_by_bin"])
    time_ms = cfg.time_start_ms + np.arange(n_time) * 1000.0 / cfg.target_fs
    figure, axes = plt.subplots(1, 4, figsize=(22, 5))

    # Panel 1: train (dashed) and validation (solid) MSE, mean over splits.
    for name in cfg.models.split(",")[1:]:
        for key, style in (("validation_mse_curve", "-"), ("train_mse_curve", "--")):
            curves = pad_curves([scores[name][key] for scores in all_scores])
            axes[0].plot(np.arange(1, curves.shape[1] + 1), np.nanmean(curves, axis=0),
                         style, linewidth=2, color=MODEL_COLORS[name],
                         label=MODEL_LABELS[name] if style == "-" else None)
        # end for train / validation curve
    # end for trained model
    ridge_mse = np.mean([scores["ridge_site"]["validation_mse"] for scores in all_scores])
    axes[0].axhline(ridge_mse, color=MODEL_COLORS["ridge_site"], linewidth=2,
                    linestyle=":", label=f"{MODEL_LABELS['ridge_site']} (validation)")
    axes[0].set(xlabel="Epoch", ylabel="MSE (solid = validation, dashed = train)",
                title=f"{dataset_name}: loss curves")

    # Panels 2-4: per-bin test metrics, mean +- SEM over splits.
    panels = (
        ("r2_by_bin", "Test R2 (pooled over sites)", "R2 per time bin"),
        ("corr_by_bin", "Mean Pearson r over sites", "Correlation per time bin"),
        ("rsa_by_bin", f"{cfg.drsa_rsa_metric} RDM correlation", "RSA per time bin"),
    )
    for axis, (key, ylabel, title) in zip(axes[1:], panels):
        for name in cfg.models.split(","):
            values = np.stack([scores[name][key] for scores in all_scores])
            mean = np.nanmean(values, axis=0)
            sem = np.nanstd(values, axis=0) / np.sqrt(len(values))
            axis.plot(time_ms, mean, linewidth=2, color=MODEL_COLORS[name],
                      label=MODEL_LABELS[name])
            axis.fill_between(time_ms, mean - sem, mean + sem,
                              color=MODEL_COLORS[name], alpha=0.15, linewidth=0)
        # end for model
        axis.axhline(0, color="gray", linewidth=0.8)
        axis.set(xlabel="Time from image onset (ms)", ylabel=ylabel,
                 title=f"{dataset_name}: test {title}")
    # end for per-bin panel
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    # end for axis
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
# EOF


def main():
    cfg = parse_args()
    if cfg.smoke_test:
        cfg.seeds = "0"
        cfg.epochs = 2
    # end if a smoke run was requested
    paths = config[cfg.env or ENV]["paths"]
    device = resolve_device(cfg.device)
    output_root = Path(
        cfg.output_dir or Path(paths["data_path"]) / "results" / "four_model_comparison"
    ).expanduser()
    if cfg.smoke_test:
        output_root = output_root / "smoke"
    # end if smoke outputs must not mix with real runs
    seeds = [int(seed) for seed in cfg.seeds.split(",")]

    summary = {}
    for dataset_name in cfg.datasets.split(","):
        targets, ann_index, features = load_dataset(paths, dataset_name, cfg)
        print(f"{dataset_name} | device {device} | targets {targets.shape} | "
              f"features {features.shape}", flush=True)
        run_dir = output_root / dataset_name / "runs"
        run_dir.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            run_split(cfg, seed, targets, ann_index, features, device, run_dir)
        # end for split

        all_scores = [score_split(cfg, seed, targets, ann_index, run_dir) for seed in seeds]
        plot_dataset(cfg, dataset_name, all_scores,
                     output_root / dataset_name / "four_model_comparison.png")
        summary[dataset_name] = {}
        for name in cfg.models.split(","):
            pooled = np.array([scores[name]["r2"] for scores in all_scores])
            summary[dataset_name][name] = {
                "test_r2": float(pooled.mean()),
                "test_r2_sem": float(pooled.std() / np.sqrt(len(pooled))),
                "mean_corr": float(np.nanmean([s[name]["corr_by_bin"] for s in all_scores])),
                "mean_rsa": float(np.nanmean([s[name]["rsa_by_bin"] for s in all_scores])),
                "n_parameters": int(all_scores[0][name]["n_parameters"]),
                "best_epochs": [int(s[name].get("best_epoch", -1)) for s in all_scores],
            }
        # end for model
    # end for dataset

    # One file per dataset, so parallel sweeps over different datasets
    # never overwrite each other's summary.
    for dataset_name, rows in summary.items():
        with open(output_root / dataset_name / "results.json", "w") as results_file:
            json.dump(rows, results_file, indent=2)
        # end with saved metrics
        with open(output_root / dataset_name / "config.json", "w") as config_out:
            json.dump(asdict(cfg), config_out, indent=2)
        # end with saved configuration
    # end for dataset
    for dataset_name, rows in summary.items():
        print(f"\n{dataset_name}\n{'model':<16}{'params':>11}{'test R2':>16}"
              f"{'corr':>8}{'RSA':>8}  best epochs")
        for name, row in rows.items():
            print(f"{name:<16}{row['n_parameters']:>11,}{row['test_r2']:>9.4f}"
                  f"±{row['test_r2_sem']:.4f}{row['mean_corr']:>8.4f}"
                  f"{row['mean_rsa']:>8.4f}  {row['best_epochs']}")
        # end for model row
    # end for dataset
    print(f"\nsaved to {output_root}")
# EOF


if __name__ == "__main__":
    main()
# EOC
