import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import sys
    import urllib.request
    from dataclasses import asdict, dataclass, field
    from pathlib import Path

    import h5py
    import matplotlib.pyplot as plt
    import marimo as mo
    import numpy as np
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from transformers import AutoImageProcessor

    return (
        AutoImageProcessor,
        DataLoader,
        Path,
        asdict,
        dataclass,
        field,
        h5py,
        mo,
        np,
        os,
        plt,
        sys,
        torch,
        urllib,
        yaml,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Monkey F TVSD → `BaselineModel`

    This notebook trains the project's time-local `BaselineModel` to predict
    baseline-corrected, time-resolved multi-unit activity (MUA) from frozen
    DINOv3 layer features for the exact THINGS image shown on each presentation.

    The neural file alone is **not a complete supervised example**: it contains
    MUA and stimulus IDs, but not image pixels or ANN features. The preparation
    section therefore aligns the official THINGS images, extracts DINO features
    once, and caches them before decoder training.

    Primary sources:

    - [Papale et al., *Neuron* (2025), DOI 10.1016/j.neuron.2024.12.003](https://doi.org/10.1016/j.neuron.2024.12.003)
    - [TVSD data and code repository](https://gin.g-node.org/paolo_papale/TVSD)
    - [Official channel-remapping clarification](https://gin.g-node.org/paolo_papale/TVSD/issues/2)

    The workflow uses a clean three-way split: a seeded subset of the 22,248
    unique training images is held out for model selection; the 100 official
    test images × 30 repetitions remain untouched until final evaluation.
    """)
    return


@app.cell
def _(Path, dataclass, field, os, sys, yaml):
    # Locate the repository whether marimo starts in the root or scripts folder.
    _cwd = Path.cwd().resolve()
    _candidate_roots = [_cwd, *_cwd.parents]
    PROJECT_ROOT = next(
        (_path for _path in _candidate_roots if (_path / "config.yaml").is_file()),
        None,
    )
    if PROJECT_ROOT is None:
        raise FileNotFoundError("Could not locate config.yaml from this notebook.")
    # end if the project root is unavailable

    ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
    with open(PROJECT_ROOT / "config.yaml", "r") as _config_file:
        _project_config = yaml.safe_load(_config_file)
    # end with project configuration
    paths = _project_config[ENV]["paths"]

    # Keep this project's source ahead of other development packages.
    _project_src = str((PROJECT_ROOT / "python_scripts" / "src").resolve())
    _useful_stuff_src = str(Path(paths["useful_stuff_path"]).resolve())
    for _source_path in (_useful_stuff_src, _project_src):
        while _source_path in sys.path:
            sys.path.remove(_source_path)
        # end while a source path is already registered
        sys.path.insert(0, _source_path)
    # end for local source path

    from IT_recap.hf_feature_extraction import ProcessorTransform
    from IT_recap.neural_prediction_training import (
        aggregate_attention_by_layer,
        neural_activity_timebin_mse_loss,
        test_step,
        training_step,
    )
    from IT_recap.tvsd import (
        TVSD_METADATA_COLUMNS,
        TVSD_MONKEY_F_AREAS,
        TVSD_MONKEY_F_ARRAYS,
        TVSDOrderedImageDataset,
        TVSDTrialDataset,
        compute_tvsd_channel_standardization,
        extract_tvsd_ann_features,
        get_tvsd_monkey_f_area_channels,
        load_tvsd_metadata,
        load_tvsd_stimulus_paths,
        make_tvsd_headstage_mapping,
        preprocess_tvsd_mua_targets,
    )
    from model_classes.temporal_models import BaselineModel
    from useful_stuff.image_processing.computational_models import imgANN

    @dataclass
    class Cfg:
        # Local data and cache names are resolved through config.yaml.
        mua_file_name: str = "f_THINGS_MUA_trials.mat"
        things_metadata_file_name: str = "things_imgs.mat"
        things_image_folder_name: str = "THINGS"
        feature_archive_name: str = "tvsd_monkeyF_dino_v3_l_224_features.npz"

        # Neural target: one anatomically coherent ROI at a time.
        area: str = "IT"
        original_fs: int = 1000
        target_fs: int = 100
        baseline_start_ms: float = -100.0
        baseline_end_ms: float = 0.0
        time_start_ms: float = 0.0
        time_end_ms: float = 200.0
        preprocessing_chunk_size: int = 64

        # Frozen DINOv3 representation supplied to BaselineModel.
        model_name: str = "dino_v3_l"
        model_source: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"
        img_size: int = 224
        pooling: str = "mean"
        layer_names: list[str] = field(
            default_factory=lambda: [
                "layer.3.mlp.down_proj",
                "layer.13.mlp.down_proj",
                "layer.16.mlp.down_proj",
                "layer.20.mlp.down_proj",
            ]
        )
        use_fast_processor: bool = True
        trust_remote_code: bool = True
        attn_implementation: str = "sdpa"

        # Split and optimization parameters.
        validation_fraction: float = 0.1
        random_seed: int = 0
        max_training_presentations: int | None = None
        batch_size: int = 64
        feature_extraction_batch_size: int = 64
        num_workers: int = 0
        epochs: int = 20
        learning_rate: float = 1e-3
        weight_decay: float = 1e-4
        temporal_embedding_dim: int = 128
        value_dim: int = 128
        mlp_hidden_dim: int = 64
        dropout: float = 0.2
        attention_granularity: str = "layer"

    # EOC

    cfg = Cfg()
    data_root = Path(paths["data_path"])
    mua_path = data_root / "data"/ cfg.mua_file_name
    things_metadata_path = data_root / cfg.things_metadata_file_name
    things_image_root = (
        Path(paths["livingstone_lab"])
        / "Stimuli"
        / cfg.things_image_folder_name
    )
    feature_archive_path = data_root / "models" / cfg.feature_archive_name
    target_cache_path = (
        data_root
        / "data"
        / (
            f"tvsd_monkeyF_{cfg.area}_{cfg.time_start_ms:g}-"
            f"{cfg.time_end_ms:g}ms_{cfg.target_fs}Hz_baseline_corrected.npy"
        )
    )
    checkpoint_path = (
        data_root
        / "models"
        / f"tvsd_monkeyF_{cfg.area}_baseline_model.pt"
    )
    return (
        BaselineModel,
        ProcessorTransform,
        TVSDOrderedImageDataset,
        TVSDTrialDataset,
        TVSD_METADATA_COLUMNS,
        TVSD_MONKEY_F_AREAS,
        TVSD_MONKEY_F_ARRAYS,
        aggregate_attention_by_layer,
        cfg,
        checkpoint_path,
        compute_tvsd_channel_standardization,
        extract_tvsd_ann_features,
        feature_archive_path,
        imgANN,
        load_tvsd_metadata,
        load_tvsd_stimulus_paths,
        make_tvsd_headstage_mapping,
        mua_path,
        neural_activity_timebin_mse_loss,
        preprocess_tvsd_mua_targets,
        target_cache_path,
        test_step,
        things_image_root,
        things_metadata_path,
        training_step,
    )


@app.cell
def _(
    cfg,
    feature_archive_path,
    mo,
    mua_path,
    target_cache_path,
    things_image_root,
    things_metadata_path,
):
    _requirements = [
        {
            "resource": "MUA source",
            "path": str(mua_path),
            "available": mua_path.is_file(),
            "required for": "targets",
        },
        {
            "resource": "things_imgs.mat",
            "path": str(things_metadata_path),
            "available": things_metadata_path.is_file(),
            "required for": "feature extraction only",
        },
        {
            "resource": "THINGS image root",
            "path": str(things_image_root),
            "available": things_image_root.is_dir(),
            "required for": "feature extraction only",
        },
        {
            "resource": "DINO feature cache",
            "path": str(feature_archive_path),
            "available": feature_archive_path.is_file(),
            "required for": "training",
        },
        {
            "resource": f"{cfg.area} target cache",
            "path": str(target_cache_path),
            "available": target_cache_path.is_file(),
            "required for": "training",
        },
    ]
    mo.vstack(
        [
            mo.md("## Local inputs and caches"),
            mo.ui.table(_requirements, selection=None, pagination=False),
        ]
    )
    return


@app.cell
def _(TVSD_METADATA_COLUMNS, h5py, load_tvsd_metadata, mo, mua_path, np):
    if not mua_path.is_file():
        mo.stop(True, mo.md(f"**Missing neural file:** `{mua_path}`"))
    # end if the local MUA file is absent

    allmat, source_time_ms = load_tvsd_metadata(mua_path)
    with h5py.File(mua_path, "r") as _mua_file:
        _stored_shapes = {
            _key: tuple(_mua_file[_key].shape)
            for _key in ("ALLMAT", "ALLMUA", "tb")
        }
    # end with MUA metadata inspection

    _metadata_summary = [
        {
            "field": _column_name,
            "minimum": int(allmat[:, _column_idx].min()),
            "maximum": int(allmat[:, _column_idx].max()),
            "unique": int(len(np.unique(allmat[:, _column_idx]))),
        }
        for _column_idx, _column_name in enumerate(TVSD_METADATA_COLUMNS)
    ]
    mo.vstack(
        [
            mo.md(
                f"""
                ## What is actually in `f_THINGS_MUA_trials.mat`

                File size: **{mua_path.stat().st_size / 1024**3:.2f} GiB**.

                HDF5 reports `{_stored_shapes['ALLMUA']}` because MATLAB v7.3
                dimensions appear reversed in h5py. The scientific organization is:

                - MATLAB `ALLMUA`: `[raw channel, presentation, time]`
                - h5py `ALLMUA`: `[time, presentation, raw channel]`
                - after preprocessing: `[presentation, binned time, physical channel]`
                - `tb`: {source_time_ms[0]:g}…{source_time_ms[-1]:g} ms at 1 kHz

                There are {len(allmat):,} completed image presentations:
                22,248 unique train images once each, plus 100 test images shown
                30 times each. Every `ALLMAT` row is
                `[trial_idx, train_idx, test_idx, rep, count, day]`. Exactly one of
                `train_idx` and `test_idx` is non-zero. `count` is position 1–4 in
                the four-image sequence; `day` is recording day 1–4.
                """
            ),
            mo.ui.table(_metadata_summary, selection=None, pagination=False),
        ]
    )
    return (allmat,)


@app.cell
def _(
    TVSD_MONKEY_F_AREAS,
    TVSD_MONKEY_F_ARRAYS,
    make_tvsd_headstage_mapping,
    mo,
    np,
):
    headstage_mapping = make_tvsd_headstage_mapping()
    _channel_rows = []
    for _area_name in ("V1", "IT", "V4"):
        _channel_start, _channel_end = TVSD_MONKEY_F_AREAS[_area_name]
        _channel_rows.append(
            {
                "ROI": _area_name,
                "physical MATLAB channels": f"{_channel_start + 1}–{_channel_end}",
                "NumPy slice": f"{_channel_start}:{_channel_end}",
                "Utah arrays": ", ".join(
                    str(_array) for _array in TVSD_MONKEY_F_ARRAYS[_area_name]
                ),
                "MUA sites": _channel_end - _channel_start,
            }
        )
    # end for monkey F visual area

    mo.vstack(
        [
            mo.md(
                r"""
                ## Channel organization — the important part

                `ALLMUA` is in **recording-system order**, not anatomical order.
                The CerePlex/Gemini headstages permute 32-channel banks. This
                notebook reproduces the official `1024chns_mapping_20220105.mat`
                permutation before selecting an ROI. The map is identical for both
                monkeys, but the ROI ranges below are specifically for monkey F.

                After remapping, every contiguous 64 channels are one 8×8 Utah
                array. These outputs are best called **MUA sites/channels**, not
                neurons: each electrode reflects pooled spiking near its tip. The
                64 within-array sites have 400 µm pitch; cross-array distances
                cannot be recovered reliably from the channel number alone.

                The paper's physical-layout plotting order for monkey F is
                `[16,15,14,13,12,11,10,9,1,2,3,4,5,7,6,8]`. That layout is useful
                for topographic plots, but it is separate from the array-contiguous
                channel order used as the model's output axis.
                """
            ),
            mo.ui.table(_channel_rows, selection=None, pagination=False),
            mo.md(
                f"Mapping check: **{len(headstage_mapping)} channels**, valid "
                f"permutation = **{np.array_equal(np.sort(headstage_mapping), np.arange(1024))}**."
            ),
        ]
    )
    return


@app.cell
def _(mo):
    run_pipeline_button = mo.ui.run_button(
        label="Load DINO features + fit ridge and BaselineModel"
    )
    mo.vstack(
        [
            mo.md("## Run the complete comparison"),
            run_pipeline_button,
            mo.md(
                "The button loads existing caches when available. If needed, it "
                "prepares the neural targets, downloads the stimulus-order "
                "metadata, extracts DINO features, then fits ridge and trains "
                "BaselineModel on the same split."
            ),
        ]
    )
    return (run_pipeline_button,)


@app.cell
def _(
    cfg,
    mo,
    mua_path,
    np,
    preprocess_tvsd_mua_targets,
    run_pipeline_button,
    target_cache_path,
):
    if target_cache_path.is_file():
        tvsd_targets = np.load(target_cache_path, mmap_mode="r")
    elif run_pipeline_button.value:
        tvsd_targets = preprocess_tvsd_mua_targets(
            mat_path=mua_path,
            output_path=target_cache_path,
            area=cfg.area,
            time_start_ms=cfg.time_start_ms,
            time_end_ms=cfg.time_end_ms,
            target_fs=cfg.target_fs,
            baseline_start_ms=cfg.baseline_start_ms,
            baseline_end_ms=cfg.baseline_end_ms,
            chunk_size=cfg.preprocessing_chunk_size,
        )
    else:
        mo.stop(
            True,
            mo.md(
                "Click **Load DINO features + fit ridge and BaselineModel** to "
                f"prepare the `{cfg.area}` target cache at `{target_cache_path}`."
            ),
        )
    # end if target cache exists or preprocessing was requested

    mo.md(
        f"Prepared targets: `{tvsd_targets.shape}` = "
        "[presentations, time bins, physical MUA channels]."
    )
    return (tvsd_targets,)


@app.cell
def _(cfg, imgANN, mo, run_pipeline_button, torch):
    if not run_pipeline_button.value:
        mo.stop(
            True,
            mo.md(
                "Click **Load DINO features + fit ridge and BaselineModel** to "
                "start the comparison."
            ),
        )
    # end if no model-backed action was requested

    # BaselineModel requires the wrapped frozen backbone even when cached
    # activations bypass its forward pass during decoder training.
    ann = imgANN(
        model_name=cfg.model_name,
        pkg="hf",
        img_size=cfg.img_size,
        pooling=cfg.pooling,
        dtype=torch.float32,
        attn_implementation=cfg.attn_implementation,
        repo_url=cfg.model_source,
        trust_remote_code=cfg.trust_remote_code,
    )
    return (ann,)


@app.cell
def _(
    AutoImageProcessor,
    ProcessorTransform,
    TVSDOrderedImageDataset,
    ann,
    cfg,
    extract_tvsd_ann_features,
    feature_archive_path,
    load_tvsd_stimulus_paths,
    mo,
    np,
    run_pipeline_button,
    things_image_root,
    things_metadata_path,
    urllib,
):
    if feature_archive_path.is_file():
        with np.load(feature_archive_path) as _feature_archive:
            train_features = _feature_archive["train_features"].astype(
                np.float32, copy=False
            )
            test_features = _feature_archive["test_features"].astype(
                np.float32, copy=False
            )
            _saved_layer_names = _feature_archive["layer_names"].tolist()
        # end with DINO feature archive
    elif run_pipeline_button.value:
        if not things_metadata_path.is_file():
            _metadata_url = (
                "https://gin.g-node.org/paolo_papale/TVSD/raw/master/"
                "monkeyF/_logs/things_imgs.mat"
            )
            things_metadata_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(_metadata_url, things_metadata_path)
        # end if the official image order is missing
        if not things_image_root.is_dir():
            raise FileNotFoundError(
                "Download the THINGS images from things-initiative.org and set "
                f"cfg.things_image_folder_name. Expected {things_image_root}."
            )
        # end if THINGS pixels are absent

        _processor = AutoImageProcessor.from_pretrained(
            cfg.model_source,
            use_fast=cfg.use_fast_processor,
        )
        _transform = ProcessorTransform(_processor)
        _train_paths = load_tvsd_stimulus_paths(
            things_metadata_path,
            split="train",
        )
        _test_paths = load_tvsd_stimulus_paths(
            things_metadata_path,
            split="test",
        )
        _train_images = TVSDOrderedImageDataset(
            things_image_root,
            _train_paths,
            _transform,
        )
        _test_images = TVSDOrderedImageDataset(
            things_image_root,
            _test_paths,
            _transform,
        )
        train_features, test_features = extract_tvsd_ann_features(
            ann=ann,
            train_dataset=_train_images,
            test_dataset=_test_images,
            layer_names=cfg.layer_names,
            output_path=feature_archive_path,
            batch_size=cfg.feature_extraction_batch_size,
            num_workers=cfg.num_workers,
        )
        _saved_layer_names = list(cfg.layer_names)
    else:
        mo.stop(
            True,
            mo.md(
                "Training needs frozen visual features. Supply the existing "
                f"archive at `{feature_archive_path}`, or download THINGS pixels "
                "and run the complete comparison."
            ),
        )
    # end if features are cached or extraction was requested

    if _saved_layer_names != cfg.layer_names:
        raise ValueError(
            "Cached layer order does not match cfg.layer_names: "
            f"{_saved_layer_names} vs {cfg.layer_names}."
        )
    # end if cached feature layers differ from the model
    if train_features.shape[0] != 22248 or test_features.shape[0] != 100:
        raise ValueError(
            "Expected 22,248 train features and 100 test features; found "
            f"{train_features.shape[0]} and {test_features.shape[0]}."
        )
    # end if the visual feature archive is incomplete

    mo.md(
        f"DINO features: train `{train_features.shape}`, test "
        f"`{test_features.shape}`."
    )
    return test_features, train_features


@app.cell
def _(
    DataLoader,
    TVSDTrialDataset,
    allmat,
    cfg,
    compute_tvsd_channel_standardization,
    np,
    test_features,
    torch,
    train_features,
    tvsd_targets,
):
    # Split only the 22,248 unique-image presentations for optimization and
    # validation. The official repeated test pool remains completely untouched.
    official_training_indices = np.flatnonzero(allmat[:, 1] > 0)
    test_trial_indices = np.flatnonzero(allmat[:, 2] > 0)
    _split_rng = np.random.default_rng(cfg.random_seed)
    _shuffled_training_indices = _split_rng.permutation(
        official_training_indices
    )
    _n_validation = max(
        1,
        round(len(official_training_indices) * cfg.validation_fraction),
    )
    validation_trial_indices = _shuffled_training_indices[:_n_validation]
    training_trial_indices = _shuffled_training_indices[_n_validation:]

    # A cap is useful for a smoke run; random selection preserves the clean split.
    if cfg.max_training_presentations is not None:
        training_trial_indices = training_trial_indices[
            :cfg.max_training_presentations
        ]
    # end if a smaller demonstration fit was requested

    channel_mean, channel_scale = compute_tvsd_channel_standardization(
        tvsd_targets,
        training_trial_indices,
    )
    _dataset_inputs = {
        "train_inputs": train_features,
        "test_inputs": test_features,
        "targets": tvsd_targets,
        "metadata": allmat,
        "input_mode": "activations",
        "channel_mean": channel_mean,
        "channel_scale": channel_scale,
    }
    training_dataset = TVSDTrialDataset(
        **_dataset_inputs,
        trial_indices=training_trial_indices,
    )
    validation_dataset = TVSDTrialDataset(
        **_dataset_inputs,
        trial_indices=validation_trial_indices,
    )
    test_dataset = TVSDTrialDataset(
        **_dataset_inputs,
        trial_indices=test_trial_indices,
    )

    _loader_generator = torch.Generator().manual_seed(cfg.random_seed)
    training_loader = DataLoader(
        training_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        generator=_loader_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    return (
        channel_mean,
        channel_scale,
        test_dataset,
        test_loader,
        test_trial_indices,
        training_dataset,
        training_loader,
        validation_dataset,
        validation_loader,
    )


@app.cell
def _(
    allmat,
    cfg,
    channel_mean,
    channel_scale,
    mo,
    np,
    run_pipeline_button,
    test_dataset,
    test_features,
    train_features,
    training_dataset,
    tvsd_targets,
    validation_dataset,
):
    mo.stop(
        not run_pipeline_button.value,
        mo.md(
            "## Ridge-regression reference\n\n"
            "Run the complete comparison to fit ridge and BaselineModel on the "
            "same split."
        ),
    )

    from sklearn.linear_model import RidgeCV

    _fit_indices = np.asarray(training_dataset.trial_indices)
    _validation_indices = np.asarray(validation_dataset.trial_indices)
    _test_indices = np.asarray(test_dataset.trial_indices)

    # Concatenate selected DINO layers and normalize from the fit split only.
    _fit_image_ids = allmat[_fit_indices, 1] - 1
    _validation_image_ids = allmat[_validation_indices, 1] - 1
    _fit_features = train_features[_fit_image_ids].reshape(len(_fit_indices), -1)
    _validation_features = train_features[_validation_image_ids].reshape(
        len(_validation_indices), -1
    )
    _test_image_features = test_features.reshape(len(test_features), -1)
    _feature_mean = _fit_features.mean(axis=0, keepdims=True)
    _feature_scale = _fit_features.std(axis=0, keepdims=True) + 1e-6
    _fit_features = (_fit_features - _feature_mean) / _feature_scale
    _validation_features = (
        _validation_features - _feature_mean
    ) / _feature_scale
    _test_image_features = (_test_image_features - _feature_mean) / _feature_scale

    # Standardize targets with the same per-channel statistics as TVSDTrialDataset.
    def _standardize(raw_targets):
        raw_targets = np.asarray(raw_targets, dtype=np.float32)
        return (
            raw_targets - channel_mean[None, None, :]
        ) / channel_scale[None, None, :]
    # end per-channel target standardization

    _fit_targets = _standardize(tvsd_targets[_fit_indices])
    _validation_targets = _standardize(tvsd_targets[_validation_indices])
    _n_time, _n_channels = _fit_targets.shape[1], _fit_targets.shape[2]

    ridge_model = RidgeCV(alphas=np.logspace(1.0, 7.0, 13))
    ridge_model.fit(
        _fit_features, _fit_targets.reshape(len(_fit_indices), -1)
    )
    _ridge_validation_predictions = ridge_model.predict(_validation_features)
    ridge_validation_mse = float(
        np.mean(
            (
                _ridge_validation_predictions
                - _validation_targets.reshape(len(_validation_indices), -1)
            )
            ** 2
        )
    )

    # Repeated-test pool -> per-image mean response.
    _test_image_ids = allmat[_test_indices, 2] - 1
    _test_trial_targets = _standardize(tvsd_targets[_test_indices])
    _test_image_targets = np.stack(
        [
            _test_trial_targets[_test_image_ids == _image_id].mean(axis=0)
            for _image_id in range(100)
        ]
    )
    ridge_test_predictions = ridge_model.predict(
        _test_image_features
    ).reshape(100, _n_time, _n_channels)

    def _stimulus_correlation(prediction, target):
        # Pearson r across the 100 images, independently per (time, channel).
        _p = prediction - prediction.mean(axis=0, keepdims=True)
        _t = target - target.mean(axis=0, keepdims=True)
        _den = np.sqrt(
            np.square(_p).sum(axis=0) * np.square(_t).sum(axis=0)
        )
        return np.divide(
            (_p * _t).sum(axis=0),
            _den,
            out=np.full(_den.shape, np.nan),
            where=_den > 0,
        )
    # end stimulus-resolved correlation

    ridge_channel_time_correlations = _stimulus_correlation(
        ridge_test_predictions, _test_image_targets
    )
    ridge_median_test_correlation = np.nanmedian(
        ridge_channel_time_correlations, axis=1
    )

    # Split-half noise ceiling: reliability of the 30-repetition test mean,
    # Spearman-Brown corrected to the full average, then sqrt -> stim_r ceiling.
    _rng = np.random.default_rng(0)
    _rows_by_image = [
        np.flatnonzero(_test_image_ids == _image_id) for _image_id in range(100)
    ]
    _half_correlations = []
    for _ in range(40):
        _first_half, _second_half = [], []
        for _rows in _rows_by_image:
            _shuffled = _rng.permutation(_rows)
            _mid = len(_shuffled) // 2
            _first_half.append(
                _test_trial_targets[_shuffled[:_mid]].mean(axis=0)
            )
            _second_half.append(
                _test_trial_targets[_shuffled[_mid:]].mean(axis=0)
            )
        # end for repeated-test image
        _half_correlations.append(
            _stimulus_correlation(
                np.stack(_first_half), np.stack(_second_half)
            )
        )
    # end for split-half resample
    _half_reliability = np.nanmean(_half_correlations, axis=0)
    _full_reliability = np.clip(
        2 * _half_reliability / (1 + _half_reliability), 0.0, 1.0
    )
    noise_ceiling_stim_r = np.sqrt(_full_reliability)

    # Response window of the 0-200 ms cache: MUA drive begins ~50 ms.
    _response = slice(int(round(50 / (1000 / cfg.target_fs))), _n_time)
    _predict_mean_mse = float(
        np.mean(
            (_fit_targets.mean(axis=0)[None] - _test_image_targets) ** 2
        )
    )
    _ridge_mse = float(
        np.mean((ridge_test_predictions - _test_image_targets) ** 2)
    )
    ridge_summary = {
        "linear map": f"RidgeCV, alpha = {ridge_model.alpha_:g}",
        "validation MSE (standardized)": round(ridge_validation_mse, 5),
        "test MSE (standardized)": round(_ridge_mse, 5),
        "variance explained vs. mean": round(
            1 - _ridge_mse / _predict_mean_mse, 4
        ),
        "mean stim_r (>=50 ms)": round(
            float(np.nanmean(ridge_channel_time_correlations[_response])), 4
        ),
        "fraction of noise ceiling": round(
            float(
                np.nanmean(
                    ridge_channel_time_correlations[_response]
                    / np.clip(noise_ceiling_stim_r[_response], 1e-3, None)
                )
            ),
            3,
        ),
    }

    mo.vstack(
        [
            mo.md(
                "## Ridge-regression reference\n\n"
                "One ridge map fit on the same "
                f"**{len(_fit_indices):,}** fit presentations, from the "
                "concatenated 4 x 1024 DINO features to the flattened "
                f"[{_n_time} time x {_n_channels} MUA site] response, then scored "
                "on the 100 repeated test images with the identical per-"
                "(time, channel) stimulus correlation used for the decoder below "
                "-- so the two are directly comparable in the combined figure."
            ),
            mo.ui.table([ridge_summary], selection=None, pagination=False),
        ]
    )
    return (
        noise_ceiling_stim_r,
        ridge_median_test_correlation,
        ridge_validation_mse,
    )


@app.cell
def _(
    BaselineModel,
    ann,
    cfg,
    mo,
    test_dataset,
    torch,
    train_features,
    training_dataset,
    validation_dataset,
):
    torch.manual_seed(cfg.random_seed)
    model = BaselineModel(
        ann,
        layers=cfg.layer_names,
        temporal_embedding_dim=cfg.temporal_embedding_dim,
        value_dim=cfg.value_dim,
        n_timepoints=test_dataset.targets.shape[1],
        temporal_compression_ratio=1,
        n_neurons=test_dataset.targets.shape[2],
        mlp_hidden_dim=cfg.mlp_hidden_dim,
        dropout=cfg.dropout,
        attention_granularity=cfg.attention_granularity,
    ).to(ann.get_device())

    _trainable_parameters = sum(
        _parameter.numel()
        for _parameter in model.parameters()
        if _parameter.requires_grad
    )
    mo.md(
        f"""
        ## Aligned learning problem

        - fit: **{len(training_dataset):,}** unique-image presentations
        - validation: **{len(validation_dataset):,}** disjoint unique images
        - final test: **{len(test_dataset):,}** trials = 100 images × 30 repeats
        - input: `{train_features.shape[1:]}` = [DINO layers, embedding]
        - target: `{test_dataset.targets.shape[1:]}` = [time, {cfg.area} MUA sites]
        - trainable decoder parameters: **{_trainable_parameters:,}**

        DINO is frozen. `BaselineModel` learns one temporal query per target bin,
        attends over the selected static DINO representations, and applies an
        independent readout at every time bin. It contains no recurrence and does
        not mix neural time bins.
        """
    )
    return (model,)


@app.cell
def _(
    asdict,
    cfg,
    channel_mean,
    channel_scale,
    checkpoint_path,
    mo,
    model,
    neural_activity_timebin_mse_loss,
    np,
    run_pipeline_button,
    test_step,
    torch,
    training_loader,
    training_step,
    validation_loader,
):
    if not run_pipeline_button.value:
        mo.stop(
            True,
            mo.md(
                "Click **Load DINO features + fit ridge and BaselineModel** "
                "to start optimization."
            ),
        )
    # end if model training was not explicitly requested

    _optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    _optimizer.zero_grad(set_to_none=True)
    _cost_function = neural_activity_timebin_mse_loss
    initial_validation_loss = test_step(
        model,
        validation_loader,
        _cost_function,
        use_precomputed_features=True,
        device=model.device,
    )
    training_losses = []
    validation_losses = []
    best_validation_loss = initial_validation_loss
    best_epoch = 0
    _best_state = {
        _name: _value.detach().cpu().clone()
        for _name, _value in model.state_dict().items()
    }

    for _epoch in range(1, cfg.epochs + 1):
        _training_loss = training_step(
            model,
            training_loader,
            _optimizer,
            _cost_function,
            use_precomputed_features=True,
            device=model.device,
        )
        _validation_loss = test_step(
            model,
            validation_loader,
            _cost_function,
            use_precomputed_features=True,
            device=model.device,
        )
        training_losses.append(_training_loss)
        validation_losses.append(_validation_loss)
        if _validation_loss < best_validation_loss:
            best_validation_loss = _validation_loss
            best_epoch = _epoch
            _best_state = {
                _name: _value.detach().cpu().clone()
                for _name, _value in model.state_dict().items()
            }
        # end if this epoch is the best validation checkpoint
        print(
            f"epoch {_epoch:03d}/{cfg.epochs:03d} | "
            f"train {_training_loss:.6f} | validation {_validation_loss:.6f}"
        )
    # end for optimization epoch

    model.load_state_dict(_best_state)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": _best_state,
            "cfg": asdict(cfg),
            "channel_mean": channel_mean,
            "channel_scale": channel_scale,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
        },
        checkpoint_path,
    )
    all_validation_losses = np.asarray(
        [initial_validation_loss, *validation_losses]
    )
    mo.md(
        f"Best validation MSE: **{best_validation_loss:.6f}** at epoch "
        f"**{best_epoch}**. Checkpoint: `{checkpoint_path}`."
    )
    return (
        all_validation_losses,
        best_epoch,
        training_losses,
        validation_losses,
    )


@app.cell
def _(
    aggregate_attention_by_layer,
    all_validation_losses,
    allmat,
    best_epoch,
    cfg,
    mo,
    model,
    noise_ceiling_stim_r,
    np,
    plt,
    ridge_median_test_correlation,
    ridge_validation_mse,
    test_loader,
    test_trial_indices,
    torch,
    training_losses,
    validation_losses,
):
    # Collect final predictions and attention without updating the selected model.
    _prediction_batches = []
    _target_batches = []
    _attention_batches = []
    model.eval()
    with torch.no_grad():
        for _inputs, _targets in test_loader:
            _predictions, _attention = model(
                _inputs.to(model.device),
                use_precomputed_features=True,
            )
            _prediction_batches.append(_predictions.cpu().numpy())
            _target_batches.append(_targets.numpy())
            _attention_batches.append(
                aggregate_attention_by_layer(_attention).cpu()
            )
        # end for final-test batch
    # end with final-test inference

    _trial_predictions = np.concatenate(_prediction_batches, axis=0)
    _trial_targets = np.concatenate(_target_batches, axis=0)
    _test_image_ids = allmat[test_trial_indices, 2] - 1
    _image_predictions = np.stack(
        [
            _trial_predictions[_test_image_ids == _image_id].mean(axis=0)
            for _image_id in range(100)
        ]
    )
    _image_targets = np.stack(
        [
            _trial_targets[_test_image_ids == _image_id].mean(axis=0)
            for _image_id in range(100)
        ]
    )

    # Correlate the 100 image means independently at each time × MUA site.
    _centered_predictions = _image_predictions - _image_predictions.mean(
        axis=0, keepdims=True
    )
    _centered_targets = _image_targets - _image_targets.mean(
        axis=0, keepdims=True
    )
    _correlation_numerator = (
        _centered_predictions * _centered_targets
    ).sum(axis=0)
    _correlation_denominator = np.sqrt(
        np.square(_centered_predictions).sum(axis=0)
        * np.square(_centered_targets).sum(axis=0)
    )
    channel_time_correlations = np.divide(
        _correlation_numerator,
        _correlation_denominator,
        out=np.full_like(_correlation_numerator, np.nan),
        where=_correlation_denominator > 0,
    )
    median_test_correlation = np.nanmedian(
        channel_time_correlations,
        axis=1,
    )
    mean_test_attention = torch.cat(_attention_batches, dim=0).mean(dim=0)

    _response_time_ms = np.arange(
        cfg.time_start_ms,
        cfg.time_end_ms,
        1000 / cfg.target_fs,
    )
    _figure, _axes = plt.subplots(1, 3, figsize=(17, 4.5))
    _axes[0].plot(
        np.arange(1, len(training_losses) + 1),
        training_losses,
        label="Fit",
    )
    _axes[0].plot(
        np.arange(len(validation_losses) + 1),
        all_validation_losses,
        label="Validation",
    )
    _axes[0].scatter(
        best_epoch,
        all_validation_losses[best_epoch],
        marker="D",
        s=60,
        zorder=4,
        label=(
            "Best BaselineModel MSE "
            f"({all_validation_losses[best_epoch]:.4f})"
        ),
    )
    _axes[0].scatter(
        len(validation_losses),
        ridge_validation_mse,
        marker="*",
        s=180,
        color="black",
        zorder=4,
        label=f"Ridge validation MSE ({ridge_validation_mse:.4f})",
    )
    _axes[0].axvline(best_epoch, color="black", linestyle="--", alpha=0.5)
    _axes[0].set(
        xlabel="BaselineModel epoch",
        ylabel="Validation MSE",
        title="Validation comparison",
    )
    _axes[0].legend()
    _axes[0].grid(alpha=0.25)

    _axes[1].plot(
        _response_time_ms,
        median_test_correlation,
        linewidth=2,
        label="BaselineModel",
    )
    _axes[1].plot(
        _response_time_ms,
        ridge_median_test_correlation,
        linewidth=2,
        label="Ridge",
    )
    _axes[1].plot(
        _response_time_ms,
        np.nanmedian(noise_ceiling_stim_r, axis=1),
        color="black",
        linestyle=":",
        label="Noise ceiling",
    )
    _axes[1].axhline(0, color="black", linewidth=1)
    _axes[1].set(
        xlabel="Time from image onset (ms)",
        ylabel="Median Pearson r across channels",
        title="100 repeated test-image means",
    )
    _axes[1].legend()
    _axes[1].grid(alpha=0.25)

    _attention_image = _axes[2].imshow(
        mean_test_attention.T.numpy(),
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[
            _response_time_ms[0],
            _response_time_ms[-1] + 1000 / cfg.target_fs,
            -0.5,
            len(cfg.layer_names) - 0.5,
        ],
    )
    _axes[2].set(
        xlabel="Time from image onset (ms)",
        ylabel="DINO layer",
        title="Mean final-test attention",
        yticks=np.arange(len(cfg.layer_names)),
        yticklabels=cfg.layer_names,
    )
    _figure.colorbar(_attention_image, ax=_axes[2], label="Layer weight")
    _figure.tight_layout()
    mo.vstack(
        [
            mo.md(
                "## Ridge versus BaselineModel\n\n"
                "Targets are averaged over 30 repetitions before correlation; "
                "this raises their SNR and matches the intended role of the TVSD "
                "test pool. Both methods use the same split, standardized targets, "
                "DINO layers, and test metric. The displayed median treats every "
                "MUA site equally."
            ),
            _figure,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
