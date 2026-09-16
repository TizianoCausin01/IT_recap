import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import os
    import sys
    from dataclasses import asdict, dataclass, field, replace
    from pathlib import Path

    import matplotlib.pyplot as plt
    import marimo as mo
    import numpy as np
    import torch
    import yaml

    return (
        Path,
        asdict,
        dataclass,
        field,
        json,
        mo,
        np,
        os,
        plt,
        replace,
        sys,
        torch,
        yaml,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Noise on the temporal embedding — monkey F TVSD

    `BaselineModel` learns one **temporal embedding** per target time bin. Those
    embeddings are the model's only representation of time: they are projected
    into queries, the queries attend over the frozen DINOv3 layers, and a
    per-bin readout maps the attended features onto MUA. Nothing else in the
    decoder knows what "60 ms" means.

    That makes the embedding table an obvious place to regularize. With 20 bins
    at 100 Hz and 22,248 training presentations, each bin's query is free to
    memorize its own idiosyncratic direction, even though neighbouring 10 ms
    bins of MUA are nearly the same signal. `TemporalNoiseBaselineModel` adds
    Gaussian jitter to the table on every forward pass, independently per image:

    ```python
    temporal_embeddings = self.temporal_embeddings.unsqueeze(0).expand(B, -1, -1)
    if self.training or self.noise_in_eval:
        temporal_embeddings = temporal_embeddings + self._sample_embedding_noise(
            temporal_embeddings
        )
    queries = self.query_projection(temporal_embeddings)
    ```

    No bin can then rely on an exact query direction, so the learned
    layer-attention schedule has to stay usable under perturbation — which
    should smooth it across neighbouring bins. The frozen visual features are
    untouched, so this isolates regularization of the **temporal code itself**.

    This notebook sweeps the jitter scale, from a noiseless anchor upwards, and
    scores every level with the same protocol as
    `train_tvsd_baseline_marimo.py`: the seeded split of the 22,248 unique-image
    presentations, train-only per-channel standardization, and the untouched 100
    repeated test images.

    Two conventions carried over from earlier TVSD experiments:

    - **Selection is on validation `stim_r`, not validation MSE.** MSE rewards
      shrinkage — a decoder that predicts closer to the mean wins on MSE while
      losing image selectivity — so it ranks these decoders wrong.
    - **The jitter is relative by default.** The embedding table starts at
      std 0.02 and grows during training, so a fixed absolute std would be
      crippling early and negligible later; `relative_temporal_noise` reads
      `temporal_noise_std` as a fraction of the table's current spread.

    Each sweep level is written to disk as soon as it finishes, so an
    interrupted run resumes instead of restarting.
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

    from IT_recap.neural_prediction_training import split_half_reliability
    from IT_recap.tvsd import preprocess_tvsd_mua_targets
    from IT_recap.tvsd_experiments import (
        N_TEST_IMAGES,
        build_datasets,
        build_loaders,
        build_variant_model,
        fit_ridge_reference,
        load_cached_data,
        predict_test_trials,
        resolve_cache_paths,
        resolve_device,
        score_and_decompose,
        standardize_targets,
        train_baseline_variant,
    )
    from useful_stuff.image_processing.computational_models import imgANN

    @dataclass
    class Cfg:
        # Caches produced by train_tvsd_baseline_marimo.py; nothing is recomputed.
        mua_file_name: str = "f_THINGS_MUA_trials.mat"
        feature_archive_name: str = "tvsd_monkeyF_dino_v3_l_224_features.npz"
        run_dir_name: str = "tvsd_temporal_embedding_noise"

        # Neural target window, identical to the notebook's cache name.
        area: str = "IT"
        target_fs: int = 100
        # The target cache's file name encodes this window, so changing it names
        # a different file rather than slicing the existing one. The preparation
        # cell below builds whichever window is selected, straight from the MUA.
        # 0-200 ms is the window every earlier TVSD result used.
        time_start_ms: float = 0.0
        time_end_ms: float = 200.0
        # Bins before this are excluded from every reported score. Cropping the
        # cache here instead is a different experiment: it also removes those
        # bins from the decoder's output, so it stops spending temporal
        # embeddings and readout capacity on bins with no stimulus drive.
        response_onset_ms: float = 50.0
        baseline_start_ms: float = -100.0
        baseline_end_ms: float = 0.0
        preprocessing_chunk_size: int = 64

        # Frozen encoder that BaselineModel wraps even when cached features are used.
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
        trust_remote_code: bool = True
        attn_implementation: str = "sdpa"

        # Split and optimization, mirroring the baseline notebook.
        validation_fraction: float = 0.1
        random_seed: int = 0
        batch_size: int = 128
        num_workers: int = 0
        epochs: int = 30
        minimum_epochs: int = 10
        patience: int = 10
        learning_rate: float = 1e-3
        weight_decay: float = 1e-4
        selection_metric: str = "stim_r"
        temporal_embedding_dim: int = 128
        value_dim: int = 128
        mlp_hidden_dim: int = 64
        dropout: float = 0.5
        attention_granularity: str = "layer"

        # The knob under study. 0.0 is the noiseless anchor: without it a sweep
        # cannot say whether any jitter level actually helped.
        temporal_noise_std: float = 0.0
        noise_std_grid: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,)
        relative_temporal_noise: bool = True
        temporal_noise_in_eval: bool = False

        # Fields build_variant_model reads only on its noise_layer branch.
        match_noise_norm: bool = True
        noise_layer_in_eval: bool = True

        # Evaluation and bookkeeping.
        noise_ceiling_resamples: int = 40
        device: str = "auto"
        smoke_test: bool = False

    # EOC

    cfg = Cfg()
    sweep_root = PROJECT_ROOT / "results" / cfg.run_dir_name
    return (
        N_TEST_IMAGES,
        build_datasets,
        build_loaders,
        build_variant_model,
        cfg,
        fit_ridge_reference,
        imgANN,
        load_cached_data,
        paths,
        predict_test_trials,
        preprocess_tvsd_mua_targets,
        resolve_cache_paths,
        resolve_device,
        score_and_decompose,
        split_half_reliability,
        standardize_targets,
        sweep_root,
        train_baseline_variant,
    )


@app.cell
def _(cfg, mo):
    noise_levels_ui = mo.ui.multiselect(
        options=[f"{_std:g}" for _std in cfg.noise_std_grid],
        value=[f"{_std:g}" for _std in cfg.noise_std_grid],
        label="Jitter levels to train (0 = noiseless anchor)",
    )
    epochs_ui = mo.ui.slider(
        start=2, stop=300, step=1, value=cfg.epochs, label="Max epochs per level"
    )
    relative_noise_ui = mo.ui.checkbox(
        value=cfg.relative_temporal_noise,
        label="Scale the jitter by the embedding table's current spread",
    )
    noise_in_eval_ui = mo.ui.checkbox(
        value=cfg.temporal_noise_in_eval,
        label="Keep perturbing at evaluation time",
    )
    smoke_test_ui = mo.ui.checkbox(
        value=False, label="Smoke run (1,024 fit presentations, few epochs)"
    )
    time_start_ui = mo.ui.number(
        start=-100.0,
        stop=190.0,
        step=10.0,
        value=cfg.time_start_ms,
        label="Target-cache window start (ms)",
    )
    time_end_ui = mo.ui.number(
        start=10.0,
        stop=400.0,
        step=10.0,
        value=cfg.time_end_ms,
        label="Target-cache window end (ms)",
    )
    prepare_cache_button = mo.ui.run_button(
        label="Prepare the target cache for this window"
    )
    run_sweep_button = mo.ui.run_button(label="Run the temporal-noise sweep")

    mo.vstack(
        [
            mo.md("## Sweep controls"),
            noise_levels_ui,
            epochs_ui,
            relative_noise_ui,
            noise_in_eval_ui,
            smoke_test_ui,
            time_start_ui,
            time_end_ui,
            prepare_cache_button,
            run_sweep_button,
        ]
    )
    return (
        epochs_ui,
        noise_in_eval_ui,
        noise_levels_ui,
        prepare_cache_button,
        relative_noise_ui,
        run_sweep_button,
        smoke_test_ui,
        time_end_ui,
        time_start_ui,
    )


@app.cell
def _(
    cfg,
    epochs_ui,
    noise_in_eval_ui,
    noise_levels_ui,
    relative_noise_ui,
    replace,
    smoke_test_ui,
    sweep_root,
    time_end_ui,
    time_start_ui,
):
    # One configuration object drives the whole sweep; the UI only overrides it.
    run_cfg = replace(
        cfg,
        time_start_ms=float(time_start_ui.value),
        time_end_ms=float(time_end_ui.value),
        epochs=2 if smoke_test_ui.value else epochs_ui.value,
        minimum_epochs=1 if smoke_test_ui.value else cfg.minimum_epochs,
        patience=1 if smoke_test_ui.value else cfg.patience,
        relative_temporal_noise=relative_noise_ui.value,
        temporal_noise_in_eval=noise_in_eval_ui.value,
        noise_ceiling_resamples=4 if smoke_test_ui.value else cfg.noise_ceiling_resamples,
        smoke_test=smoke_test_ui.value,
    )
    requested_noise_levels = sorted(
        float(_label) for _label in noise_levels_ui.value
    )

    # Stored runs are only reusable within the data configuration that produced
    # them: a different response window changes the number of target bins, so
    # its arrays cannot be plotted on this window's time axis. Keeping each
    # window in its own directory makes a stale run impossible to pick up.
    run_dir = sweep_root / (
        f"{run_cfg.area}_{run_cfg.time_start_ms:g}-{run_cfg.time_end_ms:g}ms_"
        f"{run_cfg.target_fs}Hz"
    )
    return requested_noise_levels, run_cfg, run_dir


@app.cell
def _(
    mo,
    paths,
    prepare_cache_button,
    preprocess_tvsd_mua_targets,
    resolve_cache_paths,
    run_cfg,
):
    # The window selected above names the cache; it never slices an existing
    # one. Report exactly which files this configuration will read.
    cache_paths = resolve_cache_paths(run_cfg, paths)
    target_cache_path = cache_paths["targets"]
    _cache_rows = [
        {
            "resource": _label,
            "path": str(_path),
            "available": _path.is_file(),
        }
        for _label, _path in (
            (f"{run_cfg.area} target cache", target_cache_path),
            ("DINO feature archive", cache_paths["features"]),
            ("MUA source", cache_paths["mua"]),
        )
    ]
    _status = mo.vstack(
        [
            mo.md(
                f"""
                ## Target cache

                The cache name encodes the response window, so
                **{run_cfg.time_start_ms:g}–{run_cfg.time_end_ms:g} ms** at
                {run_cfg.target_fs} Hz is a different file from any other
                window. Preparing one is a single pass over the ~58 GiB MUA
                file and takes on the order of twenty minutes; it is therefore
                behind its own button, not the sweep button.

                Cropping the cache is not the same as
                `response_onset_ms = {run_cfg.response_onset_ms:g}`, which only
                excludes early bins from the reported scores. A cache that
                starts at the response onset also removes those bins from the
                decoder's **output**, so it stops spending temporal embeddings
                and readout capacity on bins with no stimulus drive — a real
                experiment, but one whose numbers are not directly comparable
                to the 0–200 ms results.
                """
            ),
            mo.ui.table(_cache_rows, selection=None, pagination=False),
        ]
    )

    if not target_cache_path.is_file():
        if not prepare_cache_button.value:
            mo.stop(
                True,
                mo.vstack(
                    [
                        _status,
                        mo.md(
                            "**This window has no cache yet.** Click "
                            "**Prepare the target cache for this window** to "
                            "build it, or set the window back to one of the "
                            "caches listed as available."
                        ),
                    ]
                ),
            )
        # end if preparation was not requested
        if not cache_paths["mua"].is_file():
            raise FileNotFoundError(
                f"Cannot build a cache without the MUA source at "
                f"{cache_paths['mua']}."
            )
        # end if the neural source file is absent
        print(f"preparing {target_cache_path}")
        preprocess_tvsd_mua_targets(
            mat_path=cache_paths["mua"],
            output_path=target_cache_path,
            area=run_cfg.area,
            time_start_ms=run_cfg.time_start_ms,
            time_end_ms=run_cfg.time_end_ms,
            target_fs=run_cfg.target_fs,
            baseline_start_ms=run_cfg.baseline_start_ms,
            baseline_end_ms=run_cfg.baseline_end_ms,
            chunk_size=run_cfg.preprocessing_chunk_size,
        )
    # end if this window still needs a cache

    _status
    return (target_cache_path,)


@app.cell
def _(
    N_TEST_IMAGES,
    build_datasets,
    build_loaders,
    load_cached_data,
    mo,
    np,
    paths,
    resolve_device,
    run_cfg,
    run_sweep_button,
    split_half_reliability,
    standardize_targets,
    target_cache_path,
):
    mo.stop(
        not run_sweep_button.value,
        mo.md(
            "## Data\n\nClick **Run the temporal-noise sweep** to load "
            f"`{target_cache_path.name}` and the DINO feature archive. The "
            "features themselves are never recomputed here: extract them once "
            "with `train_tvsd_baseline_marimo.py`."
        ),
    )

    device = resolve_device(run_cfg.device)
    targets, train_features, test_features, allmat = load_cached_data(
        run_cfg, paths
    )
    datasets, indices, (channel_mean, channel_scale) = build_datasets(
        run_cfg, targets, train_features, test_features, allmat
    )
    loaders = build_loaders(run_cfg, datasets)
    n_timepoints, n_neurons = targets.shape[1], targets.shape[2]

    # Materialize the standardized targets once; the ridge fit and every score
    # below reuse them.
    fit_targets = standardize_targets(
        targets[indices["train"]], channel_mean, channel_scale
    )
    validation_targets = standardize_targets(
        targets[indices["validation"]], channel_mean, channel_scale
    )
    test_trial_targets = standardize_targets(
        targets[indices["test"]], channel_mean, channel_scale
    )

    # MUA drive begins around 50 ms; earlier bins carry no stimulus signal and
    # would dilute every correlation summary equally.
    scoring = {
        "test_image_ids": allmat[indices["test"], 2] - 1,
        "fit_target_mean": fit_targets.mean(axis=0),
        # Offset from the cache's own start, not from zero, so the window
        # stays correct if the target cache is ever cut at a later onset.
        "response_slice": slice(
            int(
                round(
                    (run_cfg.response_onset_ms - run_cfg.time_start_ms)
                    / (1000 / run_cfg.target_fs)
                )
            ),
            n_timepoints,
        ),
    }
    scoring["ceiling"] = split_half_reliability(
        test_trial_targets,
        scoring["test_image_ids"],
        N_TEST_IMAGES,
        reducer="mean",
        n_resamples=run_cfg.noise_ceiling_resamples,
        seed=run_cfg.random_seed,
    )

    response_time_ms = np.arange(
        run_cfg.time_start_ms, run_cfg.time_end_ms, 1000 / run_cfg.target_fs
    )
    mo.md(
        f"""
        ## Data

        - device: **{device}**
        - fit: **{len(indices['train']):,}** unique-image presentations
        - validation: **{len(indices['validation']):,}** disjoint unique images
        - final test: **{len(indices['test']):,}** trials = 100 images × 30 repeats
        - target: `[{n_timepoints} time bins × {n_neurons} {run_cfg.area} MUA sites]`
        - mean reliability of the repetition-averaged target over the response
          window: **{float(np.nanmean(scoring['ceiling'][scoring['response_slice']])):.3f}**
        """
    )
    return (
        allmat,
        device,
        fit_targets,
        indices,
        loaders,
        n_neurons,
        n_timepoints,
        response_time_ms,
        scoring,
        test_features,
        test_trial_targets,
        train_features,
        validation_targets,
    )


@app.cell
def _(
    allmat,
    fit_ridge_reference,
    fit_targets,
    indices,
    mo,
    run_cfg,
    score_and_decompose,
    scoring,
    test_features,
    test_trial_targets,
    train_features,
    validation_targets,
):
    # The same RidgeCV map the baseline notebook reports, refit here so the
    # sweep has a reference that never saw a different split.
    _ridge_predictions, ridge_validation_mse, _ridge_alpha, _ridge_parameters = (
        fit_ridge_reference(
            run_cfg,
            train_features,
            test_features,
            allmat,
            indices,
            fit_targets,
            validation_targets,
        )
    )
    ridge_row, ridge_correlations = score_and_decompose(
        _ridge_predictions, test_trial_targets, scoring, "RidgeCV"
    )
    ridge_row["ridge_alpha"] = _ridge_alpha
    ridge_row["validation_mse"] = round(ridge_validation_mse, 5)

    mo.vstack(
        [
            mo.md(
                "## Ridge reference\n\nOne linear map from the concatenated 4 × "
                "1024 DINO features to the flattened response, fit on the same "
                "presentations and scored with the same metric as every decoder "
                "below."
            ),
            mo.ui.table([ridge_row], selection=None, pagination=False),
        ]
    )
    return ridge_correlations, ridge_row, ridge_validation_mse


@app.cell
def _(cfg, imgANN, mo, run_sweep_button, torch):
    mo.stop(not run_sweep_button.value, mo.md(""))

    # BaselineModel wraps the frozen backbone even though cached activations
    # bypass its forward pass during decoder training.
    encoder = imgANN(
        model_name=cfg.model_name,
        pkg="hf",
        img_size=cfg.img_size,
        pooling=cfg.pooling,
        dtype=torch.float32,
        attn_implementation=cfg.attn_implementation,
        repo_url=cfg.model_source,
        trust_remote_code=cfg.trust_remote_code,
    )
    return (encoder,)


@app.cell
def _(
    asdict,
    build_variant_model,
    device,
    encoder,
    json,
    loaders,
    mo,
    n_neurons,
    n_timepoints,
    np,
    predict_test_trials,
    replace,
    requested_noise_levels,
    run_cfg,
    run_dir,
    score_and_decompose,
    scoring,
    torch,
    train_baseline_variant,
):
    run_dir.mkdir(parents=True, exist_ok=True)

    sweep_rows = []
    sweep_histories = {}
    sweep_correlations = {}
    sweep_embeddings = {}
    sweep_attentions = {}
    for _noise_std in requested_noise_levels:
        _run_id = f"std_{_noise_std:g}"
        _record_path = run_dir / f"{_run_id}.json"
        _array_path = run_dir / f"{_run_id}_arrays.npz"

        # A finished level is reloaded rather than retrained. Long MPS runs get
        # killed for memory, and a six-level sweep is far past that limit.
        if _record_path.is_file() and _array_path.is_file():
            with open(_record_path, "r") as _record_file:
                _record = json.load(_record_file)
            # end with stored run record
            with np.load(_array_path) as _arrays:
                _stored = {_key: _arrays[_key] for _key in _arrays.files}
            # end with stored run arrays

            # Last line of defence against a stale run: the directory already
            # separates response windows, but a record written before the
            # target cache was rebuilt would still have the wrong bin count,
            # and it would only surface as an unreadable figure.
            if _stored["site_stim_r"].shape[0] != n_timepoints:
                raise ValueError(
                    f"{_record_path} holds "
                    f"{_stored['site_stim_r'].shape[0]} time bins but the "
                    f"current targets have {n_timepoints}. Delete this run "
                    "directory and retrain the level."
                )
            # end if the stored run does not match the current targets
            sweep_correlations[_noise_std] = _stored["site_stim_r"]
            sweep_embeddings[_noise_std] = _stored["temporal_embeddings"]
            sweep_attentions[_noise_std] = _stored["mean_attention"]
            sweep_rows.append(_record["row"])
            sweep_histories[_noise_std] = _record["history"]
            print(f"reusing {_run_id}")
            continue
        # end if this level already finished

        print(f"training {_run_id}")
        # Level 0 is the plain decoder, not a zero-variance draw: keeping it a
        # real BaselineModel means the anchor consumes no extra randomness and
        # stays comparable to every earlier TVSD result.
        _level_cfg = replace(run_cfg, temporal_noise_std=_noise_std)
        _variant = "baseline" if _noise_std == 0.0 else "temporal_noise"
        torch.manual_seed(run_cfg.random_seed)
        _model = build_variant_model(
            _variant, encoder, _level_cfg, n_timepoints, n_neurons
        ).to(device)
        _history, _best_epoch, _validation_mse, _validation_stim_r = (
            train_baseline_variant(_model, loaders, _level_cfg, device)
        )
        _trial_predictions, _trial_targets, _mean_attention = predict_test_trials(
            _model, loaders["test"], device
        )
        _row, _site_correlations = score_and_decompose(
            _trial_predictions, _trial_targets, scoring, f"jitter {_noise_std:g}"
        )
        _row.update(
            {
                "temporal_noise_std": _noise_std,
                "relative_noise": run_cfg.relative_temporal_noise,
                "noise_in_eval": run_cfg.temporal_noise_in_eval,
                "best_epoch": _best_epoch,
                "validation_mse": round(_validation_mse, 5),
                "validation_stim_r": round(_validation_stim_r, 4),
            }
        )
        _temporal_embeddings = (
            _model.temporal_embeddings.detach().cpu().numpy().copy()
        )

        with open(_record_path, "w") as _record_file:
            json.dump(
                {
                    "row": _row,
                    "history": _history,
                    "cfg": asdict(_level_cfg),
                },
                _record_file,
                indent=2,
            )
        # end with saved run record
        np.savez_compressed(
            _array_path,
            site_stim_r=_site_correlations,
            temporal_embeddings=_temporal_embeddings,
            mean_attention=_mean_attention,
        )
        sweep_rows.append(_row)
        sweep_histories[_noise_std] = _history
        sweep_correlations[_noise_std] = _site_correlations
        sweep_embeddings[_noise_std] = _temporal_embeddings
        sweep_attentions[_noise_std] = _mean_attention
    # end for jitter level

    mo.md(
        f"Sweep complete: **{len(sweep_rows)}** levels at "
        f"{run_cfg.time_start_ms:g}–{run_cfg.time_end_ms:g} ms, in `{run_dir}`. "
        "Finished levels are reloaded rather than retrained; delete a "
        "`std_*.json` to force one to train again."
    )
    return (
        sweep_attentions,
        sweep_correlations,
        sweep_embeddings,
        sweep_histories,
        sweep_rows,
    )


@app.cell
def _(mo, ridge_row, sweep_rows):
    _display_columns = (
        "temporal_noise_std",
        "best_epoch",
        "validation_mse",
        "validation_stim_r",
        "mean_stim_r_response",
        "median_stim_r_response",
        "fraction_of_ceiling",
        "test_mse",
        "mse_after_rescaling",
        "mean_optimal_scale",
    )
    _table_rows = [
        {
            "model": _row["model"],
            **{
                _column: _row.get(_column) for _column in _display_columns
            },
        }
        for _row in [*sweep_rows, ridge_row]
    ]

    # The anchor is the only fair comparison: it is the same decoder without
    # the jitter, trained on the same split with the same seed.
    if not sweep_rows:
        mo.stop(True, mo.md("## Results\n\nSelect at least one jitter level."))
    # end if the sweep trained nothing

    anchor_row = next(
        (_row for _row in sweep_rows if _row["temporal_noise_std"] == 0.0), None
    )
    best_row = max(sweep_rows, key=lambda _row: _row["validation_stim_r"])
    _verdict = "the sweep needs the noiseless anchor to be interpretable."
    if anchor_row is not None:
        _delta = (
            best_row["mean_stim_r_response"] - anchor_row["mean_stim_r_response"]
        )
        _verdict = (
            f"selected jitter **{best_row['temporal_noise_std']:g}** on "
            f"validation stim_r; its test stim_r is **{_delta:+.4f}** against "
            f"the noiseless anchor "
            f"({anchor_row['mean_stim_r_response']:.4f} → "
            f"{best_row['mean_stim_r_response']:.4f}), i.e. "
            f"**{best_row['fraction_of_ceiling']:.3f}** of the noise ceiling "
            f"against the anchor's **{anchor_row['fraction_of_ceiling']:.3f}**."
        )
    # end if the anchor was part of this sweep

    mo.vstack(
        [
            mo.md(
                "## Results\n\n"
                "`stim_r` is the Pearson correlation across the 100 test images, "
                "computed independently per (time bin, MUA site) and averaged "
                "over the response window from 50 ms. `mean_optimal_scale` "
                "below 1 means the predictions are over-dispersed, which is "
                "what makes MSE and stim_r disagree.\n\n"
                f"Verdict: {_verdict}"
            ),
            mo.ui.table(_table_rows, selection=None, pagination=False),
        ]
    )
    return anchor_row, best_row


@app.cell
def _(mo, np, plt, ridge_validation_mse, sweep_histories, sweep_rows):
    # Jitter scale is an ordered variable, so it gets one hue light-to-dark.
    _levels = sorted(sweep_histories)
    _level_shades = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(_levels), 1)))
    _shade_of = {_std: _level_shades[_index] for _index, _std in enumerate(_levels)}
    _best_epoch_of = {
        _row["temporal_noise_std"]: _row["best_epoch"] for _row in sweep_rows
    }

    _figure, _axes = plt.subplots(1, 3, figsize=(17, 4.5))

    # Panels 1 and 2: the optimization itself, one curve per jitter level. Fit
    # and validation MSE are drawn separately rather than together, because six
    # levels x two curves in one axis is unreadable.
    for _panel_index, _curve_name in enumerate(("train_mse", "validation_mse")):
        _axis = _axes[_panel_index]
        for _std in _levels:
            _history = sweep_histories[_std]
            _axis.plot(
                [_entry["epoch"] for _entry in _history],
                [_entry[_curve_name] for _entry in _history],
                linewidth=2,
                color=_shade_of[_std],
                label=f"jitter {_std:g}",
            )
        # end for jitter level
        _axis.set(
            xlabel="Epoch",
            ylabel="MSE (standardized targets)",
            title="Fit MSE" if _curve_name == "train_mse" else "Validation MSE",
        )
        _axis.legend(fontsize=8)
        _axis.grid(alpha=0.25)
    # end for loss curve

    # The ridge reference belongs on the validation panel only: it is a single
    # closed-form fit, so it has no epoch axis of its own.
    _axes[1].axhline(
        ridge_validation_mse,
        color="#2a78d6",
        linestyle="--",
        linewidth=1.5,
        label=f"RidgeCV ({ridge_validation_mse:.4f})",
    )
    _axes[1].legend(fontsize=8)

    # Panel 3: the criterion checkpoints are actually selected on. Diamonds mark
    # the restored epoch, which is where MSE and stim_r visibly disagree.
    for _std in _levels:
        _history = sweep_histories[_std]
        _epochs = [_entry["epoch"] for _entry in _history]
        _stim_r = [_entry["validation_stim_r"] for _entry in _history]
        _axes[2].plot(
            _epochs,
            _stim_r,
            linewidth=2,
            color=_shade_of[_std],
            label=f"jitter {_std:g}",
        )
        _best_epoch = _best_epoch_of.get(_std)
        if _best_epoch:
            _axes[2].scatter(
                _best_epoch,
                _stim_r[_best_epoch - 1],
                marker="D",
                s=45,
                zorder=4,
                color=_shade_of[_std],
                edgecolor="black",
                linewidth=0.6,
            )
        # end if this level has a restored checkpoint
    # end for jitter level
    _axes[2].set(
        xlabel="Epoch",
        ylabel="Validation stim_r",
        title="Selection criterion (diamond = restored epoch)",
    )
    _axes[2].legend(fontsize=8)
    _axes[2].grid(alpha=0.25)

    _figure.tight_layout()
    mo.vstack(
        [
            mo.md(
                "## Learning curves\n\n"
                "Fit and validation MSE next to the criterion the checkpoints "
                "are actually chosen on. Watch for the two disagreeing: "
                "validation MSE keeps falling while validation stim_r turns "
                "over, because MSE is minimized by shrinking predictions "
                "toward the mean and stim_r is not. That is the whole reason "
                "selection here is on stim_r."
            ),
            _figure,
        ]
    )
    return


@app.cell
def _(
    anchor_row,
    best_row,
    mo,
    np,
    plt,
    response_time_ms,
    ridge_correlations,
    run_cfg,
    scoring,
    sweep_correlations,
    sweep_embeddings,
    sweep_rows,
):
    # Jitter scale is an ordered variable, so it gets one hue light-to-dark.
    # Ridge and the anchor are separate entities, drawn in neutral dashes.
    _levels = sorted(sweep_correlations)
    _level_shades = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(_levels), 1)))
    _shade_of = {_std: _level_shades[_i] for _i, _std in enumerate(_levels)}

    # The per-epoch curves live in the learning-curve figure above; these three
    # panels are about the trained models rather than the optimization.
    _figure, _axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: the headline dose-response curve.
    _std_values = [_row["temporal_noise_std"] for _row in sweep_rows]
    _order = np.argsort(_std_values)
    _sorted_stds = np.asarray(_std_values)[_order]
    _axes[0].plot(
        _sorted_stds,
        np.asarray([_row["mean_stim_r_response"] for _row in sweep_rows])[_order],
        marker="o",
        linewidth=2,
        color="#4a3aa7",
        label="Test stim_r",
    )
    _axes[0].plot(
        _sorted_stds,
        np.asarray([_row["validation_stim_r"] for _row in sweep_rows])[_order],
        marker="s",
        linewidth=2,
        linestyle="-.",
        color="#1baf7a",
        label="Validation stim_r",
    )
    if anchor_row is not None:
        _axes[0].axhline(
            anchor_row["mean_stim_r_response"],
            color="gray",
            linestyle="--",
            linewidth=1.5,
            label=f"Noiseless anchor ({anchor_row['mean_stim_r_response']:.4f})",
        )
    # end if the anchor was trained in this sweep
    _ridge_stim_r = float(
        np.nanmean(ridge_correlations[scoring["response_slice"]])
    )
    _axes[0].axhline(
        _ridge_stim_r,
        color="#2a78d6",
        linestyle=":",
        linewidth=1.5,
        label=f"RidgeCV ({_ridge_stim_r:.4f})",
    )
    _axes[0].set(
        xlabel="Temporal-embedding jitter (fraction of embedding spread)"
        if run_cfg.relative_temporal_noise
        else "Temporal-embedding jitter (absolute std)",
        ylabel="Mean stim_r over the response window",
        title="Does jitter on the temporal code help?",
    )
    _axes[0].legend(fontsize=8)
    _axes[0].grid(alpha=0.25)

    # Panel 2: where in time any difference lives.
    for _std in _levels:
        _axes[1].plot(
            response_time_ms,
            np.nanmedian(sweep_correlations[_std], axis=1),
            linewidth=2,
            color=_shade_of[_std],
            label=f"jitter {_std:g}",
        )
    # end for jitter level
    _axes[1].plot(
        response_time_ms,
        np.nanmedian(ridge_correlations, axis=1),
        linewidth=2,
        linestyle=":",
        color="#2a78d6",
        label="RidgeCV",
    )
    _axes[1].plot(
        response_time_ms,
        np.nanmedian(scoring["ceiling"], axis=1),
        linewidth=2,
        linestyle=":",
        color="black",
        label="Noise ceiling",
    )
    _axes[1].axhline(0, color="black", linewidth=1)
    _axes[1].axvline(
        run_cfg.response_onset_ms, color="gray", linewidth=1, linestyle="--"
    )
    _axes[1].set(
        xlabel="Time from image onset (ms)",
        ylabel="Median stim_r across MUA sites",
        title="100 test images, mean over 30 repetitions",
    )
    _axes[1].legend(fontsize=8)
    _axes[1].grid(alpha=0.25)

    # Panel 3: the mechanism. If the jitter works as intended, the learned
    # temporal embeddings become more similar to their neighbours, i.e. the
    # cosine-similarity-versus-lag profile decays more slowly.
    for _std in _levels:
        _embeddings = sweep_embeddings[_std]
        _unit = _embeddings / (
            np.linalg.norm(_embeddings, axis=1, keepdims=True) + 1e-8
        )
        _similarity = _unit @ _unit.T
        _n_bins = _similarity.shape[0]
        _profile = [
            float(np.mean(np.diagonal(_similarity, offset=_lag)))
            for _lag in range(_n_bins)
        ]
        _axes[2].plot(
            np.arange(_n_bins) * (1000 / run_cfg.target_fs),
            _profile,
            linewidth=2,
            color=_shade_of[_std],
            label=f"jitter {_std:g}",
        )
    # end for jitter level
    _axes[2].axhline(0, color="black", linewidth=1)
    _axes[2].set(
        xlabel="Lag between time bins (ms)",
        ylabel="Mean cosine similarity of temporal embeddings",
        title="Did the jitter smooth the learned temporal code?",
    )
    _axes[2].legend(fontsize=8)
    _axes[2].grid(alpha=0.25)

    _figure.tight_layout()
    mo.vstack(
        [
            mo.md(
                "## How well does it work?\n\n"
                "Panel 1 is the answer to the question; panel 2 says whether "
                "the difference is a real time-resolved gain or a wash; panel "
                "3 says whether the mechanism did "
                "what it was supposed to do. A jitter level that improves "
                "stim_r **without** flattening the similarity profile is "
                "improving something other than the temporal code, and a level "
                "that flattens the profile without improving stim_r has simply "
                "destroyed the model's ability to distinguish time bins."
            ),
            _figure,
            mo.md(
                f"Best level by validation stim_r: "
                f"**{best_row['temporal_noise_std']:g}** "
                f"(test stim_r {best_row['mean_stim_r_response']:.4f}, "
                f"{best_row['fraction_of_ceiling']:.3f} of the ceiling)."
            ),
        ]
    )
    return


@app.cell
def _(
    anchor_row,
    best_row,
    mo,
    np,
    plt,
    response_time_ms,
    run_cfg,
    sweep_attentions,
):
    # The schedule the decoder learned: how much weight each temporal query put
    # on each frozen DINO layer, averaged over the 100 test images.
    _levels = sorted(sweep_attentions)
    _anchor_std = (
        anchor_row["temporal_noise_std"] if anchor_row is not None else _levels[0]
    )
    _best_std = best_row["temporal_noise_std"]
    _layer_names = run_cfg.layer_names
    _bin_width_ms = 1000 / run_cfg.target_fs

    _figure, _axes = plt.subplots(2, 2, figsize=(15, 9))
    _axes = _axes.ravel()

    # Panels 1 and 2: the same heatmap the baseline notebook draws, once for the
    # noiseless anchor and once for the selected level, on a shared colour scale
    # so the two are comparable by eye.
    _compared = [(_anchor_std, "noiseless anchor"), (_best_std, "selected level")]
    _shared_max = max(
        float(sweep_attentions[_std].max()) for _std, _ in _compared
    )
    _shared_min = min(
        float(sweep_attentions[_std].min()) for _std, _ in _compared
    )
    for _panel_index, (_std, _description) in enumerate(_compared):
        _attention = sweep_attentions[_std]
        _image = _axes[_panel_index].imshow(
            _attention.T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            vmin=_shared_min,
            vmax=_shared_max,
            extent=[
                response_time_ms[0],
                response_time_ms[-1] + _bin_width_ms,
                -0.5,
                _attention.shape[1] - 0.5,
            ],
        )
        _axes[_panel_index].set(
            xlabel="Time from image onset (ms)",
            ylabel="DINO layer",
            title=f"Mean test attention — jitter {_std:g} ({_description})",
            yticks=np.arange(len(_layer_names)),
            yticklabels=_layer_names,
        )
        _figure.colorbar(_image, ax=_axes[_panel_index], label="Layer weight")
    # end for compared jitter level

    # Panel 3: the same two schedules as line profiles, which is the only way to
    # see whether the jitter moved weight between layers or just rescaled it.
    # Layer depth is ordered, so it gets one hue light-to-dark.
    _layer_shades = plt.cm.Blues(np.linspace(0.35, 0.95, len(_layer_names)))
    for _layer_index, _layer_name in enumerate(_layer_names):
        _axes[2].plot(
            response_time_ms,
            sweep_attentions[_anchor_std][:, _layer_index],
            linewidth=2,
            color=_layer_shades[_layer_index],
            label=f"{_layer_name} (jitter {_anchor_std:g})",
        )
        _axes[2].plot(
            response_time_ms,
            sweep_attentions[_best_std][:, _layer_index],
            linewidth=2,
            linestyle="--",
            color=_layer_shades[_layer_index],
            label=f"{_layer_name} (jitter {_best_std:g})",
        )
    # end for hooked DINO layer
    _axes[2].axhline(
        1.0 / len(_layer_names),
        color="gray",
        linewidth=1,
        linestyle=":",
        label="Uniform over layers",
    )
    _axes[2].set(
        xlabel="Time from image onset (ms)",
        ylabel="Mean attention weight",
        title="Attention profile: anchor (solid) vs selected (dashed)",
    )
    _axes[2].legend(fontsize=7, ncol=2)
    _axes[2].grid(alpha=0.25)

    # Panel 4: how flat the schedule became. Entropy over layers is maximal when
    # a time bin spreads its weight evenly, so if the jitter works by stopping
    # bins from committing to one depth, this rises with the jitter scale.
    _entropy_by_level = []
    for _std in _levels:
        _weights = np.clip(sweep_attentions[_std], 1e-12, None)
        _weights = _weights / _weights.sum(axis=1, keepdims=True)
        _entropy_by_level.append(
            float(np.mean(-(_weights * np.log(_weights)).sum(axis=1)))
        )
    # end for jitter level
    _axes[3].plot(
        _levels, _entropy_by_level, marker="o", linewidth=2, color="#4a3aa7"
    )
    _axes[3].axhline(
        np.log(len(_layer_names)),
        color="gray",
        linestyle=":",
        linewidth=1.5,
        label="Uniform over layers (maximum)",
    )
    _axes[3].set(
        xlabel="Temporal-embedding jitter",
        ylabel="Mean attention entropy (nats)",
        title="Does jitter flatten the layer schedule?",
    )
    _axes[3].legend(fontsize=8)
    _axes[3].grid(alpha=0.25)

    _figure.tight_layout()
    mo.vstack(
        [
            mo.md(
                "## Temporal attention profiles\n\n"
                "`BaselineModel` has one query per time bin, so this is the "
                "decoder's learned answer to *which DINO depth does IT look "
                "like at this latency*. Read it against the stim_r panels: a "
                "jitter level that changes the schedule without changing "
                "stim_r has moved attention somewhere the readout did not "
                "care about, and one that flattens it all the way to the "
                "uniform line has stopped distinguishing depths at all."
            ),
            _figure,
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
