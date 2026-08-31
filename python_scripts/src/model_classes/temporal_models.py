import math

import torch
from torch import nn


class TemporalNaiveLayerAttention(nn.Module):
    """
    Extract frozen transformer features and weight their layers by time position.

    ``img_ann`` owns the frozen image backbone. Hooks capture the requested layers,
    and one independent projection per layer maps their features into a common value
    space. Learned time-position embeddings then define a global layer-attention
    schedule. There is no self-attention, recurrence, or temporal interaction.

    INPUT (forward):
        - x: torch.Tensor | dict -> image batch or keyword inputs for imgANN
        - time_idx: torch.Tensor | None -> position indices [time]
        - time_values: torch.Tensor | None -> ordered physical times [time]

    OUTPUT:
        - pred: torch.Tensor | None -> predictions [batch, time, output_dim]
        - latent: torch.Tensor -> latent representations [batch, time, latent_dim]
        - attention_weights: torch.Tensor | None -> weights [batch, time, layers]
    """

    """
    __init__
    Store the frozen imgANN, register feature hooks, and initialize the temporal
    attention, layer-specific projections, latent map, and optional output head.

    INPUT:
        - img_ann: imgANN -> frozen image-model wrapper used for feature extraction
        - layer_names: list[str] -> ordered transformer layers to attend over
        - n_time_bins: int -> number of discrete temporal positions
        - position_embedding_dim: int -> dimension of each learned position vector
        - layer_projection_dim: int -> common dimension for projected layer features
        - latent_dim: int -> dimension of the returned latent representation
        - output_dim: int | None -> neural prediction dimension; None omits the head

    OUTPUT:
        - None
    """
    def __init__(
        self,
        img_ann,
        layer_names: list[str],
        n_time_bins: int,
        position_embedding_dim: int = 128,
        layer_projection_dim: int = 128,
        latent_dim: int = 384,
        output_dim: int | None = None,
    ):
        # Initialize nn.Module so PyTorch registers trainable temporal parameters.
        super().__init__()

        # Copy the names because their order defines the attention-layer axis.
        self.layer_names = list(layer_names)
        dimensions = (
            len(self.layer_names),
            n_time_bins,
            position_embedding_dim,
            layer_projection_dim,
            latent_dim,
        )
        if min(dimensions) <= 0 or (output_dim is not None and output_dim <= 0):
            raise ValueError("All model dimensions and layer_names must be non-empty.")
        # end if any model dimension is invalid
        if len(self.layer_names) != len(set(self.layer_names)):
            raise ValueError("layer_names must not contain duplicates.")
        # end if layer names are duplicated

        # imgANN is intentionally not an nn.Module: its frozen backbone is excluded
        # from this model's optimizer and checkpoint state dictionary.
        self.img_ann = img_ann
        self.n_layers = len(self.layer_names)
        self.n_time_bins = n_time_bins
        self.layer_projection_dim = layer_projection_dim
        self.latent_dim = latent_dim
        self.feature_dim = None  # Inferred from hooked transformer features.

        # Validate the small subset of the imgANN interface used by this class.
        required_methods = ("create_forward_hook", "extract_features", "get_model")
        missing_methods = [
            method_name
            for method_name in required_methods
            if not callable(getattr(self.img_ann, method_name, None))
        ]
        if missing_methods:
            raise TypeError(f"img_ann is missing methods: {missing_methods}.")
        # end if imgANN does not expose the required interface

        # Keep the backbone fixed even if the temporal model enters training mode.
        self.img_ann.get_model().requires_grad_(False)
        self.img_ann.get_model().eval()

        # Register hooks once; each forward pass overwrites the captured tensors.
        self.img_ann.create_forward_hook(self.layer_names)

        # One trainable vector represents each discrete time position: [T, E].
        self.position_embeddings = nn.Parameter(
            torch.empty(n_time_bins, position_embedding_dim)
        )
        nn.init.normal_(self.position_embeddings, mean=0.0, std=0.02)

        # Convert each position vector into one unnormalized score per layer.
        self.layer_attention = nn.Linear(position_embedding_dim, self.n_layers)

        # LazyLinear infers the transformer embedding width on the first batch.
        # Separate projections avoid assuming that coordinates match across layers.
        self.layer_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LazyLinear(layer_projection_dim, bias=False),
                    nn.LayerNorm(layer_projection_dim, elementwise_affine=False),
                )
                for _ in self.layer_names
            ]
        )

        # Refine the attention-combined value vector into the requested latent space.
        self.latent_projection = nn.Sequential(
            nn.Linear(layer_projection_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
        )

        # Add a neural-output map only when an output dimension is requested.
        self.prediction_head = (
            nn.Linear(latent_dim, output_dim) if output_dim is not None else None
        )

    """
    _resolve_time_idx
    Validate the requested time specification and convert it into discrete indices
    on the temporal model's device. Physical values determine only the bin count.

    INPUT:
        - time_idx: torch.Tensor | None -> explicit discrete position indices
        - time_values: torch.Tensor | None -> ordered physical time values

    OUTPUT:
        - time_idx: torch.Tensor -> validated indices [time]
    """
    def _resolve_time_idx(
        self,
        time_idx: torch.Tensor | None,
        time_values: torch.Tensor | None,
    ) -> torch.Tensor:
        # The two arguments are alternative ways to specify temporal bins.
        if time_idx is not None and time_values is not None:
            raise ValueError("Pass either time_idx or time_values, not both.")
        # end if both time specifications are provided

        device = self.position_embeddings.device
        if time_idx is None:
            if time_values is not None:
                # Learned positions use only the length, not physical magnitudes.
                time_values = torch.as_tensor(time_values, device=device)
                if time_values.ndim != 1 or time_values.numel() == 0:
                    raise ValueError(
                        f"time_values must have shape [time], got {time_values.shape}."
                    )
                # end if time_values has the wrong shape
            # end if time_values is provided

            # No temporal input selects every configured position.
            n_time = self.n_time_bins if time_values is None else len(time_values)
            if n_time > self.n_time_bins:
                raise ValueError(
                    f"Requested {n_time} bins, but the model has {self.n_time_bins}."
                )
            # end if too many time bins are requested
            time_idx = torch.arange(n_time, device=device)
        else:
            # Explicit positions must be integer indices on the temporal device.
            time_idx = torch.as_tensor(time_idx, device=device, dtype=torch.long)
        # end if time_idx is None

        if time_idx.ndim != 1 or time_idx.numel() == 0:
            raise ValueError(f"time_idx must have shape [time], got {time_idx.shape}.")
        if time_idx.min() < 0 or time_idx.max() >= self.n_time_bins:
            raise IndexError(
                f"time_idx must be in [0, {self.n_time_bins - 1}], got "
                f"[{time_idx.min().item()}, {time_idx.max().item()}]."
            )
        # end if time_idx is out of bounds
        return time_idx
    # EOF

    """
    _prepare_img_ann_input
    Translate a tensor image batch into the keyword expected by the imgANN package.

    INPUT:
        - x: torch.Tensor | dict -> image batch or explicit model keyword inputs

    OUTPUT:
        - ann_input: torch.Tensor | dict -> input accepted by imgANN.extract_features
    """
    def _prepare_img_ann_input(self, x):
        # Explicit dictionaries already name the backbone's forward arguments.
        if isinstance(x, dict):
            return x
        # Hugging Face vision models expect pixel_values rather than x.
        get_pkg = getattr(self.img_ann, "get_pkg", None)
        if callable(get_pkg) and get_pkg() == "hf":
            return {"pixel_values": x}
        # Torchvision and timm imgANN models accept the tensor through x.
        return x
    # EOF

    """
    _extract_layer_features
    Run the frozen backbone, collect pooled hooked activations, and stack them.

    INPUT:
        - x: torch.Tensor | dict -> image-model inputs

    OUTPUT:
        - layer_features: torch.Tensor -> features [batch, layers, feature_dim]
    """
    def _extract_layer_features(self, x) -> torch.Tensor:
        # imgANN performs the backbone forward pass under torch.no_grad().
        ann_input = self._prepare_img_ann_input(x)
        captured_features = self.img_ann.extract_features(ann_input)

        # Preserve the exact user-provided layer order on the attention axis.
        missing_layers = [
            layer_name
            for layer_name in self.layer_names
            if layer_name not in captured_features
        ]
        if missing_layers:
            raise KeyError(f"Hooks did not capture layers: {missing_layers}.")
        # end if requested hook outputs are missing
        ordered_features = [captured_features[name] for name in self.layer_names]

        # Mean/CLS pooling must yield one vector per image for every layer.
        invalid_shapes = {
            name: tuple(features.shape)
            for name, features in zip(self.layer_names, ordered_features)
            if not isinstance(features, torch.Tensor) or features.ndim != 2
        }
        if invalid_shapes:
            raise ValueError(
                "Hooked features must have shape [batch, feature_dim]. Configure "
                f"imgANN pooling accordingly; got {invalid_shapes}."
            )
        # end if hooked activations are not pooled vectors

        # All attended transformer layers must expose the same embedding width.
        batch_sizes = {features.shape[0] for features in ordered_features}
        feature_dims = {features.shape[1] for features in ordered_features}
        if len(batch_sizes) != 1 or len(feature_dims) != 1:
            shapes = [tuple(features.shape) for features in ordered_features]
            raise ValueError(f"Hooked layer feature shapes do not align: {shapes}.")
        # end if batch sizes or embedding dimensions differ

        current_feature_dim = ordered_features[0].shape[1]
        if self.feature_dim is None:
            # Record the transformer width inferred by the first image batch.
            self.feature_dim = current_feature_dim
        elif current_feature_dim != self.feature_dim:
            raise ValueError(
                f"Transformer feature_dim changed from {self.feature_dim} to "
                f"{current_feature_dim}."
            )
        # end if feature_dim has not yet been inferred

        # Match the trainable temporal parameters' device and floating-point dtype.
        device = self.position_embeddings.device
        dtype = self.position_embeddings.dtype
        ordered_features = [features.to(device=device, dtype=dtype) for features in ordered_features]
        return torch.stack(ordered_features, dim=1)  # [B, K, D]
    # EOF

    """
    forward
    Extract and align frozen layer features, compute position-only layer attention,
    and return latent representations plus optional neural predictions.

    INPUT:
        - x: torch.Tensor | dict -> image batch or imgANN keyword inputs
        - time_idx: torch.Tensor | None -> discrete positions to evaluate [time]
        - time_values: torch.Tensor | None -> physical times used to select positions
        - return_attention: bool -> whether to include layer weights in the output

    OUTPUT:
        - pred: torch.Tensor | None -> predictions [batch, time, output_dim]
        - latent: torch.Tensor -> representations [batch, time, latent_dim]
        - attention_weights: torch.Tensor | None -> weights [batch, time, layers]
    """
    def forward(
        self,
        x,
        time_idx: torch.Tensor | None = None,
        time_values: torch.Tensor | None = None,
        return_attention: bool = True,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        # Run the frozen transformer and collect pooled features: [B, K, D].
        layer_features = self._extract_layer_features(x)

        # Each layer learns its own D -> V alignment into the common value space.
        projected_layers = torch.stack(
            [
                projection(layer_features[:, layer_idx])
                for layer_idx, projection in enumerate(self.layer_projections)
            ],
            dim=1,
        )  # [B, K, V]

        # Resolve the requested temporal positions and look up their vectors.
        time_idx = self._resolve_time_idx(time_idx, time_values)  # [T]
        position_features = self.position_embeddings[time_idx]  # [T, E]

        # Position embeddings alone produce a probability distribution over layers.
        layer_logits = self.layer_attention(position_features)  # [T, K]
        time_layer_attention = torch.softmax(layer_logits, dim=-1)  # [T, K]

        # The time x layer schedule is global and therefore shared across images.
        attention_weights = time_layer_attention.unsqueeze(0).expand(
            projected_layers.shape[0], -1, -1
        )  # [B, T, K]

        # Combine aligned layer values independently at each temporal position.
        context = torch.einsum(
            "btk,bkv->btv", attention_weights, projected_layers
        )  # [B, T, V]

        # Apply one shared map at every image and time position.
        latent = self.latent_projection(context)  # [B, T, latent_dim]
        pred = self.prediction_head(latent) if self.prediction_head is not None else None

        # Attention can be omitted from the return value during routine training.
        returned_attention = attention_weights if return_attention else None
        return pred, latent, returned_attention
    # EOF
# EOC


class BaselineModel(nn.Module):
    """
    Predict time-resolved neural activity from selected frozen ANN layers.

    Each learned temporal query attends over static layer features. All feature
    transformations operate pointwise in time, and every target time bin owns a
    separate neural readout head. There is no recurrence or temporal mixing.

    INPUT (forward):
        - x: torch.Tensor -> images or cached features
        - use_precomputed_features: bool -> whether x is [batch, layers, embedding]

    OUTPUT:
        - neural_predictions: torch.Tensor -> activity [batch, time, neurons]
        - attention_weights: torch.Tensor -> layer weights [batch, time, layers]
    """

    """
    __init__
    Freeze the image encoder and construct the time-local attention decoder.

    INPUT:
        - encoder: imgANN -> wrapped frozen image encoder
        - layers: list[str] -> ordered hooked ANN layers
        - temporal_embedding_dim: int -> learned query width
        - value_dim: int -> projected layer-value width
        - n_timepoints: int -> number of neural target bins
        - temporal_compression_ratio: int -> must be one for time-local prediction
        - n_neurons: int -> number of neural output channels
        - mlp_hidden_dim: int -> hidden width of pointwise decoder MLPs
        - dropout: float -> decoder dropout probability
        - key_query_dim: int | None -> optional shared key/query width

    OUTPUT:
        - None
    """
    def __init__(
        self,
        encoder,
        layers,
        temporal_embedding_dim,
        value_dim,
        n_timepoints,
        temporal_compression_ratio,
        n_neurons,
        mlp_hidden_dim,
        dropout=0.0,
        key_query_dim=None,
    ):
        super().__init__()

        self.layer_names = list(layers)
        self.n_layers = len(self.layer_names)
        if self.n_layers == 0:
            raise ValueError("layers must contain at least one ANN layer.")
        # end if no ANN layers were requested
        if temporal_compression_ratio != 1:
            raise ValueError(
                "temporal_compression_ratio must be 1 while the model "
                "uses strictly time-local predictions."
            )
        # end if temporal compression would couple output bins

        # Freeze the wrapped encoder and register its underlying nn.Module so
        # model.to(), parameters(), and state_dict() handle it consistently.
        encoder.model.eval()
        self.encoder = encoder
        for parameter in self.encoder.model.parameters():
            parameter.requires_grad_(False)
        # end for encoder parameter
        self.encoder.set_relevant_layers(self.layer_names)
        self.encoder_dim = self.encoder.get_layer_output_shape(
            self.layer_names[0]
        )[1]
        self.encoder.create_forward_hook()
        self.encoder_backbone = self.encoder.model

        # One learned query represents every neural time bin: [T, E_te].
        self.temporal_compression_ratio = temporal_compression_ratio
        self.n_temporal_embeddings = n_timepoints
        self.n_timepoints = n_timepoints
        self.n_neurons = n_neurons
        self.temporal_embedding_dim = temporal_embedding_dim
        self.temporal_embeddings = nn.Parameter(
            torch.randn(n_timepoints, temporal_embedding_dim) * 0.02
        )

        # Keys and queries share a width for scaled dot-product attention.
        self.key_dim = (
            key_query_dim
            if key_query_dim is not None
            else temporal_embedding_dim
        )
        self.query_dim = self.key_dim
        self.value_dim = value_dim

        # Project the static ANN layer vectors into attention keys and values.
        self.key_projection = nn.Linear(
            self.encoder_dim,
            self.key_dim,
            bias=False,
        )
        self.value_projection = nn.Linear(
            self.encoder_dim,
            self.value_dim,
            bias=False,
        )
        if self.query_dim != self.temporal_embedding_dim:
            self.query_projection = nn.Linear(
                self.temporal_embedding_dim,
                self.query_dim,
                bias=False,
            )
        # end if temporal queries need projection
        self.key_norm = nn.LayerNorm(self.key_dim)
        self.value_norm = nn.LayerNorm(self.value_dim)
        self.dropout = nn.Dropout(dropout)

        # These MLPs read only the last tensor dimension and cannot mix time.
        self.temporal_feature_mlp = nn.Sequential(
            nn.Linear(value_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, value_dim),
        )
        self.neural_feature_mlp = nn.Sequential(
            nn.Linear(value_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Head t receives only latent t and owns its neural prediction weights.
        self.timebin_readouts = nn.ModuleList(
            [
                nn.Linear(mlp_hidden_dim, n_neurons)
                for _ in range(n_timepoints)
            ]
        )

    def train(self, mode=True):
        # Train the decoder while keeping the frozen backbone deterministic.
        super().train(mode)
        self.encoder_backbone.eval()
        return self
    # EOF

    @property
    def device(self):
        return next(self.parameters()).device
    # EOF

    # --- GETTERS ---
    def get_encoder(self):
        return self.encoder

    def get_layer_names(self) -> list[str]:
        return self.layer_names

    def get_encoder_dim(self) -> int:
        return self.encoder_dim

    def get_temporal_embedding_dim(self) -> int:
        return self.temporal_embedding_dim

    def get_n_temporal_embeddings(self) -> int:
        return self.temporal_embeddings.shape[0]

    def get_temporal_compression_ratio(self) -> int:
        return self.temporal_compression_ratio

    def get_n_timepoints(self) -> int:
        return self.n_timepoints

    def get_n_neurons(self) -> int:
        return self.n_neurons

    def get_key_dim(self) -> int:
        return self.key_dim

    def get_query_dim(self) -> int:
        return self.query_dim

    def get_value_dim(self) -> int:
        return self.value_dim

    def get_trainable_parameters(self):
        return (
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def get_trainable_named_parameters(self):
        return (
            (name, parameter)
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        )
    # EOF

    """
    _resolve_layer_features
    Select cached layer features or extract them online with frozen DINO.

    INPUT:
        - x: torch.Tensor -> images or cached features
        - use_precomputed_features: bool -> whether x is [batch, layers, embedding]

    OUTPUT:
        - layer_features: torch.Tensor -> features [batch, layers, embedding]
    """
    def _resolve_layer_features(self, x, use_precomputed_features):
        if use_precomputed_features:
            layer_features = x
        else:
            # Hooks registered on the frozen encoder collect requested layers.
            with torch.no_grad():
                self.encoder.model(x)
            # end with frozen encoder forward
            captured_features = self.encoder.features
            layer_features = torch.stack(
                [captured_features[layer] for layer in self.layer_names],
                dim=1,
            )
        # end if precomputed layer features are supplied

        expected_shape = (self.n_layers, self.encoder_dim)
        has_expected_shape = (
            layer_features.ndim == 3
            and tuple(layer_features.shape[1:]) == expected_shape
        )
        if not has_expected_shape:
            raise ValueError(
                "Expected layer features with shape [batch, layers, embedding] "
                f"and trailing dimensions {expected_shape}, got "
                f"{tuple(layer_features.shape)}."
            )
        # end if layer features have the wrong shape

        # Cached CPU arrays follow the trainable projections' device and dtype.
        return layer_features.to(
            device=self.key_projection.weight.device,
            dtype=self.key_projection.weight.dtype,
        )
    # EOF

    """
    forward
    Attend over frozen ANN layers and predict every neural time bin locally.

    INPUT:
        - x: torch.Tensor -> images or cached features
        - use_precomputed_features: bool -> whether x is [batch, layers, embedding]

    OUTPUT:
        - neural_predictions: torch.Tensor -> activity [batch, time, neurons]
        - attention_weights: torch.Tensor -> layer weights [batch, time, layers]
    """
    def forward(self, x, use_precomputed_features=False):
        # Resolve one pooled feature vector per selected ANN layer: [B, L, E].
        layer_features = self._resolve_layer_features(
            x,
            use_precomputed_features,
        )

        # Build normalized layer keys and values: [B, L, E_k/E_v].
        keys = self.dropout(self.key_norm(self.key_projection(layer_features)))
        values = self.dropout(
            self.value_norm(self.value_projection(layer_features))
        )

        # Expand the learned time queries across images: [B, T, E_k].
        if hasattr(self, "query_projection"):
            queries = self.query_projection(self.temporal_embeddings)
        else:
            queries = self.temporal_embeddings
        # end if temporal queries require projection
        queries = queries.unsqueeze(0).expand(layer_features.shape[0], -1, -1)

        # Each time query attends only over the ANN layer axis.
        attention_logits = torch.matmul(queries, keys.transpose(-1, -2))
        attention_logits = attention_logits / math.sqrt(self.key_dim)
        attention_weights = torch.softmax(attention_logits, dim=-1)

        # Weighted layer values produce one independent latent per time bin.
        coarse_latents = torch.matmul(attention_weights, values)
        fine_latents = self.temporal_feature_mlp(coarse_latents)
        neural_features = self.neural_feature_mlp(fine_latents)

        # Apply readout t only to neural feature t; never mix temporal bins.
        timebin_predictions = []
        for timebin, timebin_readout in enumerate(self.timebin_readouts):
            current_prediction = timebin_readout(
                neural_features[:, timebin, :]
            )
            timebin_predictions.append(current_prediction)
        # end for time-bin-specific readout
        neural_predictions = torch.stack(timebin_predictions, dim=1)
        return neural_predictions, attention_weights
    # EOF
# EOC
