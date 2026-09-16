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

    Each learned temporal query attends either over whole ANN layers or over the
    individual embedding coordinates within those layers. Each coarse attended
    latent is expanded into a non-overlapping block of fine-time latents, and
    every target time bin owns a separate neural readout head. There is no
    recurrence or mixing between coarse temporal blocks.

    INPUT (forward):
        - x: torch.Tensor -> images or cached features
        - use_precomputed_features: bool -> whether x is [batch, layers, embedding]

    OUTPUT:
        - neural_predictions: torch.Tensor -> activity [batch, time, neurons]
        - attention_weights: torch.Tensor -> layer weights [B, T, L] or feature
          weights [B, T, L, E], depending on attention_granularity
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
        - temporal_compression_ratio: int -> fine target bins generated by each
          coarse temporal query; must divide n_timepoints exactly
        - n_neurons: int -> number of neural output channels
        - mlp_hidden_dim: int -> hidden width of pointwise decoder MLPs
        - dropout: float -> decoder dropout probability
        - temporal_embedding_dropout: float -> dropout probability applied to
          the learned temporal queries, independently of the decoder dropout
        - key_query_dim: int | None -> optional shared key/query width
        - attention_granularity: str -> "layer" or "feature"

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
        temporal_embedding_dropout=0.0,
        key_query_dim=None,
        attention_granularity="layer",
    ):
        super().__init__()

        self.layer_names = list(layers)
        self.n_layers = len(self.layer_names)

        # Cached activations always arrive with the hooked-layer count. Keeping
        # it separate lets a subclass append pseudo-layers to self.n_layers
        # (which sizes the attention) without invalidating the input check.
        self.n_expected_input_layers = self.n_layers
        if self.n_layers == 0:
            raise ValueError("layers must contain at least one ANN layer.")
        # end if no ANN layers were requested
        integer_compression_ratio = int(temporal_compression_ratio)
        if (
            integer_compression_ratio <= 0
            or integer_compression_ratio != temporal_compression_ratio
        ):
            raise ValueError(
                "temporal_compression_ratio must be a positive integer."
            )
        # end if temporal compression is invalid
        temporal_compression_ratio = integer_compression_ratio
        if n_timepoints % temporal_compression_ratio != 0:
            raise ValueError(
                "n_timepoints must be divisible by temporal_compression_ratio."
            )
        # end if coarse temporal blocks cannot tile the target sequence
        if not 0.0 <= temporal_embedding_dropout < 1.0:
            raise ValueError(
                "temporal_embedding_dropout must lie in [0, 1)."
            )
        # end if the temporal query dropout is invalid
        if attention_granularity not in {"layer", "feature"}:
            raise ValueError(
                "attention_granularity must be either 'layer' or 'feature'."
            )
        # end if attention granularity is invalid

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
        self.attention_granularity = attention_granularity

        # One learned query represents each coarse non-overlapping time block.
        self.temporal_compression_ratio = temporal_compression_ratio
        self.n_temporal_embeddings = (
            n_timepoints // temporal_compression_ratio
        )
        self.n_timepoints = n_timepoints
        self.n_neurons = n_neurons
        self.temporal_embedding_dim = temporal_embedding_dim
        self.temporal_embeddings = nn.Parameter(
            torch.randn(
                self.n_temporal_embeddings,
                temporal_embedding_dim,
            ) * 0.02
        )

        # Keys and queries share a width for scaled dot-product attention.
        self.key_dim = (
            key_query_dim
            if key_query_dim is not None
            else temporal_embedding_dim
        )
        self.query_dim = self.key_dim
        self.value_dim = value_dim

        # Whole-layer mode preserves the original L-item cross-attention. In
        # feature mode, every one of the L * E embedding coordinates receives a
        # learned key direction and is gated separately at every time bin.
        if self.attention_granularity == "layer":
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
            self.key_norm = nn.LayerNorm(self.key_dim)
        else:
            self.n_attention_features = self.n_layers * self.encoder_dim
            self.feature_input_norm = nn.LayerNorm(
                self.encoder_dim,
                elementwise_affine=False,
            )
            self.feature_key_embeddings = nn.Parameter(
                torch.empty(
                    self.n_layers,
                    self.encoder_dim,
                    self.key_dim,
                )
            )
            nn.init.normal_(
                self.feature_key_embeddings,
                mean=0.0,
                std=0.02,
            )
            self.feature_value_projection = nn.Linear(
                self.n_attention_features,
                self.value_dim,
                bias=False,
            )
        # end if attention operates over layers or embedding features
        if self.query_dim != self.temporal_embedding_dim:
            self.query_projection = nn.Linear(
                self.temporal_embedding_dim,
                self.query_dim,
                bias=False,
            )
        # end if temporal queries need projection
        self.value_norm = nn.LayerNorm(self.value_dim)
        self.dropout = nn.Dropout(dropout)

        # The query path carries no stimulus information, so it is regularized
        # separately from the key/value path rather than sharing self.dropout.
        self.temporal_embedding_dropout = nn.Dropout(temporal_embedding_dropout)

        # Expand each coarse attended value into R ordered fine-time values.
        # The MLP is shared across coarse blocks and never mixes between them.
        self.temporal_feature_mlp = nn.Sequential(
            nn.Linear(value_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                mlp_hidden_dim,
                temporal_compression_ratio * value_dim,
            ),
        )
        # This pointwise MLP then processes every fine-time value independently.
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

    def get_temporal_embedding_dropout(self) -> float:
        return self.temporal_embedding_dropout.p

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

    def get_attention_granularity(self) -> str:
        return self.attention_granularity

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

        expected_shape = (self.n_expected_input_layers, self.encoder_dim)
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
            device=self.temporal_embeddings.device,
            dtype=self.temporal_embeddings.dtype,
        )
    # EOF

    """
    _compute_layer_attention
    Attend over one projected key/value vector per selected ANN layer.

    INPUT:
        - layer_features: torch.Tensor -> features [batch, layers, embedding]
        - queries: torch.Tensor -> temporal queries [batch, time, key_dim]

    OUTPUT:
        - coarse_latents: torch.Tensor -> attended values [batch, time, value_dim]
        - attention_weights: torch.Tensor -> layer weights [batch, time, layers]
    """
    def _compute_layer_attention(self, layer_features, queries):
        keys = self.dropout(
            self.key_norm(self.key_projection(layer_features))
        )
        values = self.dropout(
            self.value_norm(self.value_projection(layer_features))
        )
        attention_logits = torch.matmul(queries, keys.transpose(-1, -2))
        attention_logits = attention_logits / math.sqrt(self.key_dim)
        attention_weights = torch.softmax(attention_logits, dim=-1)
        coarse_latents = torch.matmul(attention_weights, values)
        return coarse_latents, attention_weights
    # EOF

    """
    _compute_feature_attention
    Attend over every embedding coordinate within every selected ANN layer.

    INPUT:
        - layer_features: torch.Tensor -> features [batch, layers, embedding]
        - queries: torch.Tensor -> temporal queries [batch, time, key_dim]

    OUTPUT:
        - coarse_latents: torch.Tensor -> attended values [batch, time, value_dim]
        - attention_weights: torch.Tensor -> feature weights
          [batch, time, layers, embedding]
    """
    def _compute_feature_attention(self, layer_features, queries):
        # Normalize each layer vector without learning a second feature gate.
        normalized_features = self.feature_input_norm(layer_features)

        # Contract queries with the learned coordinate keys before applying the
        # stimulus activation. This avoids materializing [B, L, E, E_k].
        feature_key_embeddings = self.dropout(self.feature_key_embeddings)
        attention_logits = torch.einsum(
            "btk,lek->btle",
            queries,
            feature_key_embeddings,
        )
        attention_logits = (
            attention_logits * normalized_features.unsqueeze(1)
        )

        # Each temporal query scores all L * E individual feature coordinates.
        attention_logits = attention_logits / math.sqrt(self.key_dim)
        flat_logits = attention_logits.flatten(start_dim=-2)
        attention_weights = torch.softmax(flat_logits, dim=-1).reshape_as(
            attention_logits
        )

        # Uniform attention should initially preserve the normalized feature
        # scale. Multiplication by L * E makes a uniform gate equal to one.
        gated_features = (
            normalized_features.unsqueeze(1)
            * attention_weights
            * self.n_attention_features
        )
        flat_gated_features = gated_features.flatten(start_dim=-2)
        coarse_latents = self.feature_value_projection(flat_gated_features)
        coarse_latents = self.dropout(self.value_norm(coarse_latents))
        return coarse_latents, attention_weights
    # EOF

    """
    _build_temporal_queries
    Broadcast the learned temporal embeddings over the batch, optionally drop
    query coordinates, and project the result into key space.

    INPUT:
        - batch_size: int -> number of images in the current forward pass

    OUTPUT:
        - queries: torch.Tensor -> temporal queries [batch, time, key_dim]
    """
    def _build_temporal_queries(self, batch_size):
        # Expand before dropping so every image gets its own query mask, the
        # same per-image convention TemporalNoiseBaselineModel uses: [B, T, E_te].
        temporal_embeddings = self.temporal_embeddings.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        temporal_embeddings = self.temporal_embedding_dropout(
            temporal_embeddings
        )
        if hasattr(self, "query_projection"):
            return self.query_projection(temporal_embeddings)
        # end if temporal queries require projection
        return temporal_embeddings
    # EOF

    """
    forward
    Attend over frozen ANN layers and predict every neural time bin locally.

    INPUT:
        - x: torch.Tensor -> images or cached features
        - use_precomputed_features: bool -> whether x is [batch, layers, embedding]

    OUTPUT:
        - neural_predictions: torch.Tensor -> activity [batch, time, neurons]
        - attention_weights: torch.Tensor -> coarse layer weights [B, T/R, L]
          or feature weights [B, T/R, L, E], depending on attention_granularity
    """
    def forward(self, x, use_precomputed_features=False):
        # Resolve one pooled feature vector per selected ANN layer: [B, L, E].
        layer_features = self._resolve_layer_features(
            x,
            use_precomputed_features,
        )

        # Expand the coarse learned time queries: [B, T/R, E_k].
        queries = self._build_temporal_queries(layer_features.shape[0])

        # Select the requested attention axis without changing temporal flow.
        if self.attention_granularity == "layer":
            coarse_latents, attention_weights = self._compute_layer_attention(
                layer_features,
                queries,
            )
        else:
            coarse_latents, attention_weights = self._compute_feature_attention(
                layer_features,
                queries,
            )
        # end if attention operates over layers or embedding features

        # Expand each coarse latent into R ordered, non-overlapping fine bins:
        # [B, T/R, V] -> [B, T/R, R * V] -> [B, T, V].
        fine_latent_blocks = self.temporal_feature_mlp(coarse_latents)
        fine_latent_blocks = fine_latent_blocks.reshape(
            fine_latent_blocks.shape[0],
            self.n_temporal_embeddings,
            self.temporal_compression_ratio,
            self.value_dim,
        )
        fine_latents = fine_latent_blocks.reshape(
            fine_latent_blocks.shape[0],
            self.n_timepoints,
            self.value_dim,
        )
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


# Name reported in ``get_layer_names`` for the appended stochastic pseudo-layer.
GAUSSIAN_NOISE_LAYER_NAME = "__gaussian_sphere_noise__"


class NoiseLayerBaselineModel(BaselineModel):
    """
    BaselineModel that attends over the hooked ANN layers plus one noise layer.

    A single extra pseudo-layer is drawn uniformly from the unit sphere in
    encoder-dimension space (a standard-normal draw divided by its own norm),
    independently for every image and every forward pass. The vector carries no
    stimulus information, so every unit of attention mass spent on it is wasted
    prediction budget: it regularizes the temporal queries by making a
    confidently peaked read of any single real layer expensive, and it leaves
    the decoder stochastic at evaluation time.

    INPUT (forward):
        - x: torch.Tensor -> images or cached features
        - use_precomputed_features: bool -> whether x is [batch, layers, embedding]

    OUTPUT:
        - neural_predictions: torch.Tensor -> activity [batch, time, neurons]
        - attention_weights: torch.Tensor -> weights over the L + 1 attended
          items, [B, T, L + 1] or [B, T, L + 1, E]
    """

    """
    __init__
    Build the BaselineModel decoder, then widen its attention by one pseudo-layer.

    INPUT:
        - encoder: imgANN -> wrapped frozen image encoder
        - layers: list[str] -> ordered hooked ANN layers, without the noise layer
        - match_noise_norm: bool -> rescale the sphere draw to the mean norm of
          that image's real layer vectors, so it competes on equal footing
        - noise_in_eval: bool -> keep sampling noise outside training mode
        - baseline_kwargs: dict -> remaining BaselineModel arguments

    OUTPUT:
        - None
    """
    def __init__(
        self,
        encoder,
        layers,
        match_noise_norm=True,
        noise_in_eval=True,
        **baseline_kwargs,
    ):
        super().__init__(encoder, layers, **baseline_kwargs)
        self.match_noise_norm = bool(match_noise_norm)
        self.noise_in_eval = bool(noise_in_eval)

        # Attention now ranges over L + 1 items. n_expected_input_layers stays
        # at L, so cached [batch, L, embedding] activations still validate.
        self.n_layers = self.n_layers + 1
        self.layer_names = [*self.layer_names, GAUSSIAN_NOISE_LAYER_NAME]

        # Layer-granularity keys and values are shared across layers and need no
        # change. Feature granularity indexes its parameters by layer, so those
        # must be rebuilt at the new item count.
        if self.attention_granularity == "feature":
            self.n_attention_features = self.n_layers * self.encoder_dim
            self.feature_key_embeddings = nn.Parameter(
                torch.empty(self.n_layers, self.encoder_dim, self.key_dim)
            )
            nn.init.normal_(self.feature_key_embeddings, mean=0.0, std=0.02)
            self.feature_value_projection = nn.Linear(
                self.n_attention_features,
                self.value_dim,
                bias=False,
            )
        # end if feature-granularity parameters are indexed by layer

    """
    _sample_noise_layer
    Draw one uniform unit-sphere vector per image in the batch.

    INPUT:
        - layer_features: torch.Tensor -> real features [batch, layers, embedding]

    OUTPUT:
        - noise_layer: torch.Tensor -> noise pseudo-layer [batch, 1, embedding]
    """
    def _sample_noise_layer(self, layer_features):
        # Normalizing an isotropic Gaussian draw gives a point distributed
        # uniformly on the encoder_dim-dimensional unit sphere.
        noise = torch.randn(
            layer_features.shape[0],
            1,
            self.encoder_dim,
            device=layer_features.device,
            dtype=layer_features.dtype,
        )
        noise = noise / noise.norm(dim=-1, keepdim=True).clamp_min(
            torch.finfo(noise.dtype).eps
        )

        if self.match_noise_norm:
            # Without rescaling, a unit vector is negligible next to DINO layer
            # activations and attention would simply ignore it. Matching the
            # mean real-layer norm makes it a genuine competitor.
            reference_norm = layer_features.norm(dim=-1).mean(dim=1, keepdim=True)
            noise = noise * reference_norm.unsqueeze(-1)
        # end if the noise layer is scale-matched to the real layers
        return noise
    # EOF

    """
    _resolve_layer_features
    Resolve the real ANN layers, then append the stochastic noise pseudo-layer.

    INPUT:
        - x: torch.Tensor -> images or cached features
        - use_precomputed_features: bool -> whether x is [batch, layers, embedding]

    OUTPUT:
        - layer_features: torch.Tensor -> features [batch, layers + 1, embedding]
    """
    def _resolve_layer_features(self, x, use_precomputed_features):
        layer_features = super()._resolve_layer_features(
            x,
            use_precomputed_features,
        )
        if self.training or self.noise_in_eval:
            noise_layer = self._sample_noise_layer(layer_features)
        else:
            # A zero vector keeps the extra attended item in place but makes
            # evaluation deterministic when the noise is switched off.
            noise_layer = torch.zeros(
                layer_features.shape[0],
                1,
                self.encoder_dim,
                device=layer_features.device,
                dtype=layer_features.dtype,
            )
        # end if the noise layer is sampled or silenced
        return torch.cat([layer_features, noise_layer], dim=1)
    # EOF
# EOC


class TemporalNoiseBaselineModel(BaselineModel):
    """
    BaselineModel that perturbs its learned temporal embeddings with noise.

    Every image in the batch draws its own Gaussian perturbation of the
    [time, temporal_embedding_dim] query table before that table is projected
    into key space. No target time bin can then rely on an exact query
    direction, which smooths the learned layer-attention schedule over
    neighbouring time bins. The frozen visual features are left untouched, so
    this isolates regularization of the temporal code itself.

    INPUT (forward):
        - x: torch.Tensor -> images or cached features
        - use_precomputed_features: bool -> whether x is [batch, layers, embedding]

    OUTPUT:
        - neural_predictions: torch.Tensor -> activity [batch, time, neurons]
        - attention_weights: torch.Tensor -> layer weights [B, T, L] or feature
          weights [B, T, L, E], depending on attention_granularity
    """

    """
    __init__
    Build the BaselineModel decoder and store the temporal-jitter settings.

    INPUT:
        - encoder: imgANN -> wrapped frozen image encoder
        - layers: list[str] -> ordered hooked ANN layers
        - temporal_noise_std: float -> perturbation scale
        - relative_noise: bool -> read temporal_noise_std as a fraction of the
          current embedding spread instead of an absolute standard deviation
        - noise_in_eval: bool -> keep perturbing outside training mode
        - baseline_kwargs: dict -> remaining BaselineModel arguments

    OUTPUT:
        - None
    """
    def __init__(
        self,
        encoder,
        layers,
        temporal_noise_std=0.25,
        relative_noise=True,
        noise_in_eval=False,
        **baseline_kwargs,
    ):
        super().__init__(encoder, layers, **baseline_kwargs)
        if temporal_noise_std < 0.0:
            raise ValueError("temporal_noise_std must be non-negative.")
        # end if the requested jitter is invalid
        self.temporal_noise_std = float(temporal_noise_std)
        self.relative_noise = bool(relative_noise)
        self.noise_in_eval = bool(noise_in_eval)

    """
    _sample_embedding_noise
    Draw one independent perturbation of the temporal query table per image.

    INPUT:
        - temporal_embeddings: torch.Tensor -> broadcast queries
          [batch, time, temporal_embedding_dim]

    OUTPUT:
        - noise: torch.Tensor -> additive noise with the same shape
    """
    def _sample_embedding_noise(self, temporal_embeddings):
        noise_std = self.temporal_noise_std
        if self.relative_noise:
            # The embedding table starts at std 0.02 and grows during training,
            # so a fixed absolute std would be crippling early and negligible
            # later. Tracking the current spread keeps the jitter proportional.
            noise_std = noise_std * self.temporal_embeddings.detach().std()
        # end if the jitter follows the learned embedding scale
        return torch.randn_like(temporal_embeddings) * noise_std
    # EOF

    """
    _build_temporal_queries
    Perturb the temporal embeddings per image, drop query coordinates, then
    project them into key space.

    INPUT:
        - batch_size: int -> number of images in the current forward pass

    OUTPUT:
        - queries: torch.Tensor -> temporal queries [batch, time, key_dim]
    """
    def _build_temporal_queries(self, batch_size):
        # Expand first so every image receives its own perturbation: [B, T, E_te].
        temporal_embeddings = self.temporal_embeddings.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        if self.training or self.noise_in_eval:
            temporal_embeddings = temporal_embeddings + (
                self._sample_embedding_noise(temporal_embeddings)
            )
        # end if the temporal jitter is active

        # Drop after the jitter so a masked coordinate stays at zero rather
        # than being refilled by its own perturbation.
        temporal_embeddings = self.temporal_embedding_dropout(
            temporal_embeddings
        )
        if hasattr(self, "query_projection"):
            return self.query_projection(temporal_embeddings)
        # end if temporal queries require projection
        return temporal_embeddings
    # EOF
# EOC


class ShowAttendTellGRUModel(nn.Module):
    """
    Predict neural dynamics with recurrent attention over static ANN features.

    The design follows the soft-attention decoder in Show, Attend and Tell. The
    previous GRU state scores every coordinate of every selected ANN layer, the
    attended feature context updates one shared GRUCell, and one shared readout
    predicts neural activity. Time is represented only by recurrent state: the
    model contains no temporal embeddings or time-specific parameters.

    INPUT (forward):
        - x: torch.Tensor -> images or cached features
        - use_precomputed_features: bool -> whether x is [batch, layers, embedding]

    OUTPUT:
        - neural_predictions: torch.Tensor -> activity [batch, time, neurons]
        - attention_weights: torch.Tensor -> feature weights
          [batch, time, layers, embedding]
    """

    """
    __init__
    Freeze the image encoder and construct one feature-attention GRU layer.

    INPUT:
        - encoder: imgANN -> wrapped frozen image encoder
        - layers: list[str] -> ordered hooked ANN layers
        - n_timepoints: int -> number of recurrent neural target bins
        - n_neurons: int -> number of neural output channels
        - hidden_dim: int -> GRU state and attended-context width
        - attention_dim: int -> feature-key and hidden-query width
        - dropout: float -> dropout probability before the shared readout

    OUTPUT:
        - None
    """
    def __init__(
        self,
        encoder,
        layers,
        n_timepoints,
        n_neurons,
        hidden_dim,
        attention_dim,
        dropout=0.0,
    ):
        super().__init__()

        # Validate every dimension before constructing trainable parameters.
        self.layer_names = list(layers)
        dimensions = (
            len(self.layer_names),
            n_timepoints,
            n_neurons,
            hidden_dim,
            attention_dim,
        )
        if min(dimensions) <= 0:
            raise ValueError("All model dimensions and layers must be non-empty.")
        # end if a model dimension is invalid
        if len(self.layer_names) != len(set(self.layer_names)):
            raise ValueError("layers must not contain duplicates.")
        # end if layer names are duplicated
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1).")
        # end if dropout is invalid

        self.n_layers = len(self.layer_names)
        self.n_timepoints = int(n_timepoints)
        self.n_neurons = int(n_neurons)
        self.hidden_dim = int(hidden_dim)
        self.attention_dim = int(attention_dim)

        # Freeze the wrapped backbone and register hooks in the requested order.
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

        # Normalize each pooled layer vector without adding a learned gate.
        self.feature_input_norm = nn.LayerNorm(
            self.encoder_dim,
            elementwise_affine=False,
        )
        self.n_attention_features = self.n_layers * self.encoder_dim

        # Every layer-feature coordinate has a learned key direction. Its
        # stimulus activation modulates the state-conditioned attention score.
        self.feature_key_embeddings = nn.Parameter(
            torch.empty(
                self.n_layers,
                self.encoder_dim,
                self.attention_dim,
            )
        )
        nn.init.normal_(self.feature_key_embeddings, mean=0.0, std=0.02)
        self.hidden_query_projection = nn.Linear(
            self.hidden_dim,
            self.attention_dim,
            bias=False,
        )

        # As in Show, Attend and Tell, the image initializes the recurrent state.
        self.initial_state = nn.Sequential(
            nn.Linear(self.encoder_dim, self.hidden_dim),
            nn.Tanh(),
        )

        # Project the gated full feature vector into the sole recurrent layer.
        self.context_projection = nn.Linear(
            self.n_attention_features,
            self.hidden_dim,
            bias=False,
        )
        self.context_norm = nn.LayerNorm(self.hidden_dim)
        self.recurrence = nn.GRUCell(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
        )
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(self.hidden_dim, self.n_neurons)

    def train(self, mode=True):
        # Train the recurrent decoder while keeping the frozen ANN deterministic.
        super().train(mode)
        self.encoder_backbone.eval()
        return self
    # EOF

    @property
    def device(self):
        return self.feature_key_embeddings.device
    # EOF

    # --- GETTERS ---
    def get_encoder(self):
        return self.encoder

    def get_layer_names(self) -> list[str]:
        return self.layer_names

    def get_encoder_dim(self) -> int:
        return self.encoder_dim

    def get_n_timepoints(self) -> int:
        return self.n_timepoints

    def get_temporal_compression_ratio(self) -> int:
        # One recurrent update produces exactly one neural target bin.
        return 1

    def get_n_neurons(self) -> int:
        return self.n_neurons

    def get_hidden_dim(self) -> int:
        return self.hidden_dim

    def get_attention_dim(self) -> int:
        return self.attention_dim

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
    Select cached layer features or extract them online with the frozen encoder.

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
            # Forward hooks collect the requested pooled layer representations.
            with torch.no_grad():
                self.encoder.model(x)
            # end with frozen encoder forward
            layer_features = torch.stack(
                [self.encoder.features[layer] for layer in self.layer_names],
                dim=1,
            )
        # end if cached features are supplied

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
        return layer_features.to(
            device=self.device,
            dtype=self.feature_key_embeddings.dtype,
        )
    # EOF

    """
    _compute_feature_attention
    Score individual ANN coordinates from the previous recurrent hidden state.

    INPUT:
        - normalized_features: torch.Tensor -> [batch, layers, embedding]
        - hidden_state: torch.Tensor -> previous GRU state [batch, hidden_dim]

    OUTPUT:
        - context: torch.Tensor -> attended GRU input [batch, hidden_dim]
        - attention_weights: torch.Tensor -> [batch, layers, embedding]
    """
    def _compute_feature_attention(self, normalized_features, hidden_state):
        # The recurrent state is the only time-varying attention query.
        query = self.hidden_query_projection(hidden_state)
        attention_logits = torch.einsum(
            "ba,lea->ble",
            query,
            self.feature_key_embeddings,
        )
        attention_logits = attention_logits * normalized_features
        attention_logits = attention_logits / math.sqrt(self.attention_dim)

        # All coordinates across all selected layers share one probability mass.
        flat_attention = torch.softmax(
            attention_logits.flatten(start_dim=1),
            dim=-1,
        )
        attention_weights = flat_attention.reshape_as(attention_logits)

        # Scale by feature count so initially uniform attention preserves input
        # magnitude instead of shrinking every coordinate by approximately 1/N.
        gated_features = (
            normalized_features.flatten(start_dim=1)
            * flat_attention
            * self.n_attention_features
        )
        context = self.context_norm(self.context_projection(gated_features))
        return context, attention_weights
    # EOF

    """
    forward
    Recurrently attend to static features and predict every neural time bin.

    INPUT:
        - x: torch.Tensor -> images or cached features
        - use_precomputed_features: bool -> whether x is [batch, layers, embedding]
        - return_hidden_states: bool -> also return the recurrent state sequence

    OUTPUT:
        - neural_predictions: torch.Tensor -> activity [batch, time, neurons]
        - attention_weights: torch.Tensor -> weights
          [batch, time, layers, embedding]
        - hidden_states: torch.Tensor -> [batch, time, hidden_dim], only when
          return_hidden_states is True
    """
    def forward(self, x, use_precomputed_features=False, return_hidden_states=False):
        layer_features = self._resolve_layer_features(
            x,
            use_precomputed_features,
        )
        normalized_features = self.feature_input_norm(layer_features)

        # Initialize h_0 from the mean layer representation of each image.
        hidden_state = self.initial_state(normalized_features.mean(dim=1))
        prediction_sequence = []
        attention_sequence = []
        hidden_sequence = []

        # Reuse the same attention, GRU, and readout parameters at every bin.
        for _ in range(self.n_timepoints):
            context, attention_weights = self._compute_feature_attention(
                normalized_features,
                hidden_state,
            )
            hidden_state = self.recurrence(context, hidden_state)
            prediction_sequence.append(
                self.readout(self.dropout(hidden_state))
            )
            attention_sequence.append(attention_weights)
            if return_hidden_states:
                hidden_sequence.append(hidden_state)
            # end if the recurrent state sequence was requested
        # end for recurrent time bin

        neural_predictions = torch.stack(prediction_sequence, dim=1)
        attention_weights = torch.stack(attention_sequence, dim=1)
        if return_hidden_states:
            return (
                neural_predictions,
                attention_weights,
                torch.stack(hidden_sequence, dim=1),
            )
        # end if the recurrent state sequence was requested
        return neural_predictions, attention_weights
    # EOF
# EOC


class VariationalGRUEncoder(nn.Module):
    """
    Encode a complete sequence with GRUCells and locked recurrent dropout.

    One input mask and one recurrent-state mask are sampled per sequence and
    reused at every time point. This implements recurrent variational dropout
    for a one-layer GRU, where ``nn.GRU(dropout=...)`` would have no effect.

    INPUT (forward):
        - sequence: torch.Tensor -> neural activity [batch, time, input_dim]

    OUTPUT:
        - hidden_sequence: torch.Tensor -> all final-layer states
          [batch, time, hidden_dim]
    """

    """
    __init__
    Construct one or two explicitly unrolled GRU layers.

    INPUT:
        - input_dim: int -> feature width of each neural time point
        - hidden_dim: int -> recurrent state width
        - n_layers: int -> number of recurrent layers, restricted to one or two
        - variational_dropout: float -> locked input and recurrent dropout rate

    OUTPUT:
        - None
    """
    def __init__(
        self,
        input_dim,
        hidden_dim,
        n_layers=1,
        variational_dropout=0.0,
    ):
        super().__init__()
        if min(input_dim, hidden_dim) <= 0:
            raise ValueError("input_dim and hidden_dim must be positive.")
        # end if a recurrent dimension is invalid
        if n_layers not in {1, 2}:
            raise ValueError("n_layers must be either one or two.")
        # end if an unsupported recurrent depth is requested
        if not 0.0 <= variational_dropout < 1.0:
            raise ValueError("variational_dropout must lie in [0, 1).")
        # end if variational dropout is invalid

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.variational_dropout = float(variational_dropout)
        self.cells = nn.ModuleList(
            [
                nn.GRUCell(
                    self.input_dim if layer_idx == 0 else self.hidden_dim,
                    self.hidden_dim,
                )
                for layer_idx in range(self.n_layers)
            ]
        )

    """
    _locked_mask
    Sample an inverted-dropout mask that is held fixed across time.

    INPUT:
        - reference: torch.Tensor -> [batch, features] tensor defining mask shape

    OUTPUT:
        - mask: torch.Tensor -> multiplicative mask [batch, features]
    """
    def _locked_mask(self, reference):
        if not self.training or self.variational_dropout == 0.0:
            return torch.ones_like(reference)
        # end if recurrent dropout is disabled

        keep_probability = 1.0 - self.variational_dropout
        mask = torch.empty_like(reference).bernoulli_(keep_probability)
        return mask / keep_probability
    # EOF

    """
    forward
    Unroll every GRU layer while retaining every final-layer hidden state.

    INPUT:
        - sequence: torch.Tensor -> neural activity [batch, time, input_dim]

    OUTPUT:
        - hidden_sequence: torch.Tensor -> [batch, time, hidden_dim]
    """
    def forward(self, sequence):
        if sequence.ndim != 3 or sequence.shape[-1] != self.input_dim:
            raise ValueError(
                "Expected sequence shape [batch, time, input_dim] with "
                f"input_dim={self.input_dim}, got {tuple(sequence.shape)}."
            )
        # end if the sequence shape is invalid

        layer_sequence = sequence
        for cell in self.cells:
            hidden_state = sequence.new_zeros(
                sequence.shape[0],
                self.hidden_dim,
            )
            input_mask = self._locked_mask(layer_sequence[:, 0])
            recurrent_mask = self._locked_mask(hidden_state)
            hidden_states = []

            for time_idx in range(sequence.shape[1]):
                # Drop the same input and recurrent coordinates at every step,
                # but retain the undropped state as the layer output/value.
                cell_input = layer_sequence[:, time_idx] * input_mask
                recurrent_input = hidden_state * recurrent_mask
                hidden_state = cell(cell_input, recurrent_input)
                hidden_states.append(hidden_state)
            # end for neural time point

            layer_sequence = torch.stack(hidden_states, dim=1)
        # end for recurrent layer
        return layer_sequence
    # EOF
# EOC


class NeuralToLayerFeatureGRUModel(nn.Module):
    """
    Predict several static ANN-layer targets from one neural time series.

    A small variational-dropout GRU returns every hidden state. Each target
    layer owns an independent learned query that attends across those shared
    states, followed by its own deterministic regression head.

    INPUT (forward):
        - neural_sequence: torch.Tensor -> [batch, time, neural_channels]
        - return_hidden_states: bool -> whether to return the GRU sequence

    OUTPUT:
        - predictions: torch.Tensor -> concatenated PCA targets [batch, sum(D_l)]
        - attention_weights: torch.Tensor -> [batch, target_layers, time]
        - hidden_states: torch.Tensor -> [batch, time, hidden_dim], only when
          return_hidden_states is True
    """

    """
    __init__
    Build the optional neural bottleneck, recurrent encoder, per-layer queries,
    and independent deterministic regression heads.

    INPUT:
        - n_neural_channels: int -> number of channels at each neural time point
        - target_dims: list[int] -> PCA target dimension for each ANN layer
        - target_names: list[str] | None -> ordered interpretable layer names
        - hidden_dim: int -> small GRU state width
        - n_gru_layers: int -> one initially; two only after validation support
        - bottleneck_dim: int | None -> optional channel projection width
        - variational_dropout: float -> locked GRU input/recurrent dropout
        - hidden_dropout: float -> ordinary dropout before attention pooling
        - head_dropout: float -> dropout applied before every regression head
        - head_type: str -> either "linear" or one-hidden-layer "mlp"
        - head_hidden_dim: int | None -> MLP hidden width; defaults to hidden_dim

    OUTPUT:
        - None
    """
    def __init__(
        self,
        n_neural_channels,
        target_dims,
        target_names=None,
        hidden_dim=16,
        n_gru_layers=1,
        bottleneck_dim=None,
        variational_dropout=0.15,
        hidden_dropout=0.25,
        head_dropout=0.25,
        head_type="linear",
        head_hidden_dim=None,
    ):
        super().__init__()
        target_dims = [int(target_dim) for target_dim in target_dims]
        if n_neural_channels <= 0 or not target_dims or min(target_dims) <= 0:
            raise ValueError("Neural and target dimensions must be positive.")
        # end if an input or output dimension is invalid
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        # end if the recurrent state width is invalid
        if head_type not in {"linear", "mlp"}:
            raise ValueError("head_type must be either 'linear' or 'mlp'.")
        # end if the regression-head type is unsupported
        for dropout_name, dropout_value in (
            ("hidden_dropout", hidden_dropout),
            ("head_dropout", head_dropout),
        ):
            if not 0.0 <= dropout_value < 1.0:
                raise ValueError(f"{dropout_name} must lie in [0, 1).")
            # end if a standard dropout value is invalid
        # end for standard dropout setting

        self.n_neural_channels = int(n_neural_channels)
        self.target_dims = target_dims
        self.hidden_dim = int(hidden_dim)
        self.head_type = head_type
        self.target_names = (
            [f"layer_{idx}" for idx in range(len(target_dims))]
            if target_names is None
            else list(target_names)
        )
        if len(self.target_names) != len(target_dims):
            raise ValueError("target_names and target_dims must have equal length.")
        # end if target metadata do not align

        if bottleneck_dim is None:
            self.input_bottleneck = nn.Identity()
            recurrent_input_dim = self.n_neural_channels
        else:
            if bottleneck_dim <= 0:
                raise ValueError("bottleneck_dim must be positive or None.")
            # end if the optional bottleneck width is invalid
            recurrent_input_dim = int(bottleneck_dim)
            self.input_bottleneck = nn.Sequential(
                nn.Linear(self.n_neural_channels, recurrent_input_dim),
                nn.LayerNorm(recurrent_input_dim),
                nn.Dropout(hidden_dropout),
            )
        # end if a neural-channel bottleneck is requested

        self.encoder = VariationalGRUEncoder(
            input_dim=recurrent_input_dim,
            hidden_dim=self.hidden_dim,
            n_layers=n_gru_layers,
            variational_dropout=variational_dropout,
        )
        self.hidden_sequence_dropout = nn.Dropout(hidden_dropout)
        self.layer_queries = nn.Parameter(
            torch.empty(len(target_dims), self.hidden_dim)
        )
        nn.init.normal_(self.layer_queries, mean=0.0, std=0.02)

        head_hidden_dim = (
            self.hidden_dim if head_hidden_dim is None else int(head_hidden_dim)
        )
        if head_hidden_dim <= 0:
            raise ValueError("head_hidden_dim must be positive.")
        # end if the MLP hidden width is invalid
        self.regression_heads = nn.ModuleList()
        for target_dim in target_dims:
            if head_type == "linear":
                head = nn.Sequential(
                    nn.Dropout(head_dropout),
                    nn.Linear(self.hidden_dim, target_dim),
                )
            else:
                head = nn.Sequential(
                    nn.Dropout(head_dropout),
                    nn.Linear(self.hidden_dim, head_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(head_dropout),
                    nn.Linear(head_hidden_dim, target_dim),
                )
            # end if a linear or MLP head is requested
            self.regression_heads.append(head)
        # end for target-layer regression head

    @property
    def device(self):
        return self.layer_queries.device
    # EOF

    def get_target_slices(self):
        target_slices = []
        start = 0
        for target_dim in self.target_dims:
            target_slices.append(slice(start, start + target_dim))
            start += target_dim
        # end for target PCA width
        return target_slices
    # EOF

    def get_trainable_parameters(self):
        return self.parameters()
    # EOF

    """
    _build_attended_states
    Turn the GRU sequence into the items that the layer queries attend over.

    INPUT:
        - hidden_states: torch.Tensor -> GRU states [batch, time, hidden_dim]

    OUTPUT:
        - attended_states: torch.Tensor -> attended items [batch, items, hidden_dim]
    """
    def _build_attended_states(self, hidden_states):
        # Standard output dropout is separate from the GRU's locked masks.
        return self.hidden_sequence_dropout(hidden_states)
    # EOF

    """
    forward
    Encode all neural bins, attend independently for each ANN layer, and regress
    each pooled representation to that layer's PCA coordinates.

    INPUT:
        - neural_sequence: torch.Tensor -> [batch, time, neural_channels]
        - return_hidden_states: bool -> whether to include all GRU states

    OUTPUT:
        - predictions: torch.Tensor -> [batch, sum(D_l)]
        - attention_weights: torch.Tensor -> [batch, target_layers, time]
        - hidden_states: torch.Tensor -> optional [batch, time, hidden_dim]
    """
    def forward(self, neural_sequence, return_hidden_states=False):
        if neural_sequence.ndim != 3:
            raise ValueError(
                "neural_sequence must have shape [batch, time, channels]."
            )
        # end if the neural input does not have three axes
        if neural_sequence.shape[-1] != self.n_neural_channels:
            raise ValueError(
                f"Expected {self.n_neural_channels} neural channels, got "
                f"{neural_sequence.shape[-1]}."
            )
        # end if the neural channel count is wrong

        recurrent_inputs = self.input_bottleneck(neural_sequence)
        hidden_states = self.encoder(recurrent_inputs)

        attended_states = self._build_attended_states(hidden_states)
        scores = torch.einsum(
            "lh,bth->blt",
            self.layer_queries,
            attended_states,
        ) / math.sqrt(self.hidden_dim)
        attention_weights = torch.softmax(scores, dim=-1)
        pooled_states = torch.einsum(
            "blt,bth->blh",
            attention_weights,
            attended_states,
        )

        layer_predictions = [
            head(pooled_states[:, layer_idx])
            for layer_idx, head in enumerate(self.regression_heads)
        ]
        predictions = torch.cat(layer_predictions, dim=-1)
        if return_hidden_states:
            return predictions, attention_weights, hidden_states
        # end if recurrent states were requested
        return predictions, attention_weights
    # EOF
# EOC


class NoiseStateGRUFeatureModel(NeuralToLayerFeatureGRUModel):
    """
    NeuralToLayerFeatureGRUModel whose layer queries also attend to one noise state.

    This is the neural-to-feature analogue of NoiseLayerBaselineModel. There the
    attention axis is ANN layers, so the noise is one extra layer; here the
    attention axis is neural time, so the noise is one extra pseudo-state
    appended after the last GRU state. It is drawn uniformly from the
    hidden_dim sphere, independently for every presentation and forward pass,
    and carries no neural information. Attention mass spent on it is therefore
    a direct measure of how strongly a layer query rejects uninformative items.

    INPUT (forward):
        - neural_sequence: torch.Tensor -> [batch, time, neural_channels]
        - return_hidden_states: bool -> whether to return the GRU sequence

    OUTPUT:
        - predictions: torch.Tensor -> concatenated PCA targets [batch, sum(D_l)]
        - attention_weights: torch.Tensor -> [batch, target_layers, time + 1],
          where the last item is the noise state
        - hidden_states: torch.Tensor -> real GRU states [batch, time, hidden_dim],
          only when return_hidden_states is True
    """

    """
    __init__
    Build the parent GRU decoder and store the noise-state options.

    INPUT:
        - match_noise_norm: bool -> rescale the sphere draw to the mean norm of
          that presentation's attended GRU states
        - noise_in_eval: bool -> keep sampling noise outside training mode;
          otherwise the noise item is a zero vector
        - decoder_kwargs: dict -> remaining NeuralToLayerFeatureGRUModel arguments

    OUTPUT:
        - None
    """
    def __init__(
        self,
        match_noise_norm=True,
        noise_in_eval=True,
        **decoder_kwargs,
    ):
        super().__init__(**decoder_kwargs)
        self.match_noise_norm = bool(match_noise_norm)
        self.noise_in_eval = bool(noise_in_eval)

    """
    _build_attended_states
    Append one noise pseudo-state after the dropped-out GRU states.

    INPUT:
        - hidden_states: torch.Tensor -> GRU states [batch, time, hidden_dim]

    OUTPUT:
        - attended_states: torch.Tensor -> [batch, time + 1, hidden_dim]
    """
    def _build_attended_states(self, hidden_states):
        attended_states = super()._build_attended_states(hidden_states)
        batch_size = attended_states.shape[0]
        if self.training or self.noise_in_eval:
            # A normalized isotropic Gaussian draw is uniform on the unit sphere.
            noise_state = torch.randn(
                batch_size,
                1,
                self.hidden_dim,
                device=attended_states.device,
                dtype=attended_states.dtype,
            )
            noise_state = noise_state / noise_state.norm(
                dim=-1, keepdim=True
            ).clamp_min(torch.finfo(noise_state.dtype).eps)

            if self.match_noise_norm:
                # Match the average state norm so the noise competes on equal
                # footing in the dot-product scores and in the pooled vector.
                reference_norm = attended_states.norm(dim=-1).mean(dim=1)
                noise_state = noise_state * reference_norm[:, None, None]
            # end if the noise state is scale-matched to the GRU states
        else:
            # A zero vector keeps the extra attended item but removes randomness.
            noise_state = attended_states.new_zeros(batch_size, 1, self.hidden_dim)
        # end if the noise state is sampled or silenced
        return torch.cat([attended_states, noise_state], dim=1)
    # EOF
# EOC
