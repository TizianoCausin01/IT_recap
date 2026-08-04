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
