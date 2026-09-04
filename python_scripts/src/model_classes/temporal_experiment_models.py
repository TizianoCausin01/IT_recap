import math

import torch
from torch import nn


class CachedFeatureTemporalModel(nn.Module):
    """Shared validation and normalization for cached layer-feature decoders."""

    def __init__(self, n_layers, feature_dim, n_timepoints, n_neurons):
        super().__init__()
        dimensions = (n_layers, feature_dim, n_timepoints, n_neurons)
        if min(dimensions) <= 0:
            raise ValueError("All cached-feature model dimensions must be positive.")
        # end if a model dimension is invalid

        self.n_layers = n_layers
        self.feature_dim = feature_dim
        self.n_timepoints = n_timepoints
        self.n_neurons = n_neurons
        self.feature_norm = nn.LayerNorm(feature_dim, elementwise_affine=False)

    def _validate_features(self, layer_features):
        expected_shape = (self.n_layers, self.feature_dim)
        if (
            layer_features.ndim != 3
            or tuple(layer_features.shape[1:]) != expected_shape
        ):
            raise ValueError(
                "Expected [batch, layers, embedding] with trailing shape "
                f"{expected_shape}, got {tuple(layer_features.shape)}."
            )
        # end if cached features have the wrong shape
        return layer_features
    # EOF

    def _normalized_features(self, layer_features):
        return self.feature_norm(self._validate_features(layer_features))
    # EOF
# EOC


class LowRankLinearDecoder(CachedFeatureTemporalModel):
    """Factorize the static-feature to complete-response linear mapping."""

    def __init__(
        self,
        n_layers,
        feature_dim,
        n_timepoints,
        n_neurons,
        hidden_dim,
        **_,
    ):
        super().__init__(n_layers, feature_dim, n_timepoints, n_neurons)
        input_dim = n_layers * feature_dim
        output_dim = n_timepoints * n_neurons
        self.input_projection = nn.Linear(input_dim, hidden_dim, bias=False)
        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, layer_features, return_diagnostics=False):
        normalized = self._normalized_features(layer_features)
        latent = self.input_projection(normalized.flatten(start_dim=1))
        predictions = self.output_projection(latent).reshape(
            -1,
            self.n_timepoints,
            self.n_neurons,
        )
        diagnostics = {"latent": latent} if return_diagnostics else None
        return predictions, diagnostics
    # EOF
# EOC


class TimeConditionedMLP(CachedFeatureTemporalModel):
    """Predict bins independently with shared weights and learned time embeddings."""

    def __init__(
        self,
        n_layers,
        feature_dim,
        n_timepoints,
        n_neurons,
        hidden_dim,
        time_embedding_dim,
        dropout,
        **_,
    ):
        super().__init__(n_layers, feature_dim, n_timepoints, n_neurons)
        self.image_projection = nn.Sequential(
            nn.Linear(n_layers * feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.time_embeddings = nn.Parameter(
            torch.randn(n_timepoints, time_embedding_dim) * 0.02
        )
        self.temporal_mlp = nn.Sequential(
            nn.Linear(hidden_dim + time_embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_neurons),
        )

    def forward(self, layer_features, return_diagnostics=False):
        normalized = self._normalized_features(layer_features)
        image_features = self.image_projection(normalized.flatten(start_dim=1))
        batch_size = image_features.shape[0]
        repeated_image_features = image_features.unsqueeze(1).expand(
            -1, self.n_timepoints, -1
        )
        time_features = self.time_embeddings.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        predictions = self.temporal_mlp(
            torch.cat([repeated_image_features, time_features], dim=-1)
        )
        diagnostics = {"latent": image_features} if return_diagnostics else None
        return predictions, diagnostics
    # EOF
# EOC


class AutonomousGRUDecoder(CachedFeatureTemporalModel):
    """Initialize a recurrent state from the image and evolve using time only."""

    def __init__(
        self,
        n_layers,
        feature_dim,
        n_timepoints,
        n_neurons,
        hidden_dim,
        time_embedding_dim,
        dropout,
        **_,
    ):
        super().__init__(n_layers, feature_dim, n_timepoints, n_neurons)
        self.initial_state = nn.Sequential(
            nn.Linear(n_layers * feature_dim, hidden_dim),
            nn.Tanh(),
        )
        self.time_embeddings = nn.Parameter(
            torch.randn(n_timepoints, time_embedding_dim) * 0.02
        )
        self.recurrence = nn.GRU(
            input_size=time_embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.readout = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_neurons),
        )

    def forward(self, layer_features, return_diagnostics=False):
        normalized = self._normalized_features(layer_features)
        initial_state = self.initial_state(
            normalized.flatten(start_dim=1)
        ).unsqueeze(0)
        time_features = self.time_embeddings.unsqueeze(0).expand(
            layer_features.shape[0], -1, -1
        )
        hidden_sequence, _ = self.recurrence(time_features, initial_state)
        predictions = self.readout(hidden_sequence)
        diagnostics = (
            {"hidden_sequence": hidden_sequence}
            if return_diagnostics
            else None
        )
        return predictions, diagnostics
    # EOF
# EOC


class RepeatedInputGRUDecoder(CachedFeatureTemporalModel):
    """Feed a compressed static image representation into a GRU at every bin."""

    def __init__(
        self,
        n_layers,
        feature_dim,
        n_timepoints,
        n_neurons,
        hidden_dim,
        time_embedding_dim,
        dropout,
        **_,
    ):
        super().__init__(n_layers, feature_dim, n_timepoints, n_neurons)
        self.image_projection = nn.Sequential(
            nn.Linear(n_layers * feature_dim, hidden_dim),
            nn.GELU(),
        )
        self.initial_state = nn.Linear(hidden_dim, hidden_dim)
        self.time_embeddings = nn.Parameter(
            torch.randn(n_timepoints, time_embedding_dim) * 0.02
        )
        self.recurrence = nn.GRU(
            input_size=hidden_dim + time_embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.readout = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_neurons),
        )

    def forward(self, layer_features, return_diagnostics=False):
        normalized = self._normalized_features(layer_features)
        image_features = self.image_projection(normalized.flatten(start_dim=1))
        repeated_image_features = image_features.unsqueeze(1).expand(
            -1, self.n_timepoints, -1
        )
        time_features = self.time_embeddings.unsqueeze(0).expand(
            layer_features.shape[0], -1, -1
        )
        recurrent_inputs = torch.cat(
            [repeated_image_features, time_features], dim=-1
        )
        initial_state = torch.tanh(self.initial_state(image_features)).unsqueeze(0)
        hidden_sequence, _ = self.recurrence(recurrent_inputs, initial_state)
        predictions = self.readout(hidden_sequence)
        diagnostics = (
            {"hidden_sequence": hidden_sequence}
            if return_diagnostics
            else None
        )
        return predictions, diagnostics
    # EOF
# EOC


class RecurrentLayerAttentionDecoder(CachedFeatureTemporalModel):
    """Condition layer attention on both time and the previous recurrent state."""

    def __init__(
        self,
        n_layers,
        feature_dim,
        n_timepoints,
        n_neurons,
        hidden_dim,
        time_embedding_dim,
        attention_dim,
        dropout,
        **_,
    ):
        super().__init__(n_layers, feature_dim, n_timepoints, n_neurons)
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.time_embeddings = nn.Parameter(
            torch.randn(n_timepoints, time_embedding_dim) * 0.02
        )
        self.key_projection = nn.Linear(feature_dim, attention_dim, bias=False)
        self.value_projection = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.query_projection = nn.Linear(
            hidden_dim + time_embedding_dim,
            attention_dim,
            bias=False,
        )
        self.initial_state = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
        )
        self.recurrence = nn.GRUCell(
            input_size=hidden_dim + time_embedding_dim,
            hidden_size=hidden_dim,
        )
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # Let the neural readout distinguish absolute latency without requiring
        # the recurrent state to preserve an exact clock on this small dataset.
        self.readout = nn.Linear(
            hidden_dim + time_embedding_dim,
            n_neurons,
        )

    def forward(self, layer_features, return_diagnostics=False):
        normalized = self._normalized_features(layer_features)
        keys = self.key_projection(normalized)
        values = self.value_projection(normalized)
        hidden_state = self.initial_state(normalized.mean(dim=1))

        predictions = []
        attention_sequence = []
        hidden_sequence = []
        for time_idx in range(self.n_timepoints):
            time_features = self.time_embeddings[time_idx].unsqueeze(0).expand(
                layer_features.shape[0], -1
            )
            query = self.query_projection(
                torch.cat([hidden_state, time_features], dim=-1)
            )
            attention_logits = torch.einsum("bk,blk->bl", query, keys)
            attention_logits = attention_logits / math.sqrt(self.attention_dim)
            attention_weights = torch.softmax(attention_logits, dim=-1)
            context = torch.einsum("bl,blh->bh", attention_weights, values)
            recurrent_input = torch.cat(
                [self.context_norm(context), time_features], dim=-1
            )
            hidden_state = self.recurrence(recurrent_input, hidden_state)
            readout_features = torch.cat(
                [self.dropout(hidden_state), time_features], dim=-1
            )
            predictions.append(self.readout(readout_features))
            if return_diagnostics:
                attention_sequence.append(attention_weights)
                hidden_sequence.append(hidden_state)
            # end if diagnostics are requested
        # end for temporal bin

        predictions = torch.stack(predictions, dim=1)
        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                "attention": torch.stack(attention_sequence, dim=1),
                "hidden_sequence": torch.stack(hidden_sequence, dim=1),
            }
        # end if diagnostics are requested
        return predictions, diagnostics
    # EOF
# EOC


class RecurrentFeatureAttentionDecoder(CachedFeatureTemporalModel):
    """Use previous state and time to select individual DINO coordinates."""

    def __init__(
        self,
        n_layers,
        feature_dim,
        n_timepoints,
        n_neurons,
        hidden_dim,
        time_embedding_dim,
        attention_dim,
        dropout,
        attention_mode="softmax",
        **_,
    ):
        super().__init__(n_layers, feature_dim, n_timepoints, n_neurons)
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.n_attention_features = n_layers * feature_dim
        if attention_mode not in {"softmax", "sigmoid"}:
            raise ValueError("attention_mode must be 'softmax' or 'sigmoid'.")
        # end if the feature-gating rule is invalid
        self.attention_mode = attention_mode
        self.time_embeddings = nn.Parameter(
            torch.randn(n_timepoints, time_embedding_dim) * 0.02
        )
        self.feature_key_embeddings = nn.Parameter(
            torch.randn(n_layers, feature_dim, attention_dim) * 0.02
        )
        self.query_projection = nn.Linear(
            hidden_dim + time_embedding_dim,
            attention_dim,
            bias=False,
        )
        self.value_projection = nn.Linear(
            self.n_attention_features,
            hidden_dim,
            bias=False,
        )
        self.initial_state = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
        )
        self.recurrence = nn.GRUCell(
            input_size=hidden_dim + time_embedding_dim,
            hidden_size=hidden_dim,
        )
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(
            hidden_dim + time_embedding_dim,
            n_neurons,
        )

    def forward(self, layer_features, return_diagnostics=False):
        normalized = self._normalized_features(layer_features)
        hidden_state = self.initial_state(normalized.mean(dim=1))
        predictions = []
        attention_sequence = []
        hidden_sequence = []

        for time_idx in range(self.n_timepoints):
            time_features = self.time_embeddings[time_idx].unsqueeze(0).expand(
                layer_features.shape[0], -1
            )
            query = self.query_projection(
                torch.cat([hidden_state, time_features], dim=-1)
            )

            # Coordinate keys are stimulus-modulated as in the notebook's
            # feature-attention baseline, but the query also sees h_(t-1).
            attention_logits = torch.einsum(
                "bk,lek->ble",
                query,
                self.feature_key_embeddings,
            )
            attention_logits = attention_logits * normalized
            attention_logits = attention_logits / math.sqrt(self.attention_dim)
            flat_logits = attention_logits.flatten(start_dim=1)
            if self.attention_mode == "softmax":
                flat_attention = torch.softmax(flat_logits, dim=-1)
                attention_scale = self.n_attention_features
            else:
                # Independent sigmoid gates avoid forcing 4096 DINO coordinates
                # to compete for a fixed unit of attention mass. Multiplication
                # by two makes the near-0 logits start as identity gates.
                flat_attention = torch.sigmoid(flat_logits)
                attention_scale = 2.0
            # end if competitive or independent feature gates are requested

            # Preserve feature scale when the initial gate is nearly uniform.
            gated_features = (
                normalized.flatten(start_dim=1)
                * flat_attention
                * attention_scale
            )
            context = self.value_projection(gated_features)
            recurrent_input = torch.cat(
                [self.context_norm(context), time_features], dim=-1
            )
            hidden_state = self.recurrence(recurrent_input, hidden_state)
            readout_features = torch.cat(
                [self.dropout(hidden_state), time_features], dim=-1
            )
            predictions.append(self.readout(readout_features))
            if return_diagnostics:
                attention_sequence.append(
                    flat_attention.reshape(
                        -1, self.n_layers, self.feature_dim
                    )
                )
                hidden_sequence.append(hidden_state)
            # end if diagnostics are requested
        # end for temporal bin

        predictions = torch.stack(predictions, dim=1)
        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                "attention": torch.stack(attention_sequence, dim=1),
                "hidden_sequence": torch.stack(hidden_sequence, dim=1),
            }
        # end if diagnostics are requested
        return predictions, diagnostics
    # EOF
# EOC


class RecurrentSigmoidFeatureAttentionDecoder(
    RecurrentFeatureAttentionDecoder
):
    """Recurrent hidden/time-conditioned feature selection with independent gates."""

    def __init__(self, **model_kwargs):
        super().__init__(attention_mode="sigmoid", **model_kwargs)
    # EOF
# EOC


class TimeLocalFeatureAttentionDecoder(CachedFeatureTemporalModel):
    """Cached-feature version of the notebook's non-recurrent decoder."""

    def __init__(
        self,
        n_layers,
        feature_dim,
        n_timepoints,
        n_neurons,
        hidden_dim,
        time_embedding_dim,
        attention_dim,
        dropout,
        **_,
    ):
        super().__init__(n_layers, feature_dim, n_timepoints, n_neurons)
        self.n_attention_features = n_layers * feature_dim
        self.time_embeddings = nn.Parameter(
            torch.randn(n_timepoints, time_embedding_dim) * 0.02
        )
        self.query_projection = nn.Linear(
            time_embedding_dim,
            attention_dim,
            bias=False,
        )
        self.feature_key_embeddings = nn.Parameter(
            torch.randn(n_layers, feature_dim, attention_dim) * 0.02
        )
        self.value_projection = nn.Linear(
            self.n_attention_features,
            hidden_dim,
            bias=False,
        )
        self.value_norm = nn.LayerNorm(hidden_dim)
        self.temporal_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.timebin_readouts = nn.ModuleList(
            [nn.Linear(hidden_dim, n_neurons) for _ in range(n_timepoints)]
        )

    def forward(self, layer_features, return_diagnostics=False):
        normalized = self._normalized_features(layer_features)
        queries = self.query_projection(self.time_embeddings)
        attention_logits = torch.einsum(
            "tk,lek->tle", queries, self.feature_key_embeddings
        )
        attention_logits = attention_logits.unsqueeze(0) * normalized.unsqueeze(1)
        attention_logits = attention_logits / math.sqrt(queries.shape[-1])
        attention = torch.softmax(
            attention_logits.flatten(start_dim=-2), dim=-1
        ).reshape_as(attention_logits)
        gated_features = (
            normalized.unsqueeze(1)
            * attention
            * self.n_attention_features
        )
        latents = self.value_norm(
            self.value_projection(gated_features.flatten(start_dim=-2))
        )
        latents = self.temporal_mlp(latents)

        predictions = []
        for time_idx, readout in enumerate(self.timebin_readouts):
            predictions.append(readout(latents[:, time_idx]))
        # end for time-bin readout
        predictions = torch.stack(predictions, dim=1)
        diagnostics = {"attention": attention} if return_diagnostics else None
        return predictions, diagnostics
    # EOF
# EOC


class TemporalTransformerDecoder(CachedFeatureTemporalModel):
    """Cross-attend temporal queries to layers, then mix bins by self-attention."""

    def __init__(
        self,
        n_layers,
        feature_dim,
        n_timepoints,
        n_neurons,
        hidden_dim,
        dropout,
        n_attention_heads,
        **_,
    ):
        super().__init__(n_layers, feature_dim, n_timepoints, n_neurons)
        if hidden_dim % n_attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_attention_heads.")
        # end if attention heads do not divide the hidden width
        self.layer_projection = nn.Linear(feature_dim, hidden_dim)
        self.image_projection = nn.Linear(feature_dim, hidden_dim)
        self.time_embeddings = nn.Parameter(
            torch.randn(n_timepoints, hidden_dim) * 0.02
        )
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            n_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_attention_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.readout = nn.Linear(hidden_dim, n_neurons)

    def forward(self, layer_features, return_diagnostics=False):
        normalized = self._normalized_features(layer_features)
        layer_memory = self.layer_projection(normalized)
        image_features = self.image_projection(normalized.mean(dim=1))
        queries = self.time_embeddings.unsqueeze(0).expand(
            layer_features.shape[0], -1, -1
        )
        queries = queries + image_features.unsqueeze(1)
        cross_attended, attention = self.cross_attention(
            queries,
            layer_memory,
            layer_memory,
            need_weights=return_diagnostics,
            average_attn_weights=True,
        )
        temporal_features = self.temporal_encoder(queries + cross_attended)
        predictions = self.readout(self.output_norm(temporal_features))
        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                "attention": attention,
                "hidden_sequence": temporal_features,
            }
        # end if diagnostics are requested
        return predictions, diagnostics
    # EOF
# EOC


MODEL_CLASSES = {
    "low_rank_linear": LowRankLinearDecoder,
    "time_mlp": TimeConditionedMLP,
    "autonomous_gru": AutonomousGRUDecoder,
    "repeated_gru": RepeatedInputGRUDecoder,
    "recurrent_layer_attention": RecurrentLayerAttentionDecoder,
    "recurrent_feature_attention": RecurrentFeatureAttentionDecoder,
    "recurrent_sigmoid_feature_attention": (
        RecurrentSigmoidFeatureAttentionDecoder
    ),
    "time_local_feature_attention": TimeLocalFeatureAttentionDecoder,
    "temporal_transformer": TemporalTransformerDecoder,
}


"""
build_temporal_experiment_model
Construct one cached-feature temporal decoder from a shared configuration.

INPUT:
    - model_name: str -> key in MODEL_CLASSES
    - model_kwargs: dict -> shared data-shape and architecture parameters

OUTPUT:
    - model: nn.Module -> requested temporal neural prediction model
"""
def build_temporal_experiment_model(model_name, **model_kwargs):
    if model_name not in MODEL_CLASSES:
        raise KeyError(
            f"Unknown model {model_name!r}; choose from {list(MODEL_CLASSES)}."
        )
    # end if the requested architecture is unavailable
    return MODEL_CLASSES[model_name](**model_kwargs)
# EOF
