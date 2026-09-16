"""
Decoders for window-averaged TVSD population responses.

The models in ``temporal_experiment_models`` predict a complete time course.
Here the neural target is collapsed to a single vector -- the mean MUA over the
high-SNR response window -- so the decoder carries no temporal axis at all:
cached DINOv3 layer tokens exchange information by self-attention and one MLP
reads the whole population out at once.

The prediction is still returned with shape [batch, 1, neurons] so that every
scoring utility written for the time-resolved decoders (stimulus_correlation,
aggregate_trials_by_image, split_half_reliability, score_predictions) applies
unchanged, with a time axis of length one.
"""

import torch
from torch import nn

from model_classes.temporal_experiment_models import CachedFeatureTemporalModel


class LayerAttentionPopulationDecoder(CachedFeatureTemporalModel):
    """
    Predict one window-averaged population vector from cached ANN layer tokens.

    Each selected ANN depth becomes one token. A single self-attention block is
    the only place where depths interact; the pooled tokens then pass through
    one MLP that emits the full population vector.

    INPUT (forward):
        - layer_features: torch.Tensor -> cached features [batch, layers, embedding]
        - return_diagnostics: bool -> whether to return attention and tokens

    OUTPUT:
        - predictions: torch.Tensor -> population activity [batch, 1, neurons]
        - diagnostics: dict | None -> attention [batch, layers, layers] and tokens
    """

    """
    __init__
    Build the per-depth projections, the self-attention block, and the readout MLP.

    INPUT:
        - n_layers: int -> number of cached ANN depths used as tokens
        - feature_dim: int -> cached embedding width of every depth
        - n_neurons: int -> number of neural output channels
        - hidden_dim: int -> shared token width used by self-attention
        - n_attention_heads: int -> attention heads; must divide hidden_dim
        - mlp_hidden_dim: int -> hidden width of the readout MLP
        - dropout: float -> dropout probability in attention and readout
        - readout_pooling: str -> "concat" or "mean" over the attended tokens

    OUTPUT:
        - None
    """
    def __init__(
        self,
        n_layers,
        feature_dim,
        n_neurons,
        hidden_dim,
        n_attention_heads,
        mlp_hidden_dim,
        dropout=0.0,
        readout_pooling="concat",
        **_,
    ):
        # One output bin: the window average is a single population vector.
        super().__init__(n_layers, feature_dim, 1, n_neurons)
        if hidden_dim % n_attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_attention_heads.")
        # end if attention heads do not divide the token width
        if readout_pooling not in {"concat", "mean"}:
            raise ValueError("readout_pooling must be either 'concat' or 'mean'.")
        # end if the token pooling is unsupported
        if mlp_hidden_dim <= 0 or not 0.0 <= dropout < 1.0:
            raise ValueError(
                "mlp_hidden_dim must be positive and dropout must lie in [0, 1)."
            )
        # end if the readout configuration is invalid

        self.hidden_dim = hidden_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.readout_pooling = readout_pooling

        # Different DINOv3 depths do not share a coordinate system, so every
        # token gets its own projection into the common attention width.
        self.layer_projections = nn.ModuleList(
            [nn.Linear(feature_dim, hidden_dim) for _ in range(n_layers)]
        )

        # Depth identity is the only "position" available here; without it the
        # self-attention block would be permutation invariant over the tokens.
        self.layer_embeddings = nn.Parameter(
            torch.randn(n_layers, hidden_dim) * 0.02
        )

        # Pre-norm attention, as in the temporal transformer decoder.
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim,
            n_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.readout_norm = nn.LayerNorm(hidden_dim)

        # Concatenation keeps depth-specific structure that a mean would erase;
        # with only a handful of tokens the extra readout width is affordable.
        readout_input_dim = (
            hidden_dim * n_layers if readout_pooling == "concat" else hidden_dim
        )
        self.readout = nn.Sequential(
            nn.Linear(readout_input_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, n_neurons),
        )

    def forward(self, layer_features, return_diagnostics=False):
        # LayerNorm over the embedding axis puts every depth on a common scale.
        normalized = self._normalized_features(layer_features)

        # Project each depth with its own weights, then restack to [B, L, D].
        tokens = torch.stack(
            [
                projection(normalized[:, layer_index])
                for layer_index, projection in enumerate(self.layer_projections)
            ],
            dim=1,
        )
        tokens = tokens + self.layer_embeddings.unsqueeze(0)

        # The only interaction between depths happens in this block.
        attention_input = self.attention_norm(tokens)
        attended, attention = self.self_attention(
            attention_input,
            attention_input,
            attention_input,
            need_weights=return_diagnostics,
            average_attn_weights=True,
        )
        tokens = self.readout_norm(tokens + attended)

        pooled = (
            tokens.flatten(start_dim=1)
            if self.readout_pooling == "concat"
            else tokens.mean(dim=1)
        )

        # [B, neurons] -> [B, 1, neurons] keeps the [batch, time, neurons]
        # contract shared with the time-resolved decoders.
        predictions = self.readout(pooled).unsqueeze(1)

        diagnostics = None
        if return_diagnostics:
            diagnostics = {"attention": attention, "tokens": tokens}
        # end if diagnostics are requested
        return predictions, diagnostics
    # EOF
# EOC
