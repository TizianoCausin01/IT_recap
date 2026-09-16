"""
Decoders that read a time-resolved neural population response and predict
several frozen ANN layer representations of the viewed image at once.

Every other model class in this project runs the encoding direction -- image
features in, neural response out. These run the decoding direction, and the
asymmetry matters for the design: the neural time series is the only genuine
sequence here, while the ANN layers are independent regression targets that
happen to share an encoder.

The encoder is a small GRU that keeps *every* hidden state. Each target layer
then owns a free query vector and pools that hidden-state sequence with its own
softmax attention over time, so the three targets are read off the same
dynamics through three different temporal weightings. Those weightings are the
scientific output as much as the regression scores are: they say which part of
the response a shallow ANN layer is recoverable from and which part a deep one
is.

Capacity is deliberately tiny. With ~20k single-repetition presentations and a
few hundred channels, hidden widths in the usual recurrent-network range fit
trial noise long before they run out of parameters, so the defaults here sit in
the 8-32 range and dropout is variational rather than per-step.
"""

import torch
from torch import nn


class VariationalDropoutGRU(nn.Module):
    """
    A GRU that returns the whole hidden-state sequence, with Gal-Ghahramani
    dropout: one mask per sequence, reused at every time step, on both the
    layer input and the recurrent state.

    `nn.GRU` would be faster, but it returns a dropout pattern that is resampled
    at every step and -- more importantly -- it is built around the final state.
    The hidden-state sequence is the substrate attention pooling needs, so it is
    the primary output here rather than an afterthought.

    INPUT (forward):
        - sequence: torch.Tensor -> [batch, time, input_dim]

    OUTPUT:
        - hidden_sequence: torch.Tensor -> [batch, time, hidden_dim]
    """

    """
    __init__
    Build the per-layer GRU cells and record the dropout rates.

    INPUT:
        - input_dim: int -> channels (or input PCs) entering the encoder
        - hidden_dim: int -> width of the recurrent state
        - n_layers: int -> stacked GRU layers
        - dropout: float -> variational dropout on every layer input
        - recurrent_dropout: float -> variational dropout on the recurrent state

    OUTPUT:
        - None
    """
    def __init__(
        self,
        input_dim,
        hidden_dim,
        n_layers=1,
        dropout=0.0,
        recurrent_dropout=0.0,
    ):
        super().__init__()
        if min(input_dim, hidden_dim, n_layers) <= 0:
            raise ValueError("input_dim, hidden_dim, and n_layers must be positive.")
        # end if an encoder dimension is invalid
        for rate in (dropout, recurrent_dropout):
            if not 0.0 <= rate < 1.0:
                raise ValueError("dropout rates must lie in [0, 1).")
            # end if a dropout rate is invalid
        # end for dropout rate

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = float(dropout)
        self.recurrent_dropout = float(recurrent_dropout)
        self.cells = nn.ModuleList(
            nn.GRUCell(input_dim if layer_idx == 0 else hidden_dim, hidden_dim)
            for layer_idx in range(n_layers)
        )

    """
    _sample_mask
    Draw one inverted-dropout mask shared by every time step of a sequence.

    INPUT:
        - reference: torch.Tensor -> tensor giving batch size, dtype, and device
        - width: int -> feature width the mask covers
        - rate: float -> dropout probability

    OUTPUT:
        - mask: torch.Tensor | None -> [batch, width] mask, or None when inactive
    """
    def _sample_mask(self, reference, width, rate):
        if rate <= 0.0 or not self.training:
            return None
        # end if dropout is inactive
        keep_probability = 1.0 - rate
        mask = torch.bernoulli(
            torch.full(
                (reference.shape[0], width),
                keep_probability,
                device=reference.device,
                dtype=reference.dtype,
            )
        )
        # Inverted dropout keeps the expected activation unchanged at test time.
        return mask / keep_probability

    def forward(self, sequence):
        if sequence.ndim != 3:
            raise ValueError("sequence must have shape [batch, time, input_dim].")
        # end if the input axes are invalid
        batch_size, n_steps, _ = sequence.shape

        # One input mask per layer and one recurrent mask per layer, all fixed
        # for the whole sequence -- this is what makes the dropout variational.
        input_masks = [
            self._sample_mask(sequence, cell.input_size, self.dropout)
            for cell in self.cells
        ]
        hidden_masks = [
            self._sample_mask(sequence, self.hidden_dim, self.recurrent_dropout)
            for _ in self.cells
        ]
        hidden_states = [
            sequence.new_zeros(batch_size, self.hidden_dim)
            for _ in self.cells
        ]

        step_outputs = []
        for step in range(n_steps):
            layer_input = sequence[:, step]
            for layer_idx, cell in enumerate(self.cells):
                if input_masks[layer_idx] is not None:
                    layer_input = layer_input * input_masks[layer_idx]
                # end if this layer's input is dropped
                previous_state = hidden_states[layer_idx]
                if hidden_masks[layer_idx] is not None:
                    # The stored state stays clean; only the copy the gates see
                    # is dropped, which is the standard recurrent formulation.
                    previous_state = previous_state * hidden_masks[layer_idx]
                # end if the recurrent state is dropped
                hidden_states[layer_idx] = cell(layer_input, previous_state)
                layer_input = hidden_states[layer_idx]
            # end for stacked layer
            step_outputs.append(hidden_states[-1])
        # end for time step
        return torch.stack(step_outputs, dim=1)
# EOC


"""
build_regression_head
Build one deterministic head mapping a pooled code to a target layer's PCs.

Deterministic is not a style preference: every score in this experiment is a
conditional-mean quantity, so a sampling head would not be comparable to ridge
or to the repetition-ceiling extrapolation.

INPUT:
    - input_dim: int -> width of the pooled code
    - output_dim: int -> retained PCs of the target layer
    - head_hidden_dim: int | None -> hidden width, or None for a plain linear map
    - dropout: float -> dropout applied before the head

OUTPUT:
    - head: nn.Module -> pooled code to predicted PCs
"""
def build_regression_head(input_dim, output_dim, head_hidden_dim, dropout):
    if head_hidden_dim:
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, output_dim),
        )
    # end if the head has a hidden layer
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(input_dim, output_dim))
# EOF


class AttentionPooledNeuralDecoder(nn.Module):
    """
    Shared GRU encoder, one temporal pooling per target layer, one head each.

    `pooling="attention"` gives every target layer a free query vector and lets
    it build its own softmax distribution over neural time bins. `"mean"` and
    `"last"` replace all of those with a single shared pooled vector and are the
    ablations that say whether the per-target attention earns its parameters:
    if they score the same, the attention is decoration.

    Predictions come back as one concatenated tensor so the training loop can
    treat the multi-target problem as a single regression; `target_slices` says
    which columns belong to which layer.

    INPUT (forward):
        - sequence: torch.Tensor -> [batch, time, n_channels] neural response

    OUTPUT:
        - predictions: torch.Tensor -> [batch, sum of target dims]
        - diagnostics: dict -> "attention" [batch, targets, time] or None
    """

    """
    __init__
    Build the optional bottleneck, the recurrent encoder, the per-target
    queries, and the per-target heads.

    INPUT:
        - n_channels: int -> channels (or input PCs) per time bin
        - n_timepoints: int -> neural time bins entering the encoder
        - target_dims: sequence[int] -> retained PCs of each target layer
        - hidden_dim: int -> GRU width
        - n_layers: int -> stacked GRU layers
        - dropout: float -> variational input dropout, also used before the heads
        - recurrent_dropout: float -> variational dropout on the recurrent state
        - bottleneck_dim: int | None -> linear channel bottleneck before the GRU
        - head_hidden_dim: int | None -> hidden width of each regression head
        - pooling: str -> "attention", "mean", or "last"

    OUTPUT:
        - None
    """
    def __init__(
        self,
        n_channels,
        n_timepoints,
        target_dims,
        hidden_dim=16,
        n_layers=1,
        dropout=0.0,
        recurrent_dropout=0.0,
        bottleneck_dim=None,
        head_hidden_dim=None,
        pooling="attention",
    ):
        super().__init__()
        if pooling not in {"attention", "mean", "last"}:
            raise ValueError("pooling must be 'attention', 'mean', or 'last'.")
        # end if the pooling mode is unsupported
        target_dims = [int(dim) for dim in target_dims]
        if not target_dims or min(target_dims) <= 0:
            raise ValueError("target_dims must be a non-empty list of positive ints.")
        # end if the target specification is invalid

        self.pooling = pooling
        self.n_timepoints = int(n_timepoints)
        self.target_dims = target_dims
        # Column ranges of the concatenated prediction, one per target layer.
        boundaries = torch.tensor([0] + target_dims).cumsum(0).tolist()
        self.target_slices = [
            (boundaries[idx], boundaries[idx + 1]) for idx in range(len(target_dims))
        ]

        # An optional linear bottleneck with layer norm, in the spirit of the
        # small RNN decoders this design follows. With upstream input PCA it is
        # often redundant, which is why it is swept rather than assumed.
        if bottleneck_dim:
            self.bottleneck = nn.Sequential(
                nn.Linear(n_channels, bottleneck_dim),
                nn.LayerNorm(bottleneck_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            encoder_input_dim = bottleneck_dim
        else:
            self.bottleneck = None
            encoder_input_dim = n_channels
        # end if a channel bottleneck was requested

        self.encoder = VariationalDropoutGRU(
            encoder_input_dim,
            hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
        )
        self.hidden_dropout = nn.Dropout(dropout)

        # One free query per target layer. Small initial values keep the
        # scores near zero, so every layer starts from near-uniform
        # attention over time and sharpens only where the data ask for it.
        if pooling == "attention":
            self.queries = nn.Parameter(
                torch.randn(len(target_dims), hidden_dim) * 0.02
            )
        else:
            self.register_parameter("queries", None)
        # end if the model pools by attention

        self.heads = nn.ModuleList(
            build_regression_head(hidden_dim, dim, head_hidden_dim, dropout)
            for dim in target_dims
        )

    """
    ridge_like_parameters
    The weight matrices that play ridge's role: every linear recombination of
    channels or hidden units on the path from response to prediction.

    The gate biases, the layer-norm scales, and the queries are excluded --
    ridge penalizes a linear map, not an intercept or a pooling rule.

    OUTPUT:
        - parameters: list[torch.nn.Parameter] -> penalizable weight matrices
    """
    def ridge_like_parameters(self):
        parameters = [head[-1].weight for head in self.heads]
        # The first cell's input-to-hidden block is the channel recombination
        # that ridge's coefficient matrix is the linear counterpart of.
        parameters.append(self.encoder.cells[0].weight_ih)
        if self.bottleneck is not None:
            parameters.append(self.bottleneck[0].weight)
        # end if a channel bottleneck is active
        return parameters

    def forward(self, sequence):
        if self.bottleneck is not None:
            sequence = self.bottleneck(sequence)
        # end if a channel bottleneck is active
        hidden_sequence = self.hidden_dropout(self.encoder(sequence))

        if self.pooling == "attention":
            # [batch, time, hidden] x [targets, hidden] -> [batch, targets, time]
            scores = torch.einsum(
                "bth,lh->blt", hidden_sequence, self.queries
            ) / (hidden_sequence.shape[-1] ** 0.5)
            attention = torch.softmax(scores, dim=-1)
            # Each target layer gets its own convex combination of the same
            # hidden states: [batch, targets, hidden].
            pooled = torch.einsum("blt,bth->blh", attention, hidden_sequence)
        else:
            attention = None
            shared = (
                hidden_sequence.mean(dim=1)
                if self.pooling == "mean"
                else hidden_sequence[:, -1]
            )
            # The ablation feeds one and the same code to all three heads.
            pooled = shared.unsqueeze(1).expand(-1, len(self.heads), -1)
        # end if the model pools by attention

        predictions = torch.cat(
            [head(pooled[:, idx]) for idx, head in enumerate(self.heads)], dim=-1
        )
        return predictions, {"attention": attention}
# EOC


class IndependentNeuralDecoders(nn.Module):
    """
    One complete small decoder per target layer, sharing nothing.

    This is the ablation on the *other* side of the shared-encoder design: if
    three separate encoders do as well, the shared dynamics are not buying
    anything and the attention is only a cheaper parameterization.

    INPUT (forward):
        - sequence: torch.Tensor -> [batch, time, n_channels]

    OUTPUT:
        - predictions: torch.Tensor -> [batch, sum of target dims]
        - diagnostics: dict -> "attention" [batch, targets, time] or None
    """

    """
    __init__
    Build one single-target AttentionPooledNeuralDecoder per target layer.

    INPUT:
        - target_dims: sequence[int] -> retained PCs of each target layer
        - decoder_kwargs: dict -> arguments shared by every branch

    OUTPUT:
        - None
    """
    def __init__(self, target_dims, **decoder_kwargs):
        super().__init__()
        target_dims = [int(dim) for dim in target_dims]
        self.branches = nn.ModuleList(
            AttentionPooledNeuralDecoder(target_dims=[dim], **decoder_kwargs)
            for dim in target_dims
        )
        boundaries = torch.tensor([0] + target_dims).cumsum(0).tolist()
        self.target_slices = [
            (boundaries[idx], boundaries[idx + 1]) for idx in range(len(target_dims))
        ]
        self.target_dims = target_dims

    """
    ridge_like_parameters
    Pool the ridge-analogous maps of every branch.

    OUTPUT:
        - parameters: list[torch.nn.Parameter] -> penalizable weight matrices
    """
    def ridge_like_parameters(self):
        return [
            parameter
            for branch in self.branches
            for parameter in branch.ridge_like_parameters()
        ]

    def forward(self, sequence):
        branch_predictions, branch_attention = [], []
        for branch in self.branches:
            predictions, diagnostics = branch(sequence)
            branch_predictions.append(predictions)
            if diagnostics["attention"] is not None:
                # Each branch attends with its single query: [batch, 1, time].
                branch_attention.append(diagnostics["attention"])
            # end if this branch pools by attention
        # end for target branch
        attention = (
            torch.cat(branch_attention, dim=1) if branch_attention else None
        )
        return torch.cat(branch_predictions, dim=-1), {"attention": attention}
# EOC


"""
build_neural_decoder
Instantiate the decoder a variant name asks for.

INPUT:
    - variant: str -> "attention", "mean_pool", "last_state", or "independent"
    - n_channels: int -> channels (or input PCs) per time bin
    - n_timepoints: int -> neural time bins
    - target_dims: sequence[int] -> retained PCs of each target layer
    - decoder_kwargs: dict -> width, depth, dropout, bottleneck, head settings

OUTPUT:
    - model: nn.Module -> decoder returning (predictions, diagnostics)
"""
def build_neural_decoder(
    variant, n_channels, n_timepoints, target_dims, **decoder_kwargs
):
    shared = {
        "n_channels": n_channels,
        "n_timepoints": n_timepoints,
        **decoder_kwargs,
    }
    if variant == "attention":
        return AttentionPooledNeuralDecoder(
            target_dims=target_dims, pooling="attention", **shared
        )
    # end if the per-target attention decoder was requested
    if variant == "mean_pool":
        return AttentionPooledNeuralDecoder(
            target_dims=target_dims, pooling="mean", **shared
        )
    # end if the shared mean-pooled ablation was requested
    if variant == "last_state":
        return AttentionPooledNeuralDecoder(
            target_dims=target_dims, pooling="last", **shared
        )
    # end if the shared final-state ablation was requested
    if variant == "independent":
        return IndependentNeuralDecoders(
            target_dims=target_dims, pooling="attention", **shared
        )
    # end if the unshared-encoder ablation was requested
    raise ValueError(f"Unknown decoder variant {variant!r}.")
# EOF
