"""
Cached-feature decoders for the TVSD 20 ms time-bin architecture search.

``temporal_experiment_models`` predicts a full 100 Hz time course of the three0
recordings and ``window_models`` collapses the TVSD response to a single
vector. This module sits between the two: the target is the TVSD IT response in
a handful of 20 ms bins (80-180 ms by default), and the architectures compared
over it -- the notebook's layer-attention baseline, a tiny transformer, a GRU,
a linear dynamical system, and a GRU that attends over the depths with no time
code at all -- share one noise interface, so the noise scale is a
hyperparameter of every architecture alike.

Two noise sources are available to every decoder:
    - input noise, added to the LayerNorm-normalized ANN features, so its
      standard deviation is expressed in units of the feature scale;
    - temporal noise, an independent per-image jitter of the learned time code,
      expressed as a fraction of that code's current spread. The LDS carries no
      time embedding, so there the same knob drives process noise on the latent
      state, which is the dynamical analogue of jittering the time code.

Both are active in training only unless ``noise_in_eval`` is set, and both are
inert at 0.0, so a noiseless configuration is exactly the plain architecture.
"""

import math

import torch
from torch import nn

from model_classes.temporal_experiment_models import CachedFeatureTemporalModel


class NoisyCachedFeatureModel(CachedFeatureTemporalModel):
    """Shared feature and temporal-code jitter for the searched decoders."""

    """
    __init__
    Store the data shapes and the two noise scales searched over.

    INPUT:
        - n_layers: int -> number of cached ANN depths
        - feature_dim: int -> cached embedding width of every depth
        - n_timepoints: int -> number of neural target bins
        - n_neurons: int -> number of neural output channels
        - input_noise_std: float -> feature-noise std, in normalized units
        - temporal_noise_std: float -> time-code jitter, as a fraction of its
          own current spread (process-noise scale for the LDS)
        - noise_in_eval: bool -> keep sampling noise outside training mode
        - normalize_features: bool -> LayerNorm the cached input, which is right
          for ANN activations and wrong for inputs already in target units

    OUTPUT:
        - None
    """
    def __init__(
        self,
        n_layers,
        feature_dim,
        n_timepoints,
        n_neurons,
        input_noise_std=0.0,
        temporal_noise_std=0.0,
        noise_in_eval=False,
        normalize_features=True,
    ):
        super().__init__(
            n_layers,
            feature_dim,
            n_timepoints,
            n_neurons,
            normalize_features=normalize_features,
        )
        if input_noise_std < 0.0 or temporal_noise_std < 0.0:
            raise ValueError("Noise scales must be non-negative.")
        # end if a requested noise scale is invalid
        self.input_noise_std = float(input_noise_std)
        self.temporal_noise_std = float(temporal_noise_std)
        self.noise_in_eval = bool(noise_in_eval)

    """
    _noise_is_active
    Report whether noise should be sampled in the current mode.

    OUTPUT:
        - active: bool -> True while training, or in eval when requested
    """
    def _noise_is_active(self):
        return self.training or self.noise_in_eval
    # EOF

    """
    _noisy_features
    Normalize the cached features and perturb them with isotropic noise.

    The LayerNorm ahead of the noise fixes the feature scale, so a standard
    deviation of 0.5 means half the spread of a feature coordinate regardless
    of which encoder or depth produced it. With normalize_features off the
    input keeps its own units and the noise scale is read in those instead.

    INPUT:
        - layer_features: torch.Tensor -> cached features [batch, layers, embedding]

    OUTPUT:
        - features: torch.Tensor -> normalized, optionally noisy features
    """
    def _noisy_features(self, layer_features):
        features = self._normalized_features(layer_features)
        if self.input_noise_std > 0.0 and self._noise_is_active():
            features = features + torch.randn_like(features) * self.input_noise_std
        # end if input noise is active
        return features
    # EOF

    """
    _jittered_time_embeddings
    Broadcast the learned time code over the batch and jitter it per image.

    The table starts at std 0.02 and grows during training, so the jitter is
    read as a fraction of the current spread rather than as an absolute scale.

    INPUT:
        - batch_size: int -> number of images in the current forward pass

    OUTPUT:
        - embeddings: torch.Tensor -> [batch, time, time_embedding_dim]
    """
    def _jittered_time_embeddings(self, batch_size):
        embeddings = self.time_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        if self.temporal_noise_std > 0.0 and self._noise_is_active():
            noise_std = self.temporal_noise_std * self.time_embeddings.detach().std()
            embeddings = embeddings + torch.randn_like(embeddings) * noise_std
        # end if temporal jitter is active
        return embeddings
    # EOF
# EOC


class LayerAttentionTimebinDecoder(NoisyCachedFeatureModel):
    """
    Cached-feature twin of ``temporal_models.BaselineModel`` at layer granularity.

    One learned query per time bin attends over the selected ANN depths, two
    pointwise MLPs refine the attended value, and every bin owns its readout.
    Nothing mixes time, which is exactly the property the recurrent and
    transformer variants are meant to be tested against.

    INPUT (forward):
        - layer_features: torch.Tensor -> cached features [batch, layers, embedding]
        - return_diagnostics: bool -> whether to return attention weights

    OUTPUT:
        - predictions: torch.Tensor -> activity [batch, time, neurons]
        - diagnostics: dict | None -> attention [batch, time, layers]
    """

    """
    __init__
    Build the temporal queries, the layer key/value maps, and the bin readouts.

    INPUT:
        - n_layers, feature_dim, n_timepoints, n_neurons: see NoisyCachedFeatureModel
        - time_embedding_dim: int -> width of the learned query per time bin
        - value_dim: int -> projected layer-value width
        - mlp_hidden_dim: int -> hidden width of the two pointwise MLPs
        - dropout: float -> decoder dropout probability
        - noise_kwargs: dict -> noise scales forwarded to the shared base

    OUTPUT:
        - None
    """
    def __init__(
        self,
        n_layers,
        feature_dim,
        n_timepoints,
        n_neurons,
        time_embedding_dim,
        value_dim,
        mlp_hidden_dim,
        dropout=0.0,
        **noise_kwargs,
    ):
        super().__init__(
            n_layers, feature_dim, n_timepoints, n_neurons, **noise_kwargs
        )
        # Queries and keys share a width, so no query projection is needed.
        self.key_dim = time_embedding_dim
        self.time_embeddings = nn.Parameter(
            torch.randn(n_timepoints, time_embedding_dim) * 0.02
        )
        self.key_projection = nn.Linear(feature_dim, self.key_dim, bias=False)
        self.value_projection = nn.Linear(feature_dim, value_dim, bias=False)
        self.key_norm = nn.LayerNorm(self.key_dim)
        self.value_norm = nn.LayerNorm(value_dim)
        self.dropout = nn.Dropout(dropout)
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
        # Head t reads latent t alone and owns its neural prediction weights.
        self.timebin_readouts = nn.ModuleList(
            [nn.Linear(mlp_hidden_dim, n_neurons) for _ in range(n_timepoints)]
        )

    def forward(self, layer_features, return_diagnostics=False):
        features = self._noisy_features(layer_features)
        queries = self._jittered_time_embeddings(features.shape[0])

        keys = self.dropout(self.key_norm(self.key_projection(features)))
        values = self.dropout(self.value_norm(self.value_projection(features)))
        attention_logits = torch.matmul(queries, keys.transpose(-1, -2))
        attention = torch.softmax(attention_logits / math.sqrt(self.key_dim), dim=-1)

        # [B, T, value_dim]: one attended read of the depth stack per time bin.
        latents = torch.matmul(attention, values)
        neural_features = self.neural_feature_mlp(self.temporal_feature_mlp(latents))
        predictions = torch.stack(
            [
                readout(neural_features[:, timebin])
                for timebin, readout in enumerate(self.timebin_readouts)
            ],
            dim=1,
        )
        diagnostics = {"attention": attention} if return_diagnostics else None
        return predictions, diagnostics
    # EOF
# EOC


class TinyTransformerTimebinDecoder(NoisyCachedFeatureModel):
    """
    Cross-attend one query per time bin to the ANN depths, then mix bins.

    The depths are the memory of a single cross-attention block and the time
    bins are the tokens of a small self-attention stack, so unlike the baseline
    this decoder can let one bin's prediction depend on another's.

    INPUT (forward):
        - layer_features: torch.Tensor -> cached features [batch, layers, embedding]
        - return_diagnostics: bool -> whether to return cross-attention weights

    OUTPUT:
        - predictions: torch.Tensor -> activity [batch, time, neurons]
        - diagnostics: dict | None -> attention [batch, time, layers] and tokens
    """

    """
    __init__
    Build the depth memory, the temporal queries, and the encoder stack.

    INPUT:
        - n_layers, feature_dim, n_timepoints, n_neurons: see NoisyCachedFeatureModel
        - hidden_dim: int -> shared token width; must divide by the head count
        - n_attention_heads: int -> heads in both attention blocks
        - n_transformer_layers: int -> self-attention blocks over the time axis
        - dropout: float -> dropout inside attention and the feedforward blocks
        - noise_kwargs: dict -> noise scales forwarded to the shared base

    OUTPUT:
        - None
    """
    def __init__(
        self,
        n_layers,
        feature_dim,
        n_timepoints,
        n_neurons,
        hidden_dim,
        n_attention_heads,
        n_transformer_layers=1,
        dropout=0.0,
        **noise_kwargs,
    ):
        super().__init__(
            n_layers, feature_dim, n_timepoints, n_neurons, **noise_kwargs
        )
        if hidden_dim % n_attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_attention_heads.")
        # end if attention heads do not divide the token width
        if n_transformer_layers <= 0:
            raise ValueError("n_transformer_layers must be positive.")
        # end if the encoder stack is empty

        self.layer_projection = nn.Linear(feature_dim, hidden_dim)
        self.image_projection = nn.Linear(feature_dim, hidden_dim)
        self.time_embeddings = nn.Parameter(
            torch.randn(n_timepoints, hidden_dim) * 0.02
        )
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, n_attention_heads, dropout=dropout, batch_first=True
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
            encoder_layer, num_layers=n_transformer_layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.readout = nn.Linear(hidden_dim, n_neurons)

    def forward(self, layer_features, return_diagnostics=False):
        features = self._noisy_features(layer_features)
        # The time code is the query width here, so the jitter enters the same
        # table that the cross-attention reads.
        queries = self._jittered_time_embeddings(features.shape[0])
        queries = queries + self.image_projection(features.mean(dim=1)).unsqueeze(1)
        layer_memory = self.layer_projection(features)
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
            diagnostics = {"attention": attention, "tokens": temporal_features}
        # end if diagnostics are requested
        return predictions, diagnostics
    # EOF
# EOC


class GRUTimebinDecoder(NoisyCachedFeatureModel):
    """
    Drive a GRU with the static image code and one time embedding per bin.

    The image enters at every step rather than only through the initial state,
    which keeps the stimulus available to late bins without asking the gates to
    carry it across the whole window.

    INPUT (forward):
        - layer_features: torch.Tensor -> cached features [batch, layers, embedding]
        - return_diagnostics: bool -> whether to return the hidden sequence

    OUTPUT:
        - predictions: torch.Tensor -> activity [batch, time, neurons]
        - diagnostics: dict | None -> hidden states [batch, time, hidden_dim]
    """

    """
    __init__
    Build the image projection, the recurrence, and the shared readout.

    INPUT:
        - n_layers, feature_dim, n_timepoints, n_neurons: see NoisyCachedFeatureModel
        - hidden_dim: int -> recurrent state width, also the image code width
        - time_embedding_dim: int -> width of the per-bin time input
        - dropout: float -> dropout applied to the readout input
        - noise_kwargs: dict -> noise scales forwarded to the shared base

    OUTPUT:
        - None
    """
    def __init__(
        self,
        n_layers,
        feature_dim,
        n_timepoints,
        n_neurons,
        hidden_dim,
        time_embedding_dim,
        dropout=0.0,
        **noise_kwargs,
    ):
        super().__init__(
            n_layers, feature_dim, n_timepoints, n_neurons, **noise_kwargs
        )
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

    """
    ridge_like_parameters
    The linear maps this decoder shares with a ridge regression.

    Ridge penalizes one matrix from the features to the outputs. Here that role
    is split between the map into the recurrent width and the map back out; the
    GRU's own gate weights are the dynamics, not part of that path, so they are
    excluded and only reached by the "all" scope.

    OUTPUT:
        - parameters: list[torch.nn.Parameter] -> weights of the input and
          output maps, biases excluded exactly as ridge excludes its intercept
    """
    def ridge_like_parameters(self):
        return [
            self.image_projection[0].weight,
            self.initial_state.weight,
            self.readout[1].weight,
        ]
    # EOF

    def forward(self, layer_features, return_diagnostics=False):
        features = self._noisy_features(layer_features)
        image_features = self.image_projection(features.flatten(start_dim=1))
        repeated_image_features = image_features.unsqueeze(1).expand(
            -1, self.n_timepoints, -1
        )
        time_features = self._jittered_time_embeddings(features.shape[0])
        recurrent_inputs = torch.cat(
            [repeated_image_features, time_features], dim=-1
        )
        initial_state = torch.tanh(self.initial_state(image_features)).unsqueeze(0)
        hidden_sequence, _ = self.recurrence(recurrent_inputs, initial_state)
        predictions = self.readout(hidden_sequence)
        diagnostics = (
            {"hidden_sequence": hidden_sequence} if return_diagnostics else None
        )
        return predictions, diagnostics
    # EOF
# EOC


class LinearDynamicalSystemDecoder(NoisyCachedFeatureModel):
    """
    Evolve a latent state linearly under a constant stimulus drive.

    z_0 = B_0 x, z_t = A z_(t-1) + B x + process noise, y_t = C z_t + d_t. The
    only nonlinearity in the whole decoder is the feature normalization, so a
    win over this model is evidence that the response window needs nonlinear
    read-out rather than merely a smooth latency profile. The transition starts
    near a contracting identity, which keeps the five-step rollout stable.

    INPUT (forward):
        - layer_features: torch.Tensor -> cached features [batch, layers, embedding]
        - return_diagnostics: bool -> whether to return the latent trajectory

    OUTPUT:
        - predictions: torch.Tensor -> activity [batch, time, neurons]
        - diagnostics: dict | None -> states [batch, time, state_dim]
    """

    """
    __init__
    Build the input maps, the transition matrix, and the shared readout.

    INPUT:
        - n_layers, feature_dim, n_timepoints, n_neurons: see NoisyCachedFeatureModel
        - state_dim: int -> latent state width
        - dropout: float -> dropout applied to the readout input
        - spectral_init: float -> diagonal of the initial transition matrix
        - noise_kwargs: dict -> noise scales forwarded to the shared base

    OUTPUT:
        - None
    """
    def __init__(
        self,
        n_layers,
        feature_dim,
        n_timepoints,
        n_neurons,
        state_dim,
        dropout=0.0,
        spectral_init=0.9,
        **noise_kwargs,
    ):
        super().__init__(
            n_layers, feature_dim, n_timepoints, n_neurons, **noise_kwargs
        )
        input_dim = n_layers * feature_dim
        self.state_dim = state_dim
        self.initial_projection = nn.Linear(input_dim, state_dim)
        self.input_projection = nn.Linear(input_dim, state_dim, bias=False)
        self.transition = nn.Linear(state_dim, state_dim, bias=False)
        with torch.no_grad():
            # A contracting identity leaves the drive visible at every bin and
            # avoids an exploding rollout before the transition is learned.
            self.transition.weight.copy_(
                spectral_init * torch.eye(state_dim)
                + 0.01 * torch.randn(state_dim, state_dim)
            )
        # end with transition initialization
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(state_dim, n_neurons)
        # One bias per bin absorbs the mean latency profile that a single
        # shared readout matrix cannot express.
        self.timebin_bias = nn.Parameter(torch.zeros(n_timepoints, n_neurons))

    """
    _add_process_noise
    Perturb the latent state, using temporal_noise_std as the process-noise scale.

    INPUT:
        - state: torch.Tensor -> latent state [batch, state_dim]

    OUTPUT:
        - state: torch.Tensor -> optionally perturbed latent state
    """
    def _add_process_noise(self, state):
        if self.temporal_noise_std > 0.0 and self._noise_is_active():
            noise_std = self.temporal_noise_std * state.detach().std()
            state = state + torch.randn_like(state) * noise_std
        # end if process noise is active
        return state
    # EOF

    """
    spectral_radius
    Largest absolute eigenvalue of the learned transition matrix.

    OUTPUT:
        - radius: float -> spectral radius; below one means contracting dynamics
    """
    def spectral_radius(self):
        # MPS has no complex eigenvalue solver, so the spectrum is taken on CPU.
        transition = self.transition.weight.detach().cpu()
        return float(torch.linalg.eigvals(transition).abs().max())
    # EOF

    """
    ridge_like_parameters
    The linear maps this decoder shares with a ridge regression.

    B, B_0 and C are the stimulus-to-state and state-to-output maps, which is
    exactly what ridge penalizes. The transition A is the dynamics and is left
    out; shrinking it would pull the system toward a fixed point rather than
    toward a smaller readout, so it is only reached by the "all" scope.

    OUTPUT:
        - parameters: list[torch.nn.Parameter] -> input, initial and readout
          weights, biases excluded exactly as ridge excludes its intercept
    """
    def ridge_like_parameters(self):
        return [
            self.input_projection.weight,
            self.initial_projection.weight,
            self.readout.weight,
        ]
    # EOF

    def forward(self, layer_features, return_diagnostics=False):
        flat_features = self._noisy_features(layer_features).flatten(start_dim=1)
        drive = self.input_projection(flat_features)
        state = self._add_process_noise(self.initial_projection(flat_features))

        states = [state]
        for _ in range(1, self.n_timepoints):
            state = self._add_process_noise(self.transition(state) + drive)
            states.append(state)
        # end for latent time step
        states = torch.stack(states, dim=1)
        predictions = self.readout(self.dropout(states)) + self.timebin_bias
        diagnostics = {"states": states} if return_diagnostics else None
        return predictions, diagnostics
    # EOF
# EOC


class RecurrentAttentionTimebinDecoder(NoisyCachedFeatureModel):
    """
    A GRU that attends over the ANN depths, with no time code anywhere.

    Every other decoder here is told which bin it is predicting: the baseline
    and the transformer read a learned query per bin, the GRU is fed a time
    embedding at every step, and the LDS owns a per-bin bias. This one is not.
    The recurrence sees the same stimulus-driven context at every step, so the
    only thing separating bin t from bin t+1 is how far the hidden state has
    evolved, and the attention over depths is re-computed from that state --
    the decoder decides for itself which depth to read as the response unfolds.
    A win here is evidence that the latency profile is a dynamical property
    rather than something the decoder has to be handed.

    Because there is no time code to jitter, ``temporal_noise_std`` drives
    process noise on the hidden state, exactly as it does for the LDS.

    INPUT (forward):
        - layer_features: torch.Tensor -> cached features [batch, layers, embedding]
        - return_diagnostics: bool -> whether to return attention and states

    OUTPUT:
        - predictions: torch.Tensor -> activity [batch, time, neurons]
        - diagnostics: dict | None -> attention [batch, time, layers] and states
    """

    """
    __init__
    Build the depth key/value maps, the state-driven query, and the readout.

    INPUT:
        - n_layers, feature_dim, n_timepoints, n_neurons: see NoisyCachedFeatureModel
        - hidden_dim: int -> recurrent state width, also the context width
        - attention_dim: int -> width of the query/key space over depths
        - dropout: float -> dropout applied to the readout input
        - noise_kwargs: dict -> noise scales forwarded to the shared base

    OUTPUT:
        - None
    """
    def __init__(
        self,
        n_layers,
        feature_dim,
        n_timepoints,
        n_neurons,
        hidden_dim,
        attention_dim,
        dropout=0.0,
        **noise_kwargs,
    ):
        super().__init__(
            n_layers, feature_dim, n_timepoints, n_neurons, **noise_kwargs
        )
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.key_projection = nn.Linear(feature_dim, attention_dim, bias=False)
        self.value_projection = nn.Linear(feature_dim, hidden_dim, bias=False)
        # The query is a function of the state alone: no time index enters it.
        self.query_projection = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.initial_state = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
        )
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.recurrence = nn.GRUCell(
            input_size=hidden_dim, hidden_size=hidden_dim
        )
        self.dropout = nn.Dropout(dropout)
        # One readout shared by every bin; a per-bin head or bias would be a
        # time code by another name, which is exactly what this model drops.
        self.readout = nn.Linear(hidden_dim, n_neurons)

    """
    _add_process_noise
    Perturb the hidden state, using temporal_noise_std as the process-noise scale.

    INPUT:
        - state: torch.Tensor -> hidden state [batch, hidden_dim]

    OUTPUT:
        - state: torch.Tensor -> optionally perturbed hidden state
    """
    def _add_process_noise(self, state):
        if self.temporal_noise_std > 0.0 and self._noise_is_active():
            noise_std = self.temporal_noise_std * state.detach().std()
            state = state + torch.randn_like(state) * noise_std
        # end if process noise is active
        return state
    # EOF

    """
    ridge_like_parameters
    The linear maps this decoder shares with a ridge regression.

    The stimulus reaches the state through the value and initial-state maps and
    leaves through the readout; the GRU gates and the query map are dynamics
    and attention, so they are left to the "all" scope as elsewhere.

    OUTPUT:
        - parameters: list[torch.nn.Parameter] -> input and readout weights
    """
    def ridge_like_parameters(self):
        return [
            self.value_projection.weight,
            self.initial_state[0].weight,
            self.readout.weight,
        ]
    # EOF

    def forward(self, layer_features, return_diagnostics=False):
        features = self._noisy_features(layer_features)
        keys = self.key_projection(features)
        values = self.value_projection(features)
        state = self._add_process_noise(self.initial_state(features.mean(dim=1)))

        predictions, attention_sequence, states = [], [], []
        for _ in range(self.n_timepoints):
            # [B, L]: which depth this state wants to read, recomputed each bin.
            query = self.query_projection(state)
            attention_logits = torch.einsum("bk,blk->bl", query, keys)
            attention = torch.softmax(
                attention_logits / math.sqrt(self.attention_dim), dim=-1
            )
            context = torch.einsum("bl,blh->bh", attention, values)
            state = self._add_process_noise(
                self.recurrence(self.context_norm(context), state)
            )
            predictions.append(self.readout(self.dropout(state)))
            if return_diagnostics:
                attention_sequence.append(attention)
                states.append(state)
            # end if diagnostics are requested
        # end for latent time step

        predictions = torch.stack(predictions, dim=1)
        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                "attention": torch.stack(attention_sequence, dim=1),
                "states": torch.stack(states, dim=1),
            }
        # end if diagnostics are requested
        return predictions, diagnostics
    # EOF
# EOC


TIMEBIN_MODEL_CLASSES = {
    "baseline": LayerAttentionTimebinDecoder,
    "tiny_transformer": TinyTransformerTimebinDecoder,
    "gru": GRUTimebinDecoder,
    "lds": LinearDynamicalSystemDecoder,
    "gru_attention": RecurrentAttentionTimebinDecoder,
}


"""
build_timebin_model
Construct one searched decoder from a shared data shape and its hyperparameters.

INPUT:
    - model_name: str -> key in TIMEBIN_MODEL_CLASSES
    - model_kwargs: dict -> data shapes, architecture, and noise settings

OUTPUT:
    - model: nn.Module -> requested time-bin decoder
"""
def build_timebin_model(model_name, **model_kwargs):
    if model_name not in TIMEBIN_MODEL_CLASSES:
        raise KeyError(
            f"Unknown model {model_name!r}; choose from "
            f"{list(TIMEBIN_MODEL_CLASSES)}."
        )
    # end if the requested architecture is unavailable
    return TIMEBIN_MODEL_CLASSES[model_name](**model_kwargs)
# EOF
