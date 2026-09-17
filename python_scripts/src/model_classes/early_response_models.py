"""
Time-bin decoders that also observe the first bins of the neural response.

``timebin_models`` predicts the 80-180 ms TVSD response from frozen ANN
features alone: the decoder never sees the recording it is asked to predict.
The two decoders here keep that architecture and add one input, the same
presentation's own 0-50 ms activity. IT latency is around 70 ms, so those bins
carry no stimulus drive yet; what they do carry is the trial's own state --
where the site's baseline-corrected activity sits before the response starts.
Giving the decoder that state asks whether the response window is predictable
from the image plus the state the cortex was in, or from the image alone.

The early trace enters exactly where a state belongs, at the initial condition
of the recurrence: the GRU starts from it and the linear dynamical system takes
it as z_0. Nothing else about either architecture changes, so switching
``use_early_response`` off recovers the plain decoder of ``timebin_models``
parameter for parameter, which is the ablation this experiment is read against.
"""

import torch
from torch import nn

from model_classes.temporal_models import (
    GAUSSIAN_NOISE_LAYER_NAME,
    ShowAttendTellGRUModel,
    sample_sphere_noise_layer,
)
from model_classes.timebin_models import (
    GRUTimebinDecoder,
    LinearDynamicalSystemDecoder,
)


class EarlyResponseEncoder(nn.Module):
    """
    Compress the observed early bins of one presentation into a state code.

    The early trace is already on the target's standardized per-channel scale,
    so the LayerNorm here only removes the per-presentation offset and gain
    that a drifting baseline leaves behind. Dropout sits on the code rather
    than on the raw trace: the bins are few and highly correlated, so dropping
    raw samples would mostly delete whole time points.

    INPUT (forward):
        - early_response: torch.Tensor -> [batch, early bins, neurons]

    OUTPUT:
        - early_code: torch.Tensor -> [batch, early_dim]
    """

    """
    __init__
    Build the flattening normalization and the projection to the code width.

    INPUT:
        - n_early_bins: int -> observed response bins, 5 for 0-50 ms at 100 Hz
        - n_neurons: int -> recorded channels, the same population as the target
        - early_dim: int -> width of the state code
        - dropout: float -> dropout applied to the code

    OUTPUT:
        - None
    """
    def __init__(self, n_early_bins, n_neurons, early_dim, dropout=0.0):
        super().__init__()
        if min(n_early_bins, n_neurons, early_dim) <= 0:
            raise ValueError("All early-encoder dimensions must be positive.")
        # end if a dimension is invalid
        self.n_early_bins = n_early_bins
        self.n_neurons = n_neurons
        self.early_dim = early_dim
        self.input_norm = nn.LayerNorm(
            n_early_bins * n_neurons, elementwise_affine=False
        )
        self.projection = nn.Linear(n_early_bins * n_neurons, early_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, early_response):
        expected_shape = (self.n_early_bins, self.n_neurons)
        if (
            early_response.ndim != 3
            or tuple(early_response.shape[1:]) != expected_shape
        ):
            raise ValueError(
                "Expected [batch, early bins, neurons] with trailing shape "
                f"{expected_shape}, got {tuple(early_response.shape)}."
            )
        # end if the early trace has the wrong shape
        flat_response = self.input_norm(early_response.flatten(start_dim=1))
        return self.dropout(torch.nn.functional.gelu(self.projection(flat_response)))
    # EOF
# EOC


"""
build_early_encoder
Build the early-response encoder, or None when the decoder ignores the trace.

Returning None rather than a zeroed module keeps the ablation honest: a decoder
without the early input owns no parameter that reads it.

INPUT:
    - use_early_response: bool -> whether the decoder observes the early bins
    - n_early_bins: int -> observed response bins
    - n_neurons: int -> recorded channels
    - early_dim: int -> width of the state code
    - dropout: float -> dropout applied to the code

OUTPUT:
    - encoder: EarlyResponseEncoder | None -> the encoder, or None
    - context_dim: int -> width the encoder contributes, 0 when it is absent
"""
def build_early_encoder(
    use_early_response, n_early_bins, n_neurons, early_dim, dropout
):
    if not use_early_response:
        return None, 0
    # end if this decoder is the no-early-response ablation
    encoder = EarlyResponseEncoder(n_early_bins, n_neurons, early_dim, dropout)
    return encoder, early_dim
# EOF


class EarlyResponseGRUDecoder(GRUTimebinDecoder):
    """
    The searched GRU decoder, started from the observed 0-50 ms state.

    Everything the parent does is unchanged: the image code is projected once
    and re-entered at every step next to a learned per-bin time embedding, and
    one shared readout maps the hidden sequence to the population. The only
    difference is the initial hidden state, which now reads the early trace as
    well as the image, so the recurrence rolls the target window forward from
    where the recording actually was rather than from the image alone.

    INPUT (forward):
        - layer_features: torch.Tensor -> cached features [batch, cells, embedding]
        - early_response: torch.Tensor -> [batch, early bins, neurons]
        - return_diagnostics: bool -> whether to return the hidden sequence

    OUTPUT:
        - predictions: torch.Tensor -> activity [batch, time, neurons]
        - diagnostics: dict | None -> hidden states and the early code
    """

    """
    __init__
    Build the parent decoder, then widen its initial-state map by the early code.

    INPUT:
        - n_layers, feature_dim, n_timepoints, n_neurons: see NoisyCachedFeatureModel
        - hidden_dim: int -> recurrent state width, also the image code width
        - time_embedding_dim: int -> width of the per-bin time input
        - n_early_bins: int -> observed response bins fed to the encoder
        - early_dim: int -> width of the early state code
        - use_early_response: bool -> False builds the ablation
        - dropout: float -> dropout on the early code and the readout input
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
        n_early_bins,
        early_dim=64,
        use_early_response=True,
        dropout=0.0,
        **noise_kwargs,
    ):
        super().__init__(
            n_layers,
            feature_dim,
            n_timepoints,
            n_neurons,
            hidden_dim,
            time_embedding_dim,
            dropout=dropout,
            **noise_kwargs,
        )
        self.use_early_response = bool(use_early_response)
        self.early_encoder, context_dim = build_early_encoder(
            self.use_early_response, n_early_bins, n_neurons, early_dim, dropout
        )
        # The parent's initial-state map reads the image code alone; with the
        # early trace it reads both, and is otherwise the same linear layer.
        self.initial_state = nn.Linear(hidden_dim + context_dim, hidden_dim)

    """
    ridge_like_parameters
    The parent's ridge-analogous maps, plus the map that reads the early trace.

    OUTPUT:
        - parameters: list[torch.nn.Parameter] -> input and output map weights
    """
    def ridge_like_parameters(self):
        parameters = super().ridge_like_parameters()
        if self.early_encoder is not None:
            parameters.append(self.early_encoder.projection.weight)
        # end if this decoder observes the early trace
        return parameters
    # EOF

    def forward(self, layer_features, early_response, return_diagnostics=False):
        features = self._noisy_features(layer_features)
        image_features = self.image_projection(features.flatten(start_dim=1))
        repeated_image_features = image_features.unsqueeze(1).expand(
            -1, self.n_timepoints, -1
        )
        time_features = self._jittered_time_embeddings(features.shape[0])
        recurrent_inputs = torch.cat(
            [repeated_image_features, time_features], dim=-1
        )

        # [batch, hidden + early_dim] when the trace is observed, the image
        # code alone otherwise, which is the parent's initial condition.
        early_code = (
            self.early_encoder(early_response)
            if self.early_encoder is not None
            else None
        )
        initial_input = (
            image_features
            if early_code is None
            else torch.cat([image_features, early_code], dim=-1)
        )
        initial_state = torch.tanh(self.initial_state(initial_input)).unsqueeze(0)
        hidden_sequence, _ = self.recurrence(recurrent_inputs, initial_state)
        predictions = self.readout(hidden_sequence)

        diagnostics = None
        if return_diagnostics:
            diagnostics = {"hidden_sequence": hidden_sequence}
            if early_code is not None:
                diagnostics["early_code"] = early_code
            # end if the decoder observed the early trace
        # end if diagnostics are requested
        return predictions, diagnostics
    # EOF
# EOC


class EarlyResponseLDSDecoder(LinearDynamicalSystemDecoder):
    """
    The linear dynamical system with the observed early state as z_0.

    z_0 = B_0 [x, e], z_t = A z_(t-1) + B x + process noise, y_t = C z_t + d_t.
    Only the initial condition changes with respect to the parent, which is the
    exact linear counterpart of what the GRU is given: the stimulus drive, the
    transition and the readout stay linear, so a gain over the ablation is
    attributable to the early state and not to added nonlinearity.

    INPUT (forward):
        - layer_features: torch.Tensor -> cached features [batch, cells, embedding]
        - early_response: torch.Tensor -> [batch, early bins, neurons]
        - return_diagnostics: bool -> whether to return the latent trajectory

    OUTPUT:
        - predictions: torch.Tensor -> activity [batch, time, neurons]
        - diagnostics: dict | None -> states [batch, time, state_dim]
    """

    """
    __init__
    Build the parent system, then widen its initial projection by the early code.

    INPUT:
        - n_layers, feature_dim, n_timepoints, n_neurons: see NoisyCachedFeatureModel
        - state_dim: int -> latent state width
        - n_early_bins: int -> observed response bins fed to the encoder
        - early_dim: int -> width of the early state code
        - use_early_response: bool -> False builds the ablation
        - dropout: float -> dropout on the early code and the readout input
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
        n_early_bins,
        early_dim=64,
        use_early_response=True,
        dropout=0.0,
        spectral_init=0.9,
        **noise_kwargs,
    ):
        super().__init__(
            n_layers,
            feature_dim,
            n_timepoints,
            n_neurons,
            state_dim,
            dropout=dropout,
            spectral_init=spectral_init,
            **noise_kwargs,
        )
        self.use_early_response = bool(use_early_response)
        self.early_encoder, context_dim = build_early_encoder(
            self.use_early_response, n_early_bins, n_neurons, early_dim, dropout
        )
        input_dim = n_layers * feature_dim
        # Only z_0 sees the early state; the per-step drive B x stays the
        # stimulus, so the trace sets the initial condition and nothing else.
        self.initial_projection = nn.Linear(input_dim + context_dim, state_dim)

    """
    ridge_like_parameters
    The parent's ridge-analogous maps, plus the map that reads the early trace.

    OUTPUT:
        - parameters: list[torch.nn.Parameter] -> input and output map weights
    """
    def ridge_like_parameters(self):
        parameters = super().ridge_like_parameters()
        if self.early_encoder is not None:
            parameters.append(self.early_encoder.projection.weight)
        # end if this decoder observes the early trace
        return parameters
    # EOF

    def forward(self, layer_features, early_response, return_diagnostics=False):
        flat_features = self._noisy_features(layer_features).flatten(start_dim=1)
        drive = self.input_projection(flat_features)

        early_code = (
            self.early_encoder(early_response)
            if self.early_encoder is not None
            else None
        )
        initial_input = (
            flat_features
            if early_code is None
            else torch.cat([flat_features, early_code], dim=-1)
        )
        state = self._add_process_noise(self.initial_projection(initial_input))

        states = [state]
        for _ in range(1, self.n_timepoints):
            state = self._add_process_noise(self.transition(state) + drive)
            states.append(state)
        # end for latent time step
        states = torch.stack(states, dim=1)
        predictions = self.readout(self.dropout(states)) + self.timebin_bias

        diagnostics = None
        if return_diagnostics:
            diagnostics = {"states": states}
            if early_code is not None:
                diagnostics["early_code"] = early_code
            # end if the decoder observed the early trace
        # end if diagnostics are requested
        return predictions, diagnostics
    # EOF
# EOC


class EarlyResponseShowAttendTellGRUModel(ShowAttendTellGRUModel):
    """
    The Show-Attend-and-Tell GRU, started from the observed pre-response state.

    Attention, recurrence, and readout are those of ShowAttendTellGRUModel. The
    only change is h_0: the parent maps the mean image layer to the initial
    state, whereas here the same map also reads a code of the presentation's
    own first neural bins (0-50 ms, before IT latency). The recurrence therefore
    rolls the remaining bins forward from where the recording actually was.

    Optionally, as in NoiseLayerBaselineModel, attention also ranges over one
    extra pseudo-layer drawn uniformly on the sphere for every image and pass.
    It carries no stimulus information, so attention mass spent on it is wasted
    prediction budget. The draw is appended after the per-layer normalization
    and does not enter h_0, which reads the real layers only.

    Optionally, a learned embedding per predicted bin is added to the
    state-derived query, W_h h_(t-1) + e_t, so the attention schedule can change
    over time even when the recurrent state settles. The time variation of
    attention is then partly a learned clock rather than purely state-driven.

    INPUT (forward):
        - x: torch.Tensor -> images or cached features [batch, layers, embedding]
        - early_response: torch.Tensor -> [batch, early bins, neurons]
        - use_precomputed_features: bool -> whether x is cached features
        - return_hidden_states: bool -> also return the recurrent state sequence

    OUTPUT:
        - neural_predictions: torch.Tensor -> activity [batch, time, neurons]
        - attention_weights: torch.Tensor -> [batch, time, layers, embedding],
          with layers + 1 attended items when the noise layer is used
        - hidden_states: torch.Tensor -> [batch, time, hidden_dim], only when
          return_hidden_states is True
    """

    """
    __init__
    Build the parent decoder, then widen its initial-state map by the early code.

    INPUT:
        - encoder, layers, n_timepoints, n_neurons, hidden_dim, attention_dim,
          dropout, encoder_dim: see ShowAttendTellGRUModel; n_timepoints counts
          the predicted bins only, without the observed early bins
        - n_early_bins: int -> observed early bins, 5 for 0-50 ms at 100 Hz
        - early_dim: int -> width of the early state code
        - early_dropout: float -> dropout on the early state code
        - use_noise_layer: bool -> attend over one extra sphere-noise layer
        - match_noise_norm: bool -> scale the noise to the mean real-layer norm
        - noise_in_eval: bool -> keep sampling noise outside training mode;
          otherwise the noise item is a zero vector
        - use_temporal_embeddings: bool -> add a learned per-bin embedding
          [n_timepoints, attention_dim] to the attention query

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
        n_early_bins,
        early_dim=64,
        dropout=0.0,
        early_dropout=0.0,
        encoder_dim=None,
        use_noise_layer=False,
        match_noise_norm=True,
        noise_in_eval=True,
        use_temporal_embeddings=False,
    ):
        super().__init__(
            encoder,
            layers=layers,
            n_timepoints=n_timepoints,
            n_neurons=n_neurons,
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
            dropout=dropout,
            encoder_dim=encoder_dim,
        )
        self.n_early_bins = int(n_early_bins)
        self.early_encoder = EarlyResponseEncoder(
            self.n_early_bins, self.n_neurons, early_dim, early_dropout
        )
        # Same Linear + Tanh as the parent, reading [image mean, early code].
        self.initial_state = nn.Sequential(
            nn.Linear(self.encoder_dim + early_dim, self.hidden_dim),
            nn.Tanh(),
        )

        self.use_noise_layer = bool(use_noise_layer)
        self.match_noise_norm = bool(match_noise_norm)
        self.noise_in_eval = bool(noise_in_eval)
        if self.use_noise_layer:
            # n_layers stays at the real count so cached inputs still validate;
            # only the parameters indexed by attended item are rebuilt at L + 1.
            n_attended_items = self.n_layers + 1
            self.n_attention_features = n_attended_items * self.encoder_dim
            self.feature_key_embeddings = nn.Parameter(
                torch.empty(n_attended_items, self.encoder_dim, self.attention_dim)
            )
            nn.init.normal_(self.feature_key_embeddings, mean=0.0, std=0.02)
            self.context_projection = nn.Linear(
                self.n_attention_features,
                self.hidden_dim,
                bias=False,
            )
        # end if attention also ranges over the noise pseudo-layer

        # Only the time-embedding variant owns time-indexed parameters, so the
        # default model keeps the parent's state-only query exactly.
        self.use_temporal_embeddings = bool(use_temporal_embeddings)
        self.temporal_query_embeddings = None
        if self.use_temporal_embeddings:
            self.temporal_query_embeddings = nn.Parameter(
                torch.randn(self.n_timepoints, self.attention_dim) * 0.02
            )
        # end if the query also reads a learned per-bin embedding

    def get_n_early_bins(self) -> int:
        return self.n_early_bins

    def get_layer_names(self) -> list[str]:
        # Names of the attended items, i.e. the attention layer axis.
        if self.use_noise_layer:
            return [*self.layer_names, GAUSSIAN_NOISE_LAYER_NAME]
        # end if the noise pseudo-layer is attended
        return self.layer_names

    """
    _build_attention_query
    State query, plus the learned embedding of the predicted bin when enabled.

    INPUT:
        - hidden_state: torch.Tensor -> previous GRU state [batch, hidden_dim]
        - time_index: int -> predicted bin

    OUTPUT:
        - query: torch.Tensor -> [batch, attention_dim]
    """
    def _build_attention_query(self, hidden_state, time_index):
        query = self.hidden_query_projection(hidden_state)
        if self.temporal_query_embeddings is not None:
            # [attention_dim] broadcasts over the batch axis.
            query = query + self.temporal_query_embeddings[time_index]
        # end if the query reads the per-bin embedding
        return query
    # EOF

    def forward(
        self,
        x,
        early_response,
        use_precomputed_features=False,
        return_hidden_states=False,
    ):
        layer_features = self._resolve_layer_features(
            x,
            use_precomputed_features,
        )
        normalized_features = self.feature_input_norm(layer_features)

        # h_0 reads the mean image layer and the early code: [batch, hidden_dim].
        early_code = self.early_encoder(
            early_response.to(dtype=normalized_features.dtype)
        )
        hidden_state = self.initial_state(
            torch.cat([normalized_features.mean(dim=1), early_code], dim=-1)
        )

        if self.use_noise_layer:
            if self.training or self.noise_in_eval:
                noise_layer = sample_sphere_noise_layer(
                    normalized_features, self.match_noise_norm
                )
            else:
                # Keep the attended item but make evaluation deterministic.
                noise_layer = torch.zeros_like(normalized_features[:, :1])
            # end if the noise layer is sampled or silenced
            # [batch, layers, embedding] -> [batch, layers + 1, embedding]
            normalized_features = torch.cat(
                [normalized_features, noise_layer], dim=1
            )
        # end if attention also ranges over the noise pseudo-layer
        return self._unroll(
            normalized_features,
            hidden_state,
            return_hidden_states,
        )
    # EOF
# EOC


EARLY_RESPONSE_MODEL_CLASSES = {
    "gru": EarlyResponseGRUDecoder,
    "lds": EarlyResponseLDSDecoder,
}


"""
build_early_response_model
Construct one early-response decoder from a shared data shape and its settings.

INPUT:
    - model_name: str -> key in EARLY_RESPONSE_MODEL_CLASSES
    - model_kwargs: dict -> data shapes, architecture, early, and noise settings

OUTPUT:
    - model: nn.Module -> requested decoder
"""
def build_early_response_model(model_name, **model_kwargs):
    if model_name not in EARLY_RESPONSE_MODEL_CLASSES:
        raise KeyError(
            f"Unknown model {model_name!r}; choose from "
            f"{list(EARLY_RESPONSE_MODEL_CLASSES)}."
        )
    # end if the requested architecture is unavailable
    return EARLY_RESPONSE_MODEL_CLASSES[model_name](**model_kwargs)
# EOF
