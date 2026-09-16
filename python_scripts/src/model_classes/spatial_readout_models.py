"""
Readouts from a convolutional feature map to the TVSD time-bin population.

Everything else in this project reads pooled ANN features, one vector per
layer, which throws away where in the image a feature was found. These two
heads take the whole map instead and are meant to be compared against each
other: the factorized readout gives every IT site its own spatial weighting,
while the pooled control sees the same features with the spatial axis averaged
away. A gain that survives only in the factorized head is evidence that spatial
detail matters; a gain both heads show is about the features themselves.

Both are written to predict a *residual* left by a base decoder, so they carry
no output nonlinearity and start at zero: an untrained head returns zeros and
leaves the base prediction untouched.
"""

import torch
from torch import nn

from model_classes.timebin_models import build_timebin_model


class SpatialFeatureReadout(nn.Module):
    """Shared input normalization and channel mixing for the spatial heads."""

    """
    __init__
    Validate the map geometry and build the channel-standardizing 1x1 mixer.

    INPUT:
        - n_channels: int -> channels of the cached feature map
        - spatial_size: tuple[int, int] -> map height and width
        - n_timepoints: int -> number of neural target bins
        - n_neurons: int -> number of neural output channels
        - hidden_dim: int -> width after the 1x1 channel mixer
        - dropout: float -> dropout applied to the mixed features

    OUTPUT:
        - None
    """
    def __init__(
        self,
        n_channels,
        spatial_size,
        n_timepoints,
        n_neurons,
        hidden_dim,
        dropout=0.0,
    ):
        super().__init__()
        height, width = spatial_size
        dimensions = (n_channels, height, width, n_timepoints, n_neurons, hidden_dim)
        if min(dimensions) <= 0:
            raise ValueError("All readout dimensions must be positive.")
        # end if a dimension is invalid
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1).")
        # end if the dropout probability is invalid

        self.n_channels = n_channels
        self.height, self.width = height, width
        self.n_positions = height * width
        self.n_timepoints = n_timepoints
        self.n_neurons = n_neurons
        self.hidden_dim = hidden_dim

        # Cached activations are raw ReLU outputs with very different scales per
        # channel; batch statistics standardize them without a learned gain.
        self.input_norm = nn.BatchNorm2d(n_channels, affine=False)
        self.channel_mixer = nn.Conv2d(n_channels, hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    """
    _mixed_features
    Standardize the map and mix channels down to the readout width.

    INPUT:
        - feature_map: torch.Tensor -> [batch, channels, height, width]

    OUTPUT:
        - features: torch.Tensor -> [batch, hidden_dim, positions]
    """
    def _mixed_features(self, feature_map):
        expected = (self.n_channels, self.height, self.width)
        if feature_map.ndim != 4 or tuple(feature_map.shape[1:]) != expected:
            raise ValueError(
                f"Expected [batch, {expected[0]}, {expected[1]}, {expected[2]}], "
                f"got {tuple(feature_map.shape)}."
            )
        # end if the feature map has the wrong geometry
        mixed = self.channel_mixer(self.input_norm(feature_map))
        mixed = self.dropout(torch.nn.functional.gelu(mixed))
        return mixed.flatten(start_dim=2)
    # EOF
# EOC


class FactorizedSpatialReadout(SpatialFeatureReadout):
    """
    Give every site its own spatial mask over the feature map.

    The prediction factorizes into where a site reads (one softmax mask over the
    map positions, shared across time bins) and what it reads there (one feature
    vector per bin and site). That is the standard factorized readout of the
    CNN-to-neuron literature, and it is small enough -- about 170k parameters
    for AlexNet conv5 and 320 sites -- to be fitted on 20k images.

    INPUT (forward):
        - feature_map: torch.Tensor -> [batch, channels, height, width]
        - return_diagnostics: bool -> whether to return the spatial masks

    OUTPUT:
        - predictions: torch.Tensor -> [batch, time, neurons]
        - diagnostics: dict | None -> masks [neurons, height, width]
    """

    """
    __init__
    Build the per-site spatial masks and the per-bin feature weights.

    INPUT:
        - see SpatialFeatureReadout
        - mask_temperature: float -> softmax temperature of the spatial masks

    OUTPUT:
        - None
    """
    def __init__(
        self,
        n_channels,
        spatial_size,
        n_timepoints,
        n_neurons,
        hidden_dim,
        dropout=0.0,
        mask_temperature=1.0,
        **_,
    ):
        super().__init__(
            n_channels, spatial_size, n_timepoints, n_neurons, hidden_dim, dropout
        )
        if mask_temperature <= 0.0:
            raise ValueError("mask_temperature must be positive.")
        # end if the mask temperature is invalid
        self.mask_temperature = float(mask_temperature)

        # Near-uniform logits start every site reading the whole map, so the
        # masks sharpen only where the data ask for it.
        self.spatial_logits = nn.Parameter(
            torch.randn(n_neurons, self.n_positions) * 0.01
        )
        # The masks only receive gradient through the feature weights, so
        # exact zeros would leave them at their initial values forever. Weights
        # this small still start the head within 1e-3 of the base prediction.
        self.feature_weights = nn.Parameter(
            torch.randn(n_timepoints, n_neurons, hidden_dim) * 1e-3
        )
        self.output_bias = nn.Parameter(torch.zeros(n_timepoints, n_neurons))

    """
    spatial_masks
    Return the normalized read-out mask of every site as a map.

    OUTPUT:
        - masks: torch.Tensor -> [neurons, height, width], each summing to one
    """
    def spatial_masks(self):
        masks = torch.softmax(
            self.spatial_logits / self.mask_temperature, dim=-1
        )
        return masks.reshape(self.n_neurons, self.height, self.width)
    # EOF

    def forward(self, feature_map, return_diagnostics=False):
        features = self._mixed_features(feature_map)
        masks = torch.softmax(
            self.spatial_logits / self.mask_temperature, dim=-1
        )
        # [batch, neurons, hidden]: each site pools the map through its mask.
        pooled = torch.einsum("bhp,np->bnh", features, masks)
        predictions = torch.einsum(
            "bnh,tnh->btn", pooled, self.feature_weights
        )
        predictions = predictions + self.output_bias
        diagnostics = (
            {"spatial_masks": self.spatial_masks()} if return_diagnostics else None
        )
        return predictions, diagnostics
    # EOF
# EOC


class PooledSpatialControlReadout(SpatialFeatureReadout):
    """
    The same features with the spatial axis averaged away.

    This is the control for FactorizedSpatialReadout: identical backbone,
    identical channel mixer and comparable parameter count, but every site sees
    one global vector, so any advantage of the factorized head is spatial.

    INPUT (forward):
        - feature_map: torch.Tensor -> [batch, channels, height, width]
        - return_diagnostics: bool -> accepted for interface parity

    OUTPUT:
        - predictions: torch.Tensor -> [batch, time, neurons]
        - diagnostics: dict | None -> the pooled feature vector
    """

    def __init__(
        self,
        n_channels,
        spatial_size,
        n_timepoints,
        n_neurons,
        hidden_dim,
        dropout=0.0,
        **_,
    ):
        super().__init__(
            n_channels, spatial_size, n_timepoints, n_neurons, hidden_dim, dropout
        )
        self.readout = nn.Linear(hidden_dim, n_timepoints * n_neurons)
        # Matched to the factorized head's initial scale, so the two start from
        # the same distance to the base prediction.
        nn.init.normal_(self.readout.weight, std=1e-3)
        nn.init.zeros_(self.readout.bias)

    def forward(self, feature_map, return_diagnostics=False):
        pooled = self._mixed_features(feature_map).mean(dim=-1)
        predictions = self.readout(pooled).reshape(
            -1, self.n_timepoints, self.n_neurons
        )
        diagnostics = {"pooled": pooled} if return_diagnostics else None
        return predictions, diagnostics
    # EOF
# EOC


class JointSpatialTemporalDecoder(nn.Module):
    """
    Predict the population from pooled ANN layers and a conv map at once.

    The two-stage residual experiment froze a temporal decoder and patched what
    it missed. Here both paths are trained together and their predictions are
    summed, so the pooled branch never has to explain what the spatial branch
    already covers, and neither is fitted to the other's leftovers. Disabling a
    branch turns the same class into that branch on its own, which is how the
    ablations in this comparison are run.

    INPUT (forward):
        - layer_features: torch.Tensor -> pooled features [batch, layers, embedding]
        - feature_map: torch.Tensor -> conv map [batch, channels, height, width]
        - return_diagnostics: bool -> whether to return per-branch predictions

    OUTPUT:
        - predictions: torch.Tensor -> activity [batch, time, neurons]
        - diagnostics: dict | None -> each branch's own prediction and the masks
    """

    """
    __init__
    Build the requested branches on the shared target geometry.

    INPUT:
        - temporal_architecture: str -> key in TIMEBIN_MODEL_CLASSES
        - temporal_kwargs: dict -> arguments of the pooled-feature branch
        - spatial_readout: str -> key in SPATIAL_READOUT_CLASSES
        - spatial_kwargs: dict -> arguments of the conv-map branch
        - use_temporal: bool -> include the pooled-feature branch
        - use_spatial: bool -> include the conv-map branch

    OUTPUT:
        - None
    """
    def __init__(
        self,
        temporal_architecture,
        temporal_kwargs,
        spatial_readout,
        spatial_kwargs,
        use_temporal=True,
        use_spatial=True,
    ):
        super().__init__()
        if not (use_temporal or use_spatial):
            raise ValueError("At least one branch must be enabled.")
        # end if the decoder would have no input at all

        self.use_temporal = bool(use_temporal)
        self.use_spatial = bool(use_spatial)
        self.temporal_branch = (
            build_timebin_model(temporal_architecture, **temporal_kwargs)
            if self.use_temporal
            else None
        )
        self.spatial_branch = (
            SPATIAL_READOUT_CLASSES[spatial_readout](**spatial_kwargs)
            if self.use_spatial
            else None
        )

    """
    mask_parameter_names
    Names of the spatial-mask parameters, which need their own optimizer group.

    OUTPUT:
        - names: list[str] -> parameter names of the spatial masks
    """
    def mask_parameter_names(self):
        return [
            name
            for name, _ in self.named_parameters()
            if name.endswith("spatial_logits")
        ]
    # EOF

    def forward(self, layer_features, feature_map, return_diagnostics=False):
        branch_predictions = {}
        if self.temporal_branch is not None:
            branch_predictions["temporal"] = self.temporal_branch(layer_features)[0]
        # end if the pooled branch is enabled
        if self.spatial_branch is not None:
            branch_predictions["spatial"] = self.spatial_branch(feature_map)[0]
        # end if the conv-map branch is enabled

        predictions = sum(branch_predictions.values())
        diagnostics = None
        if return_diagnostics:
            diagnostics = dict(branch_predictions)
            if self.spatial_branch is not None:
                diagnostics["spatial_masks"] = self.spatial_branch.spatial_masks()
            # end if the spatial branch owns masks
        # end if diagnostics are requested
        return predictions, diagnostics
    # EOF
# EOC


SPATIAL_READOUT_CLASSES = {
    "factorized_spatial": FactorizedSpatialReadout,
    "pooled_control": PooledSpatialControlReadout,
}


"""
build_spatial_readout
Construct one residual readout from the cached map geometry.

INPUT:
    - readout_name: str -> key in SPATIAL_READOUT_CLASSES
    - readout_kwargs: dict -> geometry and architecture arguments

OUTPUT:
    - readout: nn.Module -> requested spatial readout
"""
def build_spatial_readout(readout_name, **readout_kwargs):
    if readout_name not in SPATIAL_READOUT_CLASSES:
        raise KeyError(
            f"Unknown readout {readout_name!r}; choose from "
            f"{list(SPATIAL_READOUT_CLASSES)}."
        )
    # end if the requested readout is unavailable
    return SPATIAL_READOUT_CLASSES[readout_name](**readout_kwargs)
# EOF
