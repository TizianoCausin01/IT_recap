"""
Multi-branch nonlinear decoders for the TVSD time-bin population.

`spatial_readout_models.JointSpatialTemporalDecoder` sums one pooled-feature
branch and one factorized spatial readout. Everything here generalizes that in
the three directions the frozen caches still leave open:

    - more than one convolutional map at a time, so AlexNet conv5 and ConvNeXt
      stage 4 can be read jointly instead of being compared;
    - a small trainable convolutional core on top of the frozen map, so the
      readout is not restricted to a linear recombination of frozen channels;
    - several masks per site, so a site whose receptive field is not a single
      blob is not forced into one.

Sites are laid out as [primary, auxiliary] along the output axis, which is how
an auxiliary area is trained jointly and then dropped at scoring time.
"""

import torch
from torch import nn

from model_classes.timebin_models import build_timebin_model


class MultiMaskSpatialReadout(nn.Module):
    """
    Read a frozen convolutional map through per-site masks, over a small core.

    With ``n_masks=1`` and ``core_layers=0`` this is the factorized readout of
    `spatial_readout_models`: a 1x1 channel mixer and one softmax mask per site.
    Adding core layers puts trainable 3x3 convolutions between the frozen map
    and the readout, and adding masks lets a site pool from several places and
    weight them differently.

    INPUT (forward):
        - feature_map: torch.Tensor -> [batch, channels, height, width]
        - return_diagnostics: bool -> whether to return masks and the core code

    OUTPUT:
        - predictions: torch.Tensor -> [batch, time, neurons]
        - diagnostics: dict | None -> masks and the globally pooled core code
    """

    """
    __init__
    Build the core, the per-site masks, and the per-bin feature weights.

    INPUT:
        - n_channels: int -> channels of the cached map
        - spatial_size: tuple[int, int] -> map height and width
        - n_timepoints: int -> number of neural target bins
        - n_neurons: int -> primary plus auxiliary output sites
        - hidden_dim: int -> width the readout reads at
        - dropout: float -> dropout on the mixed features
        - mask_temperature: float -> softmax temperature of the masks
        - n_masks: int -> masks per site
        - core_layers: int -> trainable 3x3 convolutions before the readout
        - core_dim: int | None -> width of the core, defaults to hidden_dim

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
        n_masks=1,
        core_layers=0,
        core_dim=None,
    ):
        super().__init__()
        height, width = spatial_size
        if min(n_channels, height, width, n_timepoints, n_neurons, hidden_dim) <= 0:
            raise ValueError("All readout dimensions must be positive.")
        # end if a dimension is invalid
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1).")
        # end if the dropout probability is invalid
        if n_masks < 1 or core_layers < 0:
            raise ValueError("n_masks must be >= 1 and core_layers >= 0.")
        # end if the readout shape is invalid
        if mask_temperature <= 0.0:
            raise ValueError("mask_temperature must be positive.")
        # end if the mask temperature is invalid

        self.n_channels = n_channels
        self.height, self.width = height, width
        self.n_positions = height * width
        self.n_timepoints = n_timepoints
        self.n_neurons = n_neurons
        self.hidden_dim = hidden_dim
        self.n_masks = int(n_masks)
        self.mask_temperature = float(mask_temperature)

        # Cached activations are raw ReLU outputs with wildly different channel
        # scales; batch statistics standardize them without a learned gain.
        self.input_norm = nn.BatchNorm2d(n_channels, affine=False)
        core_dim = core_dim or hidden_dim
        layers, in_channels = [], n_channels
        for _ in range(core_layers):
            layers += [
                nn.Conv2d(in_channels, core_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(core_dim),
                nn.GELU(),
            ]
            in_channels = core_dim
        # end for core convolution
        self.core = nn.Sequential(*layers)
        self.channel_mixer = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

        # Near-uniform logits start every site reading the whole map, so masks
        # sharpen only where the data ask for it.
        self.spatial_logits = nn.Parameter(
            torch.randn(n_neurons, self.n_masks, self.n_positions) * 0.01
        )
        # The masks see gradient only through these weights, so exact zeros
        # would freeze them; 1e-3 still starts the branch near zero output.
        self.feature_weights = nn.Parameter(
            torch.randn(n_timepoints, n_neurons, self.n_masks, hidden_dim) * 1e-3
        )
        self.output_bias = nn.Parameter(torch.zeros(n_timepoints, n_neurons))

    """
    spatial_masks
    Return every site's masks as maps, each summing to one.

    OUTPUT:
        - masks: torch.Tensor -> [neurons, n_masks, height, width]
    """
    def spatial_masks(self):
        masks = torch.softmax(
            self.spatial_logits / self.mask_temperature, dim=-1
        )
        return masks.reshape(self.n_neurons, self.n_masks, self.height, self.width)
    # EOF

    def forward(self, feature_map, return_diagnostics=False):
        expected = (self.n_channels, self.height, self.width)
        if feature_map.ndim != 4 or tuple(feature_map.shape[1:]) != expected:
            raise ValueError(
                f"Expected [batch, {expected[0]}, {expected[1]}, {expected[2]}], "
                f"got {tuple(feature_map.shape)}."
            )
        # end if the feature map has the wrong geometry
        coded = self.core(self.input_norm(feature_map))
        mixed = self.dropout(torch.nn.functional.gelu(self.channel_mixer(coded)))
        # [batch, hidden, positions]
        features = mixed.flatten(start_dim=2)
        masks = torch.softmax(
            self.spatial_logits / self.mask_temperature, dim=-1
        )
        # Each site pools the map once per mask: [batch, neurons, masks, hidden]
        pooled = torch.einsum("bhp,nmp->bnmh", features, masks)
        predictions = torch.einsum("bnmh,tnmh->btn", pooled, self.feature_weights)
        predictions = predictions + self.output_bias

        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                "spatial_masks": self.spatial_masks(),
                # A site-independent summary of the core, used by the feature
                # reconstruction auxiliary.
                "core_code": mixed.mean(dim=(2, 3)),
            }
        # end if diagnostics are requested
        return predictions, diagnostics
    # EOF

    """
    global_code
    Globally pool the core, for auxiliary heads that are not site specific.

    INPUT:
        - feature_map: torch.Tensor -> [batch, channels, height, width]

    OUTPUT:
        - code: torch.Tensor -> [batch, hidden_dim]
    """
    def global_code(self, feature_map):
        coded = self.core(self.input_norm(feature_map))
        return torch.nn.functional.gelu(self.channel_mixer(coded)).mean(dim=(2, 3))
    # EOF
# EOC


class MultiBranchTimebinDecoder(nn.Module):
    """
    Sum a pooled-feature branch and any number of spatial readouts.

    Each branch predicts the whole [time, sites] response and the predictions
    are added, so no branch is fitted to another's leftovers. An optional
    reconstruction head regresses a frozen pooled feature vector out of the
    first spatial branch's core, which is an auxiliary objective rather than an
    extra input.

    INPUT (forward):
        - layer_features: torch.Tensor -> [batch, layers, embedding], may be empty
        - feature_maps: list[torch.Tensor] -> one map per spatial branch
        - return_diagnostics: bool -> whether to return per-branch predictions

    OUTPUT:
        - predictions: torch.Tensor -> [batch, time, neurons]
        - diagnostics: dict | None -> branch predictions, masks, reconstruction
    """

    """
    __init__
    Build the requested branches on a shared target geometry.

    INPUT:
        - temporal_architecture: str | None -> key in TIMEBIN_MODEL_CLASSES
        - temporal_kwargs: dict -> arguments of the pooled branch
        - spatial_specs: list[dict] -> one MultiMaskSpatialReadout kwargs each
        - reconstruction_dim: int -> width of the auxiliary target, 0 to disable

    OUTPUT:
        - None
    """
    def __init__(
        self,
        temporal_architecture,
        temporal_kwargs,
        spatial_specs,
        reconstruction_dim=0,
    ):
        super().__init__()
        if temporal_architecture is None and not spatial_specs:
            raise ValueError("At least one branch must be enabled.")
        # end if the decoder would have no input at all
        self.temporal_branch = (
            build_timebin_model(temporal_architecture, **temporal_kwargs)
            if temporal_architecture is not None
            else None
        )
        self.spatial_branches = nn.ModuleList(
            [MultiMaskSpatialReadout(**spec) for spec in spatial_specs]
        )
        self.reconstruction_head = (
            nn.Linear(self.spatial_branches[0].hidden_dim, reconstruction_dim)
            if reconstruction_dim > 0 and len(self.spatial_branches) > 0
            else None
        )

    """
    mask_parameter_names
    Names of the mask logits, which need their own optimizer group.

    OUTPUT:
        - names: list[str] -> parameter names of every spatial mask
    """
    def mask_parameter_names(self):
        return [
            name
            for name, _ in self.named_parameters()
            if name.endswith("spatial_logits")
        ]
    # EOF

    def forward(self, layer_features, feature_maps, return_diagnostics=False):
        branch_predictions = {}
        if self.temporal_branch is not None:
            branch_predictions["temporal"] = self.temporal_branch(layer_features)[0]
        # end if the pooled branch is enabled
        for position, branch in enumerate(self.spatial_branches):
            branch_predictions[f"spatial_{position}"] = branch(
                feature_maps[position]
            )[0]
        # end for spatial branch

        predictions = sum(branch_predictions.values())
        diagnostics = None
        if return_diagnostics:
            diagnostics = dict(branch_predictions)
            if self.spatial_branches:
                diagnostics["spatial_masks"] = self.spatial_branches[0].spatial_masks()
            # end if a spatial branch owns masks
        # end if diagnostics are requested

        if self.reconstruction_head is not None:
            code = self.spatial_branches[0].global_code(feature_maps[0])
            reconstruction = self.reconstruction_head(code)
            diagnostics = diagnostics or {}
            diagnostics["reconstruction"] = reconstruction
        # end if the reconstruction auxiliary is enabled
        return predictions, diagnostics
    # EOF
# EOC
