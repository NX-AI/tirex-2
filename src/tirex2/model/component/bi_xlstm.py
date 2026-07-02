"""This module exposes xLSTM blocks combining recurrent kernels with feed-forward adapters."""

from typing import Literal

import torch
from torch import nn

# From xlstm large
from xlstm.xlstm_large.components import RMSNorm
from xlstm.xlstm_large.model import FeedForward

from .flashrnn_slstm import init_cell as init_slstm_cell
from .mlstm_block import init_cell as init_mlstm_cell
from .xlstm_mixed_config import xLSTMMixedConfig


class BiXLSTM(nn.Module):
    """Bi-directional xLSTM block that splits variates into forward and reverse subsets,
    runs a separate xLSTM cell over each, and recombines the results.

    The cell type is controlled by ``cell_type``:
      - ``"slstm"``: Uses sLSTM cells (original behaviour, equivalent to the former BiSLSTM).
      - ``"mlstm"``: Uses mLSTM cells.

    The split is controlled by ``split_method``:
      - ``"reverse_wo_target"``: The forward cell sees all variates, while the
        reverse cell sees only the non-target covariates. The non-target variates
        thus appear in both directions and are shared.
      - ``"forward_only_target"``: The forward cell sees only the target variates
        and the reverse cell sees only the non-target covariates. No variates are
        shared between the two directions.
      - ``"reverse_known_only"``: The forward cell sees all variates, while the
        reverse cell sees only the known covariates (those with non-NaN values in
        the future window). Past-only covariates are processed forward-only.

    Variates that appear in both directions have their forward and reverse
    representations concatenated along the embedding dimension (2*D) and
    downprojected back to D via a linear layer with dropout. Variates that appear
    in only one direction keep their single-direction output unchanged. Residual
    connections wrap both the recurrent stage and the subsequent feed-forward block.

    Parameters
    ----------
    config : xLSTMMixedConfig
        Shared xLSTM configuration object.
    block_idx : int
        Index of this block in the overall stack.
    num_blocks : int
        Total number of blocks in the stack.
    device : {"cpu", "cuda"}
        Device used to choose recurrent kernels.
    dropout : float
        Dropout probability applied in the combination projection.
    split_method : {"forward_only_target", "reverse_wo_target", "reverse_known_only"}
        Strategy for splitting variates between forward and reverse cells.
    share_weights : bool
        When True, the forward and reverse cells share the same parameters.
    cell_type : {"slstm", "mlstm"}
        Which recurrent cell to instantiate.
    """

    def __init__(
        self,
        config: xLSTMMixedConfig,
        block_idx: int,
        num_blocks: int,
        device: Literal["cpu", "cuda"],
        dropout: float = 0.0,
        share_weights: bool = True,
        cell_type: Literal["slstm", "mlstm"] = "slstm",
    ):
        super().__init__()
        self.cell_type = cell_type
        self.norm_lstm = RMSNorm(
            num_features=config.embedding_dim,
            eps=config.norm_eps,
            use_weight=True,
            use_bias=config.use_bias,
            force_float32_reductions=config.norm_reduction_force_float32,
        )
        self._split = self._split_variates
        self._recombine = self._recombine_variates

        match cell_type:
            case "slstm":
                self.fwd_cell = init_slstm_cell(config, block_idx, num_blocks, device)
            case "mlstm":
                self.fwd_cell = init_mlstm_cell(config, device)
            case _:
                raise ValueError(f"Unknown cell type '({cell_type})'. Allowed 'mlstm' or 'slstm'.")

        self.rev_cell = self.fwd_cell

        self.combination = nn.Linear(2 * config.embedding_dim, config.embedding_dim, bias=False)
        self.combination_dropout = nn.Dropout(dropout)

        self.norm_ffn = RMSNorm(
            num_features=config.embedding_dim,
            eps=config.norm_eps,
            use_weight=True,
            use_bias=config.use_bias,
            force_float32_reductions=config.norm_reduction_force_float32,
        )
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor, target_mask: torch.Tensor, known_covariate_mask: torch.Tensor | None = None):
        """Run the bidirectional xLSTM block."""
        x_norm = self.norm_lstm(x)

        x_fwd, x_rev, fwd_idc, rev_idc = self._split(x_norm, target_mask, known_covariate_mask)

        x_fwd = self.fwd_cell(x_fwd) if x_fwd.shape[0] > 0 else x_fwd
        if x_rev.shape[0] > 0:
            x_rev_flipped = torch.flip(x_rev, dims=(1,)).contiguous()
            x_rev_out = self.rev_cell(x_rev_flipped)
            x_rev = torch.flip(x_rev_out, dims=(1,))

        out = self._recombine(x_fwd, x_rev, fwd_idc, rev_idc, x)

        # residual connection around lstm + combination
        out = x + out

        # ffn with residual connection
        out = out + self.ffn(self.norm_ffn(out))

        return out

    def _split_variates(
        self,
        x: torch.Tensor,
        target_mask: torch.Tensor,
        known_covariate_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward cell sees all variates, reverse sees only known covariates (non-NaN future)."""
        if known_covariate_mask is None:
            known_idc = torch.where(~target_mask)[0]
        else:
            known_idc = torch.where(known_covariate_mask)[0]
        return x, x[known_idc], torch.arange(x.shape[0], device=x.device), known_idc

    def _recombine_variates(
        self,
        x_fwd: torch.Tensor,
        x_rev: torch.Tensor,
        fwd_idc: torch.Tensor,
        rev_idc: torch.Tensor,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        """Covariates appear in both directions and are combined; targets are fwd-only."""
        # combine covariate representations from both directions
        x_shared = torch.cat([x_fwd[rev_idc], x_rev], dim=-1)
        x_shared = self.combination_dropout(self.combination(x_shared))

        # start from fwd output (all variates), overwrite covariates with combined
        out = x_fwd.to(ref.dtype)
        out[rev_idc] = x_shared.to(ref.dtype)
        return out
