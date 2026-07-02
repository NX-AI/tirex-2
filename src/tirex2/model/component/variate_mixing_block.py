from dataclasses import dataclass
from typing import Any, Literal

import torch
from einops import rearrange
from torch import nn

from .attention_block import AttentionBlock
from .bi_xlstm import BiXLSTM
from .xlstm_mixed_config import xLSTMMixedConfig


def _unwrap_output(output: torch.Tensor | tuple[torch.Tensor, Any]) -> tuple[torch.Tensor, Any]:
    """Unwrap module output that may be a tensor or (tensor, state) tuple.

    Parameters
    ----------
    output : torch.Tensor or tuple[torch.Tensor, Any]
    Module output, either a single tensor or (tensor, state) tuple

    Returns
    -------
    tuple[torch.Tensor, Any]
    Tuple of (tensor, state). If input is a tensor, state is None.
    """
    if isinstance(output, tuple):
        return output
    return output, None


@dataclass
class TimeMixerConfig:
    """Configuration for time mixing model."""

    model_type: Literal["bi-slstm", "bi-mlstm"]

    act_fn: nn.Module

    embedding_dim: int
    num_heads: int = 4
    num_slstm_heads: int | None = None

    device: Literal["cpu", "cuda"] = "cuda"


@dataclass
class VariateMixerConfig:
    """Configuration for variate mixing model."""

    act_fn: nn.Module

    # dimensions
    embedding_dim: int
    num_heads: int = 4

    # whether to RMS-normalize query/key vectors in the attention layer
    use_qk_norm: bool = True

    # whether to use FlexAttention in the attention layer
    use_flex_attention: bool = False


@dataclass
class MultivariateBlockConfig:
    """Configuration for the MultivariateBlock.

    This block performs two-stage mixing:
        1. Time mixing: processes [B*V, L, P] - mixing across time dimension
        2. Variate mixing: processes [B*L, V, P] - mixing across variate dimension

    Normalization and residual connections are handled by the individual mixer components.
    """

    time_mixer: TimeMixerConfig
    variate_mixer: VariateMixerConfig
    block_order: list[str] | None = None

    # To initialize the xLSTM blocks
    block_idx: int | None = None
    num_blocks: int | None = None

    # Optional feed-forward network after variate mixing
    dropout: float = 0.1
    eps: float = 1e-6


class MultivariateBlock(nn.Module):
    """Two-stage mixing block for multivariate time series.

    Stage 1 (Time Mixing): Processes each variate's time series independently [B*V, L, P]
    Stage 2 (Variate Mixing): Reshapes to [L, B*V, P] and mixes across variates for each timestep.
    Stage 3 (FFN): Final feed-forward transformation on tokens.

    Parameters
    ----------
    config : MultivariateBlockConfig
    Configuration specifying time mixer, variate mixer, and FFN settings
    """

    def __init__(self, config: MultivariateBlockConfig) -> None:
        super().__init__()

        self.dropout = config.dropout if config.dropout is not None else 0.0

        self.eps = config.eps
        self.config = config

        # Build sub-blocks
        self.time_mixer = self._build_time_mixer(
            config.time_mixer,
            block_idx=config.block_idx,
            num_blocks=config.num_blocks,  # type: ignore
        )
        self.variate_mixer = self._build_variate_mixer(
            config.variate_mixer,
            block_idx=config.block_idx,
            num_blocks=config.num_blocks,  # type: ignore
        )

    def forward(
        self,
        x: torch.Tensor,
        group_vector: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
        known_covariate_mask: torch.Tensor | None = None,
        state: dict | None = None,
        *,
        return_svd: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]] | tuple[torch.Tensor, dict[str, Any], dict[str, torch.Tensor]]:
        """Forward pass with three-stage mixing.

        Normalization and residual connections are handled by the individual mixer components.

        Parameters
        ----------
        x : torch.Tensor
        Input tensor of shape [B*V, L, P] where:
            B = batch size
            V = number of variates
            L = sequence length
            P = feature dimension
            group_vector : torch.Tensor, optional
            Optional group ids used when attention is the variate mixer. Should broadcast to the
            variate dimension after reshaping (e.g., [B*V] or [B*V, 1])
            state : dict, optional
            Optional state returned from a previous call. Keys: ``time_mixer`` and ``variate_mixer``
            return_svd : bool, optional
            Enables computation of the singular values on the first batch element in each sub-block.
            `torch.svd_lowrank` is used to approximately compute the singular values (default: False)

        Returns
        -------
        tuple[torch.Tensor, dict[str, Any]] or tuple[torch.Tensor, dict[str, Any], dict[str, torch.Tensor]]
        If return_svd is False, returns (tokens, state) where tokens has shape [B*V, L, P]
        and state is a dict with keys 'time_mixer' and 'variate_mixer'.
        If return_svd is True, returns (tokens, state, svd_vals) where svd_vals contains
        the singular values of the first batch element for each sub-block.
        """
        BV, L, P = x.shape

        # Stage 1: Time Mixing [B*V, L, P] -> [B*V, L, P]
        # Each variate is processed independently across its time dimension
        if known_covariate_mask is not None and isinstance(self.time_mixer, BiXLSTM):
            time_output, _ = _unwrap_output(
                self.time_mixer(x, target_mask=target_mask, known_covariate_mask=known_covariate_mask)
            )
        else:
            time_output, _ = _unwrap_output(self.time_mixer(x, target_mask=target_mask))

        # Stage 2: Variate Mixing
        # Transpose for variate mixing: [B*V, L, P] -> [L, B*V, P]
        # This treats L as the new batch dimension and mixes across B*V (all variates)
        x_variate = rearrange(time_output, "bv l p -> l bv p", bv=BV, l=L, p=P)  # [L, B*V, P]

        # Apply variate mixing
        variate_output, _ = _unwrap_output(
            self.variate_mixer(x_variate, group_vector=group_vector, target_mask=target_mask)
        )

        # Transpose back: [L, B*V, P] -> [B*V, L, P]
        x = rearrange(variate_output, "l bv p -> bv l p", l=L, bv=BV, p=P)

        return x, {}  # TODO: return state as well for streaming

    def _build_time_mixer(self, config: TimeMixerConfig, block_idx: int, num_blocks: int) -> nn.Module:
        """Instantiate time mixing model from configuration.

        Parameters
        ----------
        config : TimeMixerConfig
        Configuration for the time mixer
        block_idx : int
        Index of this block in the overall stack
        num_blocks : int
        Total number of blocks in the stack

        Returns
        -------
        nn.Module
        Time mixing module (sLSTM, mLSTM, or attention)
        """
        cfg = dict(
            vocab_size=0,
            num_heads=config.num_heads,
            num_blocks=num_blocks,
            embedding_dim=config.embedding_dim,
            return_last_states=True,
            slstm_at=[],
        )

        match config.model_type:
            case "bi-slstm":
                num_slstm_heads = config.num_slstm_heads if config.num_slstm_heads is not None else config.num_heads
                cfg.update({"num_slstm_heads": num_slstm_heads})
                cell_type = "slstm"

            case "bi-mlstm":
                cfg.update(dict(return_last_states=False, mode="train_with_padding", use_rope=False))
                cell_type = "mlstm"
            case _:
                raise ValueError(
                    f"Unknown model_type for time mixer '{config.model_type}'. Allowed 'bi-slstm' and 'bi-mlstm'"
                )

        return BiXLSTM(
            xLSTMMixedConfig(**cfg),
            block_idx=block_idx,
            num_blocks=num_blocks,
            device=config.device,
            dropout=self.dropout,
            cell_type=cell_type,
        )

    def _build_variate_mixer(self, config: VariateMixerConfig, num_blocks: int, block_idx: int) -> nn.Module:
        """Instantiate variate mixing model from configuration.

        Parameters
        ----------
        config : VariateMixerConfig
            Configuration for the variate mixer
        num_blocks : int
            Total number of blocks in the stack
        block_idx : int
            Index of this block in the overall stack

        Returns
        -------
        nn.Module
            Variate mixing module (attention, mLSTM, or identity)
        """
        model = AttentionBlock(
            input_dim=config.embedding_dim,
            n_heads=config.num_heads,
            act_fn=config.act_fn,
            dropout=self.dropout,
            use_qk_norm=config.use_qk_norm,
            use_flex_attention=config.use_flex_attention,
        )

        return model
