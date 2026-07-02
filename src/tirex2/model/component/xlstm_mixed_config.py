from dataclasses import dataclass, field

from xlstm.xlstm_large import xLSTMLargeConfig


@dataclass
class xLSTMMixedConfig(xLSTMLargeConfig):
    """Extends xlstm config with controls for mixing sLSTM and mLSTM blocks."""

    slstm_at: list[int] = field(default_factory=list)
    rnn_type: str = "slstm"
    rnn_kwargs: dict = field(default_factory=dict)
    rnn_backend: str = "original"
    vocab_size: int = 0
    weight_mode: str = "single"
    conv1d_kernel_size: int = 0
    conv1d_channel_mixing: bool = False
    gradient_recurrent_clipval: float = None
    default_block: str = "m"  # s = sLSTM, #m = mLSTM, t = transformer
    return_last_states: bool = False
    num_slstm_heads: int = 4
    disable_singleton_attention: bool = False
    use_rope: bool = True

    @property
    def block_types(self):
        """Return a list marking each block as sLSTM (``s``) or mLSTM (``m``) or transformer (``t``)"""
        return ["s" if i in self.slstm_at else self.default_block for i in range(self.num_blocks)]
