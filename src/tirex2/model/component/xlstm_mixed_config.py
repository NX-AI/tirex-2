# Copyright (c) NXAI GmbH.
# Licensed under the Apache License, Version 2.0; see LICENSE for details.

from dataclasses import dataclass, field


@dataclass
class xLSTMLargeConfig:
    embedding_dim: int
    """Embedding dimension of the model."""
    num_heads: int
    """Number of heads."""
    num_blocks: int
    """Number of blocks."""
    vocab_size: int
    """Vocabulary size."""
    use_bias: bool = False
    """Whether to use bias in linear layers."""
    norm_eps: float = 1e-6
    """Epsilon value for numerical stability in the normalization layers."""
    norm_reduction_force_float32: bool = True
    """Whether to force float32 reductions in the normalization layers."""
    add_out_norm: bool = True
    """Whether to add a normalization layer after the block stack."""

    # mlstm layer
    qk_dim_factor: float = 0.5
    """The factor to determine the dimension of the query and key tensors."""
    v_dim_factor: float = 1.0
    """The factor to determine the dimension of the value tensor."""

    # mlstm backend
    chunkwise_kernel: str = "chunkwise--triton_limit_chunk"
    """Kernel to use for chunkwise parallel processing of the sequence.
    Also supports fully parallel (i.e. quadratic) backends for comparison.
    E.g. 'parallel--native_autograd'.
    """
    sequence_kernel: str = "native_sequence__triton"
    """The sequence kernel to use for processing sequneces step-by-step.
    Used only for parts of the prefill sequence in inference mode.
    """
    step_kernel: str = "triton"
    """The step kernel to use for processing a single step.
    Used for generation in inference mode.
    """
    mode: str = "train"
    """The mode of operation for the backend. Determines how the `forward` method behaves.
    Available modes are 'train', 'train_with_padding', 'inference'.
    'inference' works with arbitrary sequence lengths, and does not support training.
    It calls a sequence of different kernels to process the sequence.
    'train_with_padding' pads the input to multiples of `chunk_size`.
    """
    chunk_size: int = 64
    """The chunk size of the chunkwise kernel.
    If `mode` is 'train_with_padding', the inputs are padded to multiples of this size.
    """
    return_last_states: bool = False
    """Whether to return the last states of the sequence in training mode.
    Inference mode always returns the last states.
    """
    autocast_kernel_dtype: str = "bfloat16"
    """The dtype to use for autocast behavior in the kernel.
    If autocast is enabled all inputs are cast to this dtype before the kernel is called.
    """
    eps: float = 1e-6
    """Epsilon value for numerical stability in the kernel."""
    inference_state_dtype: str = "float32"
    """The dtype to use for the state tensors in inference mode."""
    # feedforward
    ffn_proj_factor: float = 2.6667
    """The factor to determine the dimension of the intermediate projection in the feedforward layer."""
    ffn_round_up_to_multiple_of: int = 64
    """Round the intermediate projection dimension to the next multiple of this value."""

    # capping
    gate_soft_cap: float = 15.0
    """Soft cap value for the gates."""
    output_logit_soft_cap: float = 30.0
    """Soft cap value for the output logits."""

    weight_mode: str = "single"
    """The weight mode to use for the mLSTM layer.
    Mode 'single' uses separate weights for the query, key, value, and gates.
    Mode 'fused' uses a single weight matrix for the query, key, value, and gates.
    'fused' is benefitial in inference settings.
    """


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
