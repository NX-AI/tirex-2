"""Feed-forward network module with optional residual connection."""

import torch
from torch import nn


class MLP(nn.Module):
    """Multi-layer perceptron with single hidden layer.

    Parameters
    ----------
    d_model : int
        Input and output dimension
    d_ff : int
        Hidden layer dimension
    dropout : float
        Dropout probability applied after activation
    act_fn : nn.Module, optional
        Activation function (default: nn.ReLU())
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float, act_fn: nn.Module = nn.ReLU()):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout

        self.wi = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.act_fn = act_fn

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply MLP transformation.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input tensor of shape [..., d_model]

        Returns
        -------
        torch.Tensor
            Output tensor of shape [..., d_model]
        """
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states
