"""Custom two-layer LSTM classifier implemented from scratch.

This module intentionally avoids `torch.nn.LSTM` and implements the cell
update equations manually using linear layers and element-wise operations.

Target behavior-service shape contract:
- Input:  (batch_size, seq_len, input_size)
- Output: (batch_size, num_classes=4)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn as nn


@dataclass(frozen=True)
class LSTMState:
    """Container for hidden and cell states of one LSTM layer."""

    h: Tensor
    c: Tensor


class CustomLSTMCell(nn.Module):
    """Single LSTM cell with explicit gate computations.

    Equations:
        i_t = sigmoid(W_ii x_t + W_hi h_{t-1} + b_i)
        f_t = sigmoid(W_if x_t + W_hf h_{t-1} + b_f)
        g_t = tanh   (W_ig x_t + W_hg h_{t-1} + b_g)
        o_t = sigmoid(W_io x_t + W_ho h_{t-1} + b_o)

        c_t = f_t * c_{t-1} + i_t * g_t
        h_t = o_t * tanh(c_t)
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input-to-hidden and hidden-to-hidden projections for all 4 gates.
        self.x_proj = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.h_proj = nn.Linear(hidden_size, 4 * hidden_size, bias=True)

    def forward(self, x_t: Tensor, state: LSTMState) -> LSTMState:
        """Run one time-step update.

        Args:
            x_t: Current input step, shape (batch_size, input_size)
            state: Previous LSTMState for this layer

        Returns:
            New LSTMState after processing x_t.
        """
        gates = self.x_proj(x_t) + self.h_proj(state.h)
        i_t, f_t, g_t, o_t = gates.chunk(4, dim=-1)

        i_t = torch.sigmoid(i_t)
        f_t = torch.sigmoid(f_t)
        g_t = torch.tanh(g_t)
        o_t = torch.sigmoid(o_t)

        c_t = (f_t * state.c) + (i_t * g_t)
        h_t = o_t * torch.tanh(c_t)

        return LSTMState(h=h_t, c=c_t)


class CustomLSTMClassifier(nn.Module):
    """Two-layer custom LSTM classifier for 4-class behavior prediction.

    Notes:
    - Does NOT use nn.LSTM.
    - Uses two explicit CustomLSTMCell layers stacked depth-wise.
    - Uses the final time-step hidden state of layer-2 for classification.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # Requirement: hidden layers = 2.
        self.layer1 = CustomLSTMCell(input_size=input_size, hidden_size=hidden_size)
        self.layer2 = CustomLSTMCell(input_size=hidden_size, hidden_size=hidden_size)

        # Fully connected output head for 4 behavior classes.
        self.classifier = nn.Linear(hidden_size, num_classes)

    def _init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> LSTMState:
        """Create zero-initialized hidden and cell states for one layer."""
        zeros = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        return LSTMState(h=zeros, c=zeros)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through two custom LSTM layers and classifier head.

        Args:
            x: Input tensor with shape (batch_size, seq_len, input_size)

        Returns:
            Logits with shape (batch_size, num_classes)
        """
        if x.dim() != 3:
            raise ValueError("Expected input shape (batch_size, seq_len, input_size)")

        batch_size, seq_len, feat_dim = x.shape
        if feat_dim != self.input_size:
            raise ValueError(
                f"Expected input feature size {self.input_size}, got {feat_dim}"
            )

        state1 = self._init_state(batch_size, x.device, x.dtype)
        state2 = self._init_state(batch_size, x.device, x.dtype)

        # Unroll sequence manually over time.
        for t in range(seq_len):
            x_t = x[:, t, :]
            state1 = self.layer1(x_t, state1)
            state2 = self.layer2(state1.h, state2)

        # Use final hidden representation from layer-2.
        logits = self.classifier(state2.h)
        return logits


if __name__ == "__main__":
    # Smoke test for shape sanity.
    model = CustomLSTMClassifier(input_size=10, hidden_size=64, num_classes=4)
    sample = torch.randn(8, 30, 10)  # batch=8, seq_len=30, input_size=10
    out = model(sample)
    print("Output shape:", tuple(out.shape))  # Expected: (8, 4)
