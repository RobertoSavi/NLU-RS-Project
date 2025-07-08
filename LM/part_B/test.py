import torch
import torch.nn as nn

# Parameters
batch_size = 2
seq_len = 5
input_dim = 10
hidden_dim = 20
num_layers = 2

# Dummy input: (batch, seq_len, input_dim)
x = torch.randn(batch_size, seq_len, input_dim)

# LSTM with dropout between layers
lstm_dropout = nn.LSTM(
    input_size=input_dim,
    hidden_size=hidden_dim,
    num_layers=num_layers,
    batch_first=True,
    dropout=0.5  # dropout between LSTM layers
)

# LSTM without dropout
lstm_no_dropout = nn.LSTM(
    input_size=input_dim,
    hidden_size=hidden_dim,
    num_layers=num_layers,
    batch_first=True,
    dropout=0.0  # no dropout
)

# Set to train mode so dropout is active
lstm_dropout.train()
lstm_no_dropout.train()

# Forward pass
out_dropout, _ = lstm_dropout(x)
out_nodrop, _ = lstm_no_dropout(x)

# Compare outputs
print("Output with dropout between LSTM layers:")
print(out_dropout)

print("\nOutput with NO dropout between LSTM layers:")
print(out_nodrop)

print("\nDifference (dropout vs no dropout):")
print(torch.abs(out_dropout - out_nodrop).mean())