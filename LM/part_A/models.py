# Model architecture definitions
import torch.nn as nn

# Custom Elman RNN cell implementation
class RNN_cell(nn.Module):
    def __init__(self, hidden_size, input_size, vocab_size):
        super(RNN_cell, self).__init__()
        
        self.W = nn.Linear(input_size, hidden_size, bias=False)
        self.U = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size
        self.sigmoid = nn.Sigmoid()
    
    # Compute hidden state and output for one time step
    def forward(self, prev_hidden, word):
        input_emb = self.W(word)
        prev_hidden_rep = self.U(prev_hidden)

        hidden_state = self.sigmoid(input_emb + prev_hidden_rep)

        output = self.output(hidden_state)
        return hidden_state, output

# Baseline RNN language model using PyTorch RNN layer
class LM_RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, 
                n_layers=1):
        super(LM_RNN, self).__init__()

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)

        self.rnn = nn.RNN(
            emb_size,
            hidden_size,
            n_layers,
            bidirectional=False,
            batch_first=True
        )

        self.pad_token = pad_index
        self.output = nn.Linear(hidden_size, output_size)
        
    # Perform forward pass through embedding, RNN, and output layers
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        rnn_out, _ = self.rnn(emb)
        output = self.output(rnn_out).permute(0, 2, 1)
        return output

# LSTM-based language model
class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, 
                n_layers=1):
        super(LM_LSTM, self).__init__()

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)

        self.rnn = nn.LSTM(
            emb_size,
            hidden_size,
            n_layers,
            bidirectional=False,
            batch_first=True
        )

        self.pad_token = pad_index
        self.output = nn.Linear(hidden_size, output_size)

    # Perform forward pass through embedding, LSTM, and output layers
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)

        rnn_out, _ = self.rnn(emb)

        output = self.output(rnn_out).permute(0, 2, 1)
        return output

# LSTM language model with embedding and output dropout
class LM_LSTM_DROPOUT(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0,
                emb_dropout=0.1, out_dropout=0.1, n_layers=1):
        super(LM_LSTM_DROPOUT, self).__init__()

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.rnn = nn.LSTM(
            emb_size,
            hidden_size,
            n_layers,
            bidirectional=False,
            batch_first=True
        )

        self.pre_output_dropout = nn.Dropout(out_dropout)

        self.pad_token = pad_index
        self.output = nn.Linear(hidden_size, output_size)

    # Perform forward pass with dropout regularization
    def forward(self, input_sequence):
        emb = self.emb_dropout(self.embedding(input_sequence))

        rnn_out, _ = self.rnn(emb)

        dropped = self.pre_output_dropout(rnn_out)
        
        output = self.output(dropped).permute(0, 2, 1)
        return output