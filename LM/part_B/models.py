# Model architecture definitions
import torch.nn as nn
  
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

# LSTM model with weight tying between embedding and output layers
class LM_LSTM_WEIGHT_TYING(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, 
                n_layers=1):
        super(LM_LSTM_WEIGHT_TYING, self).__init__()
        
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)

        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, 
                           bidirectional=False, batch_first=True)

        self.pad_token = pad_index
        if emb_size == hidden_size:
            self.proj = None
            self.output = nn.Linear(hidden_size, output_size)
            self.output.weight = self.embedding.weight
        else:
            # If hidden size and embedding size differ, add a projection layer to match dimensions before output layer
            self.proj = nn.Linear(hidden_size, emb_size)
            self.output = nn.Linear(emb_size, output_size)
            self.output.weight = self.embedding.weight
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        
        rnn_out, _ = self.rnn(emb)

        if self.proj is not None:
            rnn_out = self.proj(rnn_out)

        output = self.output(rnn_out).permute(0, 2, 1)
        return output

# This class applies the same dropout mask every time dropout is performed.
class VariationalDropout(nn.Module):
    def __init__(self, dropout): 
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        if not self.training or self.dropout == 0:
            return x
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - self.dropout)
        mask = mask.div_(1 - self.dropout)
        mask = mask.expand_as(x)
        return x * mask
    
# LSTM model with variational dropout (same dropout mask for all time steps) and weight tying
class LM_LSTM_VAR_DROPOUT(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0,
                 emb_dropout=0.1, out_dropout=0.1, n_layers=1):
        super(LM_LSTM_VAR_DROPOUT, self).__init__()

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)

        self.emb_dropout = VariationalDropout(emb_dropout)
        self.pre_output_dropout = VariationalDropout(out_dropout)

        self.rnn = nn.LSTM(
            emb_size,
            hidden_size,
            n_layers,
            bidirectional=False,
            batch_first=True
        )

        self.pad_token = pad_index
        
        if emb_size == hidden_size:
            self.proj = None
            self.output = nn.Linear(hidden_size, output_size)
            self.output.weight = self.embedding.weight
        else:
            # If hidden size and embedding size differ, add a projection layer to match dimensions before output layer
            self.proj = nn.Linear(hidden_size, emb_size)
            self.output = nn.Linear(emb_size, output_size)
            self.output.weight = self.embedding.weight

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb = self.emb_dropout(emb)

        rnn_out, _ = self.rnn(emb)

        dropped = self.pre_output_dropout(rnn_out)
        
        if self.proj is not None:
            dropped = self.proj(dropped)
        
        output = self.output(dropped).permute(0, 2, 1)
        return output