# -\-\-\ Define the architecture of the model /-/-/-
# -------------------- Import libraries --------------------
import torch
import torch.nn as nn

    
# -------------------- LSTM langauge model --------------------
class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, 
                 out_dropout=0.1, emb_dropout=0.1, n_layers=1):
        super(LM_LSTM, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)

        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, 
                           bidirectional=False, batch_first=True)

        self.pad_token = pad_index
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)

        rnn_out, _ = self.rnn(emb)

        output = self.output(rnn_out).permute(0, 2, 1)
        return output

# --- 1. Apply Weight Tying
class LM_LSTM_WEIGHT_TYING(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, 
                 out_dropout=0.1, emb_dropout=0.1, n_layers=1):
        super(LM_LSTM_WEIGHT_TYING, self).__init__()
        assert emb_size == hidden_size, "Weight tying requires emb_size == hidden_size"
        
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)

        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, 
                           bidirectional=False, batch_first=True)

        self.pad_token = pad_index
        # Commented as output will be computed via tied embedding weights
        #self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)

        rnn_out, _ = self.rnn(emb)

        output = torch.matmul(rnn_out, self.embedding.weight.T).permute(0, 2, 1)
        return output

# --- 2. Apply Variational Dropout (no DropConnect)
# This class applies the same dropout mask every time dropout is performed.
class LockedDropout(nn.Module):
    #Applies the same dropout mask.
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.1):
        if not self.training or dropout == 0:
            return x
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = mask.div_(1 - dropout)
        mask = mask.expand_as(x)
        return x * mask
    
class LM_LSTM_VAR_DROPOUT(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, 
                 out_dropout=0.1, emb_dropout=0.1, n_layers=1):
        super(LM_LSTM_VAR_DROPOUT, self).__init__()
        assert emb_size == hidden_size, "Weight tying requires emb_size == hidden_size"

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.emb_dropout = LockedDropout()
        self.out_dropout = LockedDropout()

        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, 
                           bidirectional=False, batch_first=True)

        self.pad_token = pad_index

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb = self.emb_dropout(emb, dropout=0.1)

        rnn_out, _ = self.rnn(emb)
        rnn_out = self.out_dropout(rnn_out, dropout=0.1)

        output = torch.matmul(rnn_out, self.embedding.weight.T).permute(0, 2, 1)
        return output