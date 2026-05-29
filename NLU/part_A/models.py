# Model architecture definitions
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ModelIAS(nn.Module):
    def __init__(self, emb_size, hid_size, vocab_len, out_slot, out_int, n_layer=1, pad_index=0):
        super(ModelIAS, self).__init__()
        
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        
        self.utt_encoder = nn.LSTM(
            emb_size,
            hid_size,
            n_layer,
            bidirectional=False,
            batch_first=True
        )
        self.slot_out = nn.Linear(hid_size, out_slot)
        self.intent_out = nn.Linear(hid_size, out_int)
        
    def forward(self, utterance, seq_lengths):
        utt_emb = self.embedding(utterance) 
        
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_output, (last_hidden, _) = self.utt_encoder(packed_input) 
       
        utt_encoded, _ = pad_packed_sequence(packed_output, batch_first=True)

        last_hidden = last_hidden[-1,:,:]
        
        slots = self.slot_out(utt_encoded)
        intent = self.intent_out(last_hidden)

        slots = slots.permute(0,2,1) 
        return slots, intent
    
    
class ModelIAS_Bi(nn.Module):
    def __init__(self, emb_size, hid_size, vocab_len, out_slot, out_int, n_layer=1, pad_index=0):
        super(ModelIAS_Bi, self).__init__()
        
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        
        self.utt_encoder = nn.LSTM(
            emb_size,
            hid_size,
            n_layer,
            bidirectional=True,
            batch_first=True
        )
        
        # Multiply hid_size by 2 to account for forward + backward passes
        self.slot_out = nn.Linear(hid_size * 2, out_slot)
        self.intent_out = nn.Linear(hid_size * 2, out_int)
        
    def forward(self, utterance, seq_lengths):
        utt_emb = self.embedding(utterance)
        
        packed_input = pack_padded_sequence(
            utt_emb, seq_lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        
        packed_output, (last_hidden, _) = self.utt_encoder(packed_input)
        utt_encoded, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Extract and concatenate the final forward and backward hidden states
        h_fwd = last_hidden[-2]
        h_bwd = last_hidden[-1]
        last_hidden = torch.cat((h_fwd, h_bwd), dim=1)
        
        slots = self.slot_out(utt_encoded)
        intent = self.intent_out(last_hidden)
        
        slots = slots.permute(0, 2, 1)
        return slots, intent
    
    
class ModelIAS_Bi_Drop(nn.Module):
    def __init__(self, emb_size, hid_size, vocab_len, out_slot, out_int, n_layer=1, pad_index=0, emb_dropout=0.1, out_dropout=0.2):
        super(ModelIAS_Bi_Drop, self).__init__()
        
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.out_dropout = nn.Dropout(out_dropout)
        
        self.utt_encoder = nn.LSTM(
            emb_size,
            hid_size,
            n_layer,
            bidirectional=True,
            batch_first=True,
        )
        
        self.slot_out = nn.Linear(hid_size * 2, out_slot)
        self.intent_out = nn.Linear(hid_size * 2, out_int)
        
    def forward(self, utterance, seq_lengths):
        utt_emb = self.embedding(utterance)
        utt_emb = self.emb_dropout(utt_emb) 
        
        packed_input = pack_padded_sequence(
            utt_emb, seq_lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        
        packed_output, (last_hidden, _) = self.utt_encoder(packed_input)
        utt_encoded, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        h_fwd = last_hidden[-2]
        h_bwd = last_hidden[-1]
        last_hidden = torch.cat((h_fwd, h_bwd), dim=1)
        
        # Dropout on the features right before classification
        utt_encoded = self.out_dropout(utt_encoded)
        last_hidden = self.out_dropout(last_hidden)
        
        slots = self.slot_out(utt_encoded)
        intent = self.intent_out(last_hidden)
        
        slots = slots.permute(0, 2, 1)
        return slots, intent