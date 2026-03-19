# -\-\-\ Define the architecture of the model /-/-/-
# -------------------- Import libraries --------------------
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from conll import evaluate
from sklearn.metrics import classification_report

class ModelIAS(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0):
        super(ModelIAS, self).__init__()
        # hid_size = Hidden size
        # out_slot = number of slots (output size for slot filling)
        # out_int = number of intents (output size for intent class)
        # emb_size = word embedding size
        
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        
        self.utt_encoder = nn.LSTM(
            emb_size,
            hid_size,
            n_layer,
            bidirectional=False,
            batch_first=True,
        )
        self.slot_out = nn.Linear(hid_size, out_slot)
        self.intent_out = nn.Linear(hid_size, out_int)
        # Dropout layer How/Where do we apply it?
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, utterance, seq_lengths):
        # utterance.size() = batch_size X seq_len
        utt_emb = self.embedding(utterance) # utt_emb.size() = batch_size X seq_len X emb_size
        
        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost
        
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        # Process the batch
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input) 
       
        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        # Get the last hidden state
        last_hidden = last_hidden[-1,:,:]
        
        # Is this another possible way to get the last hiddent state? (Why?)
        # utt_encoded.permute(1,0,2)[-1]
        
        # Compute slot logits
        slots = self.slot_out(utt_encoded)
        # Compute intent logits
        intent = self.intent_out(last_hidden)
        
        # Slot size: batch_size, seq_len, classes 
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent
    
    
class ModelIAS_Bi(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0):
        super(ModelIAS_Bi, self).__init__()
        
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        
        self.utt_encoder = nn.LSTM(
            emb_size,
            hid_size,
            n_layer,
            bidirectional=True,
            batch_first=True
        )
        
        self.slot_out = nn.Linear(hid_size * 2, out_slot)
        self.intent_out = nn.Linear(hid_size * 2, out_int)
        
    def forward(self, utterance, seq_lengths):
        utt_emb = self.embedding(utterance)
        
        packed_input = pack_padded_sequence(
            utt_emb, seq_lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input)
        utt_encoded, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # last_hidden: (num_layers * 2, batch, hid_size)
        # Take last layer forward and backward and concatenate
        h_fwd = last_hidden[-2]
        h_bwd = last_hidden[-1]
        last_hidden = torch.cat((h_fwd, h_bwd), dim=1)
        
        # Slot logits
        slots = self.slot_out(utt_encoded)
        # Intent logits
        intent = self.intent_out(last_hidden)
        
        # For CrossEntropyLoss on slots
        slots = slots.permute(0, 2, 1)
        return slots, intent
    
    
class ModelIAS_Bi_Drop(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0, dropout=0.1):
        super(ModelIAS_Bi_Drop, self).__init__()
        
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        self.dropout = nn.Dropout(dropout)
        
        self.utt_encoder = nn.LSTM(
            emb_size,
            hid_size,
            n_layer,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if n_layer > 1 else 0
        )
        
        self.slot_out = nn.Linear(hid_size * 2, out_slot)
        self.intent_out = nn.Linear(hid_size * 2, out_int)
        
    def forward(self, utterance, seq_lengths):
        utt_emb = self.embedding(utterance)
        utt_emb = self.dropout(utt_emb)
        
        packed_input = pack_padded_sequence(
            utt_emb, seq_lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input)
        utt_encoded, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # last_hidden: (num_layers * 2, batch, hid_size)
        # Take last layer forward and backward and concatenate
        h_fwd = last_hidden[-2]
        h_bwd = last_hidden[-1]
        last_hidden = torch.cat((h_fwd, h_bwd), dim=1)
        
        utt_encoded = self.dropout(utt_encoded)
        
        # Slot logits
        slots = self.slot_out(utt_encoded)
        # Intent logits
        intent = self.intent_out(last_hidden)
        
        # For CrossEntropyLoss on slots
        slots = slots.permute(0, 2, 1)
        return slots, intent