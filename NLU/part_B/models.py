# -\-\-\ Define the architecture of the model /-/-/-
# -------------------- Import libraries --------------------
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from conll import evaluate
from sklearn.metrics import classification_report
from transformers import BertModel, BertPreTrainedModel


class JointBERT(nn.Module):
    
    def __init__(self, out_slot, out_int, dropout):
        super(JointBERT, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")  
        
        self.slots_out = nn.Linear(self.bert.config.hidden_size, out_slot)
        self.int_out = nn.Linear(self.bert.config.hidden_size, out_int)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_out = output.last_hidden_state
        cls_out = output.pooler_output

        sequence_out = self.dropout(sequence_out)
        cls_out = self.dropout(cls_out)   
           
        slots = self.slots_out(sequence_out)
        # Shape (batch_size, num_slot_labels, seq_len)
        slots = slots.permute(0, 2, 1) 
        # Shape (batch_size, num_intent_labels)
        intent = self.int_out(cls_out) 

        return slots, intent
