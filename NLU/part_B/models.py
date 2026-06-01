# Model architecture definitions
import torch.nn as nn
from transformers import BertModel

class JointBERT(nn.Module):
    def __init__(self, out_slot, out_int, intent_dropout=0.1, slot_dropout=0.1):
        super(JointBERT, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")  
        
        self.slots_out = nn.Linear(self.bert.config.hidden_size, out_slot)
        self.int_out = nn.Linear(self.bert.config.hidden_size, out_int)

        self.intent_dropout_layer = nn.Dropout(intent_dropout)
        self.slot_dropout_layer = nn.Dropout(slot_dropout)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_out = output.last_hidden_state
        cls_out = output.pooler_output

        sequence_out = self.slot_dropout_layer(sequence_out)
        cls_out = self.intent_dropout_layer(cls_out)   
           
        slots = self.slots_out(sequence_out)
        slots = slots.permute(0, 2, 1) 
        
        intent = self.int_out(cls_out) 

        return slots, intent