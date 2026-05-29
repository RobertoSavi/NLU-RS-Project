# Training, evaluation, and pipeline utilities
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from conll import evaluate
from sklearn.metrics import classification_report
from utils import PAD_TOKEN

# Initialize linear layer weights
def init_weights(mat):
    for m in mat.modules():
        if isinstance(m, (nn.Linear)):
            torch.nn.init.uniform_(m.weight, -0.01, 0.01)
            if m.bias != None:
                m.bias.data.fill_(0.01)

# -------------------- Train loop --------------------
def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        slots, intent = model(sample['utterances'], sample['attention_mask'])
        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        loss = loss_intent + loss_slot # In joint training we sum the losses. 
                                       # Is there another way to do that?
        loss_array.append(loss.item())
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
    return loss_array

# -------------------- Evaluation loop --------------------
def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            slots, intents = model(sample['utterances'], sample['attention_mask'])
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot 
            loss_array.append(loss.item())
            # Intent inference
            # Get the highest probable class
            out_intents = [lang.id2intent[x] 
                           for x in torch.argmax(intents, dim=1).tolist()] 
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            # Slot inference 
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                utt_ids = sample['utterance'][id_seq].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                
                utterance = tokenizer.convert_ids_to_tokens(utt_ids)
                
                tmp_ref = []
                tmp_hyp = []
                for i, gt_id in enumerate(gt_ids):
                    # Remove subword tokens for evaluation
                    if gt_id != PAD_TOKEN:
                        word = utterance[i]
                        ref_slot = lang.id2slot[gt_id]
                        hyp_slot = lang.id2slot[seq[i].item()]
                        
                        # Append tuple (word, label) to temp lists
                        tmp_ref.append((word, ref_slot))
                        tmp_hyp.append((word, hyp_slot))
                # Append the filtered sequences to the main lists
                ref_slots.append(tmp_ref)
                hyp_slots.append(tmp_hyp)
    try:            
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([label for sent in ref_slots for word, label in sent])
        hyp_s = set([label for sent in hyp_slots for word, label in sent])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}
        
    report_intent = classification_report(ref_intents, hyp_intents, 
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array