# -\-\-\ Define the hyperparameters for the model /-/-/-
# -------------------- Import libraries --------------------
import torch
import torch.optim as optim
from conll import evaluate
from sklearn.metrics import classification_report
import itertools
# -------------------- Import functions from other files --------------------
from models import *
from utils import *

# -------------------- Define hyperparameters for the model --------------------
n_epochs = 65
lr = 5e-5
batch_size = 128
dropout = 0.3
patience_value = 3
runs = 5

hyperparams_to_try = [
    {"lr": lr, "hid_size": 768, "emb_size": 768} # Hidden size 768 for BERT-base
]

out_slot = len(lang.slot2id)
out_int = len(lang.intent2id)

models = []
optimizers = []

# For each combination of hyperparameters, create model and optimizer
for hyperparams in hyperparams_to_try:
    model = JointBERT(out_slot, out_int, dropout).to(DEVICE)
    model.apply(init_weights)
    models.append(model)
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['lr'])
    optimizers.append(optimizer)
    
criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token

# -------------------- Train loop and evaluation loop --------------------
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
