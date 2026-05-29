from ast import List
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
from sklearn.metrics import classification_report
from tqdm import tqdm
from omegaconf import OmegaConf
import optuna
import logging
from pathlib import Path
from conll import evaluate

# Import model architectures and device configuration
from models import ModelIAS, ModelIAS_Bi, ModelIAS_Bi_Drop
from utils import DEVICE, PAD_TOKEN

logger = logging.getLogger(__name__)

# Initialize recurrent and linear layer weights
def init_weights(mat):
    for m in mat.modules():
        if isinstance(m, (nn.GRU, nn.LSTM, nn.RNN)):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.xavier_uniform_(param[idx * mul:(idx + 1) * mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.orthogonal_(param[idx * mul:(idx + 1) * mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.uniform_(m.weight, -0.01, 0.01)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
                
# Build model architecture and optimizer from configuration
def build_model_and_optim(config, vocab_len, out_slot, out_int, pad_index) -> Tuple[nn.Module, optim.Optimizer]:
    if config.part == "2a0":
        model = ModelIAS(
            config.emb_size,
            config.hid_size,
            vocab_len,
            out_slot,
            out_int,
            pad_index
        )
    elif config.part == "2a1":
        model = ModelIAS_Bi(
            config.emb_size,
            config.hid_size,
            vocab_len,
            out_slot,
            out_int,
            pad_index
        )
    elif config.part == "2a2":
        model = ModelIAS_Bi_Drop(
            config.emb_size,
            config.hid_size,
            vocab_len,
            out_slot,
            out_int,
            pad_index=pad_index,
            emb_dropout=config.emb_dropout,
            out_dropout=config.out_dropout
        )
    else:
        raise ValueError(f"Unknown part {config.part}")

    model = model.to(DEVICE)
    model.apply(init_weights)

    if config.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
    else:
        raise ValueError(f"Unknown optimizer {config.optimizer}")

    return model, optimizer
                    
# Execute one full training epoch
def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5) -> List[float]:
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad()
        slots, intent = model(sample['utterances'], sample['slots_len'])
        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        loss = loss_intent + loss_slot
        loss_array.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step()
    return loss_array

# Evaluate model performance on validation or test data
def eval_loop(data, criterion_slots, criterion_intents, model, lang) -> Tuple[Dict, Dict, List[float]]:
    model.eval()
    loss_array = []
    
    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []
    
    with torch.no_grad():
        for sample in data:
            slots, intents = model(sample['utterances'], sample['slots_len'])
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot 
            loss_array.append(loss.item())
            out_intents = [lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()] 
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
    try:            
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}
        
    report_intent = classification_report(ref_intents, hyp_intents, 
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array

# Train model with validation monitoring and early stopping
def train_model(config, model, optimizer, train_loader, dev_loader, lang, pad_index) -> Tuple[nn.Module, list, list]:
    criterion_slots = nn.CrossEntropyLoss(ignore_index=pad_index)
    criterion_intents = nn.CrossEntropyLoss() 
    
    losses_train = []
    losses_dev = []
    best_f1 = float('-inf')
    best_model = None
    patience = config.patience
    
    pbar = tqdm(range(1, config.n_epochs + 1))
    for epoch in pbar:
        loss_t = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, config.clip)
        losses_train.append(loss_t)
        ppl_dev, loss_d = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang)
        losses_dev.append(loss_d)
        pbar.set_description(f"Dev PPL: {ppl_dev:.4f}")
        
        if ppl_dev < best_ppl:
            best_ppl = ppl_dev
            best_model = copy.deepcopy(model).cpu()
            patience = config.patience_value
        else:
            patience -= 1
        if patience <= 0:
            pbar.close()
            logger.info("Early stopping triggered.")
            break
            
    return best_model.to(DEVICE), losses_train, losses_dev