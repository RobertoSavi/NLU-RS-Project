# -\-\-\ Define functions and classes used for data loading and preprocessing /-/-/-
# -------------------- Import libraries --------------------
import os
import json
import random
import numpy as np
from pprint import pprint
from collections import Counter

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# -------------------- Device configuration --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Used to report errors on CUDA side
PAD_TOKEN = 0
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------- Data loading --------------------
def load_data(path):
    '''
        input: path/to/data
        output: json 
    '''
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset

tmp_train_raw = load_data(os.path.join(BASE_DIR, 'dataset', 'ATIS', 'train.json'))
test_raw = load_data(os.path.join(BASE_DIR, 'dataset', 'ATIS', 'test.json'))

# =========================
# Train / Dev split
# =========================
portion = 0.10

intents = [x['intent'] for x in tmp_train_raw] # We stratify on intents
count_y = Counter(intents)

labels = []
inputs = []
mini_train = []

for id_y, y in enumerate(intents):
    if count_y[y] > 1: # If some intents occurs only once, we put them in training
        inputs.append(tmp_train_raw[id_y])
        labels.append(y)
    else:
        mini_train.append(tmp_train_raw[id_y])

# Random Stratify
X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion, 
                                                    random_state=42, 
                                                    shuffle=True,
                                                    stratify=labels)
X_train.extend(mini_train)
train_raw = X_train
dev_raw = X_dev

y_test = [x['intent'] for x in test_raw]

# -------------------- Word 2 ID --------------------
w2id = {'pad':PAD_TOKEN, 'unk': 1}
slot2id = {'pad':PAD_TOKEN}
intent2id = {}

# Map the words only from the train set
# Map slot and intent labels of train, dev and test set. 'unk' is not needed.
for example in train_raw:
    for w in example['utterance'].split():
        if w not in w2id:
            w2id[w] = len(w2id)   
    for slot in example['slots'].split():
        if slot not in slot2id:
            slot2id[slot] = len(slot2id)
    if example['intent'] not in intent2id:
        intent2id[example['intent']] = len(intent2id)
        
for example in dev_raw:
    for slot in example['slots'].split():
        if slot not in slot2id:
            slot2id[slot] = len(slot2id)
    if example['intent'] not in intent2id:
        intent2id[example['intent']] = len(intent2id)
        
for example in test_raw:
    for slot in example['slots'].split():
        if slot not in slot2id:
            slot2id[slot] = len(slot2id)
    if example['intent'] not in intent2id:
        intent2id[example['intent']] = len(intent2id)

sent = 'I wanna a flight from Toronto to Kuala Lumpur'
mapping = [w2id[w] if w in w2id else w2id['unk'] for w in sent.split()]

# -------------------- Language class --------------------
class Lang():
    def __init__(self, words, intents, slots, cutoff=0):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        
    def w2id(self, elements, cutoff=None, unk=True):
        vocab = {'pad': PAD_TOKEN}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab
    
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab

words = sum([x['utterance'].split() for x in train_raw], []) # No set() since we want to compute 
                                                            # the cutoff
corpus = train_raw + dev_raw + test_raw # We do not want unk labels, 
                                        # however this depends on the research purpose
slots = set(sum([line['slots'].split() for line in corpus],[]))
intents = set([line['intent'] for line in corpus])

lang = Lang(words, intents, slots, cutoff=0)

# -------------------- Dataset class --------------------
class IntentsAndSlots (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, unk='unk'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk
        
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent}
        return sample
    
    # Auxiliary methods
    
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    # Map utterances and slot labels to ids handling BERT sub-tokenization
    def mapping_seq(self, utterance_list, slot_list, mapper):
        utt_ids = []
        slot_ids = []

        # Process each utterance with its slot sequence
        for utt, slots in zip(utterance_list, slot_list):
            words = utt.split()
            slot_labels = slots.split()

            # BERT WordPiece tokenization, keeping track of word ↔ subtoken alignment
            encoding = self.tokenizer(
                words,
                is_split_into_words=True,
                truncation=True,
                return_tensors=None
            )

            # Subtoken ids (includes [CLS], [SEP])
            input_ids = encoding["input_ids"]
            word_ids = encoding.word_ids()

            aligned_slots = []
            prev_word_id = None
            slot_ptr = 0

            # Iterate over BERT subtokens
            for word_id in word_ids:
                if word_id is None:
                    aligned_slots.append(PAD_TOKEN)
                elif word_id != prev_word_id:
                    # First sub-token of a word
                    label = slot_labels[slot_ptr]
                    aligned_slots.append(mapper.get(label, mapper[self.unk]))
                    slot_ptr += 1
                    prev_word_id = word_id
                else:
                    # Continuation sub-token
                    aligned_slots.append(PAD_TOKEN)

            utt_ids.append(input_ids)
            slot_ids.append(aligned_slots)

        return utt_ids, slot_ids


# Initialize Dataset Objects
train_dataset = IntentsAndSlots(train_raw, lang)
dev_dataset = IntentsAndSlots(dev_raw, lang)
test_dataset = IntentsAndSlots(test_raw, lang)

# -------------------- Collate function and Dataloader --------------------
def collate_fn(data):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        # print(padded_seqs)
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    # Sort data by seq lengths
    data.sort(key=lambda x: len(x['utterance']), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
        
    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['utterance'], PAD_TOKEN)
    # Attention mask for BERT
    attention_mask = torch.LongTensor([[1 if id != PAD_TOKEN else 0 for id in seq] for seq in src_utt])
    y_slots, _ = merge(new_item["slots"], PAD_TOKEN)
    intent = torch.LongTensor(new_item["intent"])
    
    src_utt = src_utt.to(DEVICE) # We load the Tensor on our selected device
    y_slots = y_slots.to(DEVICE)
    intent = intent.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)
    
    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["attention_mask"] = attention_mask
    return new_item

# Dataloader instantiations
train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)
