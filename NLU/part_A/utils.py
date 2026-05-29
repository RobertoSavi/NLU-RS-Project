# Data loading and preprocessing utilities
import os
import json
import logging
from collections import Counter
from functools import partial

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Device and environment configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
PAD_TOKEN = 0

# Read and load JSON data
def load_data(path):
    with open(path) as f:
        return json.loads(f.read())

# Load raw train, validation, and test datasets with stratification
def load_raw_data(train_path, test_path, portion=0.10):
    tmp_train_raw = load_data(train_path)
    test_raw = load_data(test_path)

    intents = [x['intent'] for x in tmp_train_raw]
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    # Stratify on intents, ensuring single-occurrence intents stay in training
    for id_y, y in enumerate(intents):
        if count_y[y] > 1:
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])

    X_train, X_dev, y_train, y_dev = train_test_split(
        inputs, labels, test_size=portion, 
        random_state=42, shuffle=True, stratify=labels
    )
    X_train.extend(mini_train)

    return X_train, X_dev, test_raw

# Vocabulary wrapper for token-id mappings (Words, Intents, Slots)
class Lang():
    def __init__(self, words, intents, slots, cutoff=0):
        self.word2id = self.w2id(words, cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.id2slot = {v: k for k, v in self.slot2id.items()}
        self.id2intent = {v: k for k, v in self.intent2id.items()}
        
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

# Initialize language object with dataset vocabulary
def init_lang(train_raw, dev_raw, test_raw):
    # Compute cutoff on train set only, but gather labels from entire corpus
    words = sum([x['utterance'].split() for x in train_raw], [])
    corpus = train_raw + dev_raw + test_raw
    slots = set(sum([line['slots'].split() for line in corpus], []))
    intents = set([line['intent'] for line in corpus])
    
    return Lang(words, intents, slots, cutoff=0)

# Dataset class for Intents and Slots
class IntentsAndSlots(data.Dataset):
    def __init__(self, dataset, lang, unk='unk'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk
        
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
        return {'utterance': utt, 'slots': slots, 'intent': intent}
    
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    def mapping_seq(self, data, mapper):
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res

# Initialize train, validation, and test datasets
def init_datasets(train_raw, dev_raw, test_raw, lang):
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)
    
    return train_dataset, dev_dataset, test_dataset

# Pad and batch variable-length sequences
def collate_fn(data, pad_token=PAD_TOKEN):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)
        
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq
            
        return padded_seqs.detach(), lengths
    
    data.sort(key=lambda x: len(x['utterance']), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
        
    src_utt, _ = merge(new_item['utterance'])
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])
    
    src_utt = src_utt.to(DEVICE)
    y_slots = y_slots.to(DEVICE)
    intent = intent.to(DEVICE)
    y_lengths = torch.LongTensor(y_lengths).to(DEVICE)
    
    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    
    return new_item

# Create dataloaders for training and evaluation
def init_dataloaders(train_dataset, dev_dataset, test_dataset, train_batch_size=128, eval_batch_size=64):
    train_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, 
        collate_fn=partial(collate_fn, pad_token=PAD_TOKEN)
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=eval_batch_size, 
        collate_fn=partial(collate_fn, pad_token=PAD_TOKEN)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=eval_batch_size, 
        collate_fn=partial(collate_fn, pad_token=PAD_TOKEN)
    )
    
    return train_loader, dev_loader, test_loader

# Initialize the complete data processing pipeline
def init_data_pipeline(train_path, test_path, train_batch_size=128, eval_batch_size=64, portion=0.10):
    train_raw, dev_raw, test_raw = load_raw_data(train_path, test_path, portion)
    lang = init_lang(train_raw, dev_raw, test_raw)
    vocab_len = len(lang.word2id)
    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    train_dataset, dev_dataset, test_dataset = init_datasets(train_raw, dev_raw, test_raw, lang)
    train_loader, dev_loader, test_loader = init_dataloaders(train_dataset, dev_dataset, test_dataset, lang, train_batch_size, eval_batch_size)
    return train_loader, dev_loader, test_loader, vocab_len, out_slot, out_int