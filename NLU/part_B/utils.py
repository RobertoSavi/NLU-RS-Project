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
from transformers import AutoTokenizer

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

# Vocabulary wrapper for token-id mappings (Intents and Slots only for BERT)
class Lang():
    def __init__(self, intents, slots, cutoff=0):
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2slot = {v: k for k, v in self.slot2id.items()}
        self.id2intent = {v: k for k, v in self.intent2id.items()}
    
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
            vocab[elem] = len(vocab)
        return vocab

# Initialize language object with dataset vocabulary
def init_lang(train_raw, dev_raw, test_raw):
    # Labels are gathered from the whole corpus to build the dictionaries
    corpus = train_raw + dev_raw + test_raw 
    slots = set(sum([line['slots'].split() for line in corpus], []))
    intents = set([line['intent'] for line in corpus])
    
    return Lang(intents, slots, cutoff=0)

# Dataset class for Intents and Slots using BERT Tokenizer
class IntentsAndSlots(data.Dataset):
    def __init__(self, dataset, lang, unk='unk', bert_model="bert-base-uncased"):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk
        
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.utt_ids, self.slot_ids = self.mapping_seq(self.utterances, self.slots, lang.slot2id)
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
    
    # Map utterances and slot labels to ids handling BERT sub-tokenization
    def mapping_seq(self, utterance_list, slot_list, mapper):
        utt_ids = []
        slot_ids = []

        # Process each utterance with its slot sequence
        for utt, slots in zip(utterance_list, slot_list):
            words = utt.split()
            slot_labels = slots.split()

            # BERT WordPiece tokenization, keeping track of word <-> subtoken alignment
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
                    aligned_slots.append(mapper.get(label, PAD_TOKEN))
                    slot_ptr += 1
                    prev_word_id = word_id
                else:
                    # Continuation sub-token
                    aligned_slots.append(PAD_TOKEN)

            utt_ids.append(input_ids)
            slot_ids.append(aligned_slots)

        return utt_ids, slot_ids

# Initialize train, validation, and test datasets
def init_datasets(train_raw, dev_raw, test_raw, lang, bert_model="bert-base-uncased"):
    train_dataset = IntentsAndSlots(train_raw, lang, bert_model)
    dev_dataset = IntentsAndSlots(dev_raw, lang, bert_model)
    test_dataset = IntentsAndSlots(test_raw, lang, bert_model)
    return train_dataset, dev_dataset, test_dataset

# Pad sequences and generate BERT attention masks
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
    
    # Generate the attention mask specifically required for BERT
    attention_mask = torch.LongTensor([[1 if id != pad_token else 0 for id in seq] for seq in src_utt])
    
    y_slots, _ = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])
    
    src_utt = src_utt.to(DEVICE)
    y_slots = y_slots.to(DEVICE)
    intent = intent.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)
    
    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["attention_mask"] = attention_mask
    
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
def prepare_data(train_path, test_path, train_batch_size=128, eval_batch_size=64, portion=0.10, bert_model="bert-base-uncased"):
    train_raw, dev_raw, test_raw = load_raw_data(train_path, test_path, portion)
    lang = init_lang(train_raw, dev_raw, test_raw)
    train_dataset, dev_dataset, test_dataset = init_datasets(train_raw, dev_raw, test_raw, lang, bert_model)
    train_loader, dev_loader, test_loader = init_dataloaders(train_dataset, dev_dataset, test_dataset, train_batch_size, eval_batch_size)  
    return lang, train_loader, dev_loader, test_loader