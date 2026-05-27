# Data loading and preprocessing utilities
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from functools import partial
import logging

logger = logging.getLogger(__name__)

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Special tokens configuration
PAD_TOKEN = "<pad>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS=[PAD_TOKEN, EOS_TOKEN]


# Read a text file and append EOS token to each sentence
def read_file(path, eos_token):
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output

# Load raw train, validation, and test datasets
def load_raw_data(train_path, dev_path, test_path, eos_token):
    train_raw = read_file(train_path, eos_token)
    dev_raw = read_file(dev_path, eos_token)
    test_raw = read_file(test_path, eos_token)

    return train_raw, dev_raw, test_raw

# Vocabulary wrapper for token-id mappings
class Lang():
    def __init__(self, corpus, special_tokens=[]):
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v:k for k, v in self.word2id.items()}
    
    # Generate vocabulary dictionary from corpus
    def get_vocab(self, corpus, special_tokens=[]):
        output = {}
        i = 0 
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1
        return output

# Initialize language object with vocabulary
def init_lang(train_raw, special_tokens):
    return Lang(train_raw, special_tokens)

# Dataset class for Penn TreeBank language modeling
class PennTreeBank (data.Dataset):
    def __init__(self, corpus, lang):
        self.source = []
        self.target = []
        
        for sentence in corpus:
            self.source.append(sentence.split()[0:-1]) 
            self.target.append(sentence.split()[1:]) 
        
        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    # Return dataset size
    def __len__(self):
        return len(self.source)

    # Retrieve source-target tensor pair
    def __getitem__(self, idx):
        src = torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        return {'source': src, 'target': trg}
    
    # Convert token sequences into token ids
    def mapping_seq(self, data, lang):
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    logger.info('OOV found!')
                    logger.info('You have to deal with that')
                    break
            res.append(tmp_seq)
        return res

# Initialize train, validation, and test datasets
def init_datasets(train_raw, dev_raw, test_raw, lang):
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)
    
    return train_dataset, dev_dataset, test_dataset

# Pad and batch variable-length sequences
def collate_fn(data, pad_token):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)

        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq 
        
        padded_seqs = padded_seqs.detach()
        return padded_seqs, lengths
    
    data.sort(key=lambda x: len(x["source"]), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    source, _ = merge(new_item["source"])
    target, lengths = merge(new_item["target"])
    
    new_item["source"] = source.to(DEVICE)
    new_item["target"] = target.to(DEVICE)
    new_item["number_tokens"] = sum(lengths)
    
    return new_item

# Create dataloaders for training and evaluation
def init_dataloaders(train_dataset, dev_dataset, test_dataset, lang, train_batch_size=64, eval_batch_size=128):
    pad_token = lang.word2id[PAD_TOKEN]
    train_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        collate_fn=partial(collate_fn, pad_token=pad_token)
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=eval_batch_size,
        collate_fn=partial(collate_fn, pad_token=pad_token)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=eval_batch_size,
        collate_fn=partial(collate_fn, pad_token=pad_token)
    )
    
    return train_loader, dev_loader, test_loader

# Initialize the complete data processing pipeline
def init_data_pipeline(train_path, dev_path, test_path, special_tokens=SPECIAL_TOKENS, train_batch_size=64, eval_batch_size=128):
    train_raw, dev_raw, test_raw = load_raw_data(train_path, dev_path, test_path, EOS_TOKEN)
    lang = init_lang(train_raw, special_tokens)
    train_dataset, dev_dataset, test_dataset = init_datasets(train_raw, dev_raw, test_raw, lang)
    train_loader, dev_loader, test_loader = init_dataloaders(train_dataset, dev_dataset, test_dataset, lang, train_batch_size, eval_batch_size)

    return lang, train_loader, dev_loader, test_loader