# -\-\-\ Define functions and classes used for data loading and preprocessing
# -------------------- Import libraries --------------------
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from functools import partial

# -------------------- Device configuration --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- File reading function --------------------
def read_file(path, eos_token="<eos>"):
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output

# -------------------- Vocabulary handling --------------------
def get_vocab(corpus, special_tokens=[]):
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

# -------------------- Load dataset --------------------
train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")
test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")

# Vocab is computed only on training set 
vocab = get_vocab(train_raw, ["<pad>", "<eos>"])

# -------------------- Language class --------------------
class Lang():
    def __init__(self, corpus, special_tokens=[]):
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v:k for k, v in self.word2id.items()}
    
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

# Initialize Language Object
lang = Lang(train_raw, ["<pad>", "<eos>"])

# -------------------- Dataset class --------------------
class PennTreeBank (data.Dataset):
    def __init__(self, corpus, lang):
        self.source = []
        self.target = []
        
        for sentence in corpus:
            self.source.append(sentence.split()[0:-1]) 
            self.target.append(sentence.split()[1:]) 
        
        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src = torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        return {'source': src, 'target': trg}
    
    def mapping_seq(self, data, lang):
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    print('OOV found!')
                    print('You have to deal with that') 
                    break
            res.append(tmp_seq)
        return res

# Initialize Dataset Objects
train_dataset = PennTreeBank(train_raw, lang)
dev_dataset = PennTreeBank(dev_raw, lang)
test_dataset = PennTreeBank(test_raw, lang)

# -------------------- Collate function --------------------
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
    
    # Sort data by sequence lengths
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

# -------------------- DataLoader initialization --------------------
train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True,
    collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"])
)
dev_loader = DataLoader(
    dev_dataset, batch_size=128,
    collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"])
)
test_loader = DataLoader(
    test_dataset, batch_size=128,
    collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"])
)