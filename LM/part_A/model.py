# -\-\-\ Define the hyperparameters for the model /-/-/-
# -------------------- Import libraries --------------------
import torch
import torch.optim as optim
import torch.nn as nn
import math
import itertools
# -------------------- Import functions from other files --------------------
from functions import *
from utils import *
# Import the number of the exercise part
import builtins
PART = getattr(builtins, 'PART', 3)  # default to part 3 if missing
EVAL = getattr(builtins, 'EVAL', False) # default to false if missing

# -------------------- Define hyperparameters for the model --------------------
n_epochs = 100  # Number of epochs
patience_value = 5    # Early stopping patience
hid_size = 250  # Hidden layer size
emb_size = 350  # Embedding layer size
lr = 1  # Learning rate
clip = 5  # Gradient clipping
vocab_len = len(lang.word2id)  # Vocabulary size
""" lr_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]  # Learning rates to try --- Uncomment these lines to try different configuration of hyperparameters
hid_size_values = [150, 200, 250]  # Hidden layer sizes to try
emb_size_values = [250, 300, 350]  # Embedded layer sizes to try

hyperparams_to_try = [
    {"lr": lr, "hid_size": hid_size, "emb_size": emb_size}
    for lr in lr_values
] 

# Create all combinations of hyperparameters using itertools.product
hyperparams_to_try = [
    {"lr": lr, "hid_size": hid_size, "emb_size": emb_size}
    for lr, hid_size, emb_size in itertools.product(lr_values, hid_size_values, emb_size_values)
] """

# -------------------- Model initialization function --------------------
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

# -------------------- Model and training configuration --------------------
# Initialize model
""" model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
model.apply(init_weights) """

# Optimizer and loss functions
""" optimizer = optim.SGD(model.parameters(), lr=lr) """
criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

# -------------------- Multi training model configuration --------------------
models = []
""" models.append(model) """
optimizers = []
""" optimizers.append(optimizer) """

# Initialize the model based on the PART variable
if PART == 0:
    model = LM_RNN(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"])
    lr = 0.5
    optimizer = optim.SGD(model.parameters(), lr=lr)
elif PART == 1:
    model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"])
    lr = 1.0
    optimizer = optim.SGD(model.parameters(), lr=lr)
elif PART == 2:
    model = LM_LSTM_DROPOUT(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"])
    lr = 1.0
    optimizer = optim.SGD(model.parameters(), lr=lr)
elif PART == 3:
    model = LM_LSTM_DROPOUT(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"])
    lr = 0.0005
    optimizer = optim.AdamW(model.parameters(), lr=lr)

model = model.to(DEVICE)

# Load the pre-trained model if EVAL is True
if EVAL:
    model_paths = {
        0: "models/best_RNN_opt-SGD_hid-250_emb-350_lr-5e-01.pt",
        1: "models/best_LSTM_opt-SGD_hid-250_emb-350_lr-1e+00.pt",
        2: "models/best_LSTM_dropout_opt-SGD_hid-250_emb-350_lr-1e+00.pt",
        3: "models/best_LSTM_dropout_opt-AdamW_hid-250_emb-350_lr-5e-04.pt"
    }
    model.load_state_dict(torch.load(model_paths[PART], map_location=DEVICE))
else:
    model.apply(init_weights)

models.append(model)
optimizers.append(optimizer)


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

# -------------------- Training loop function --------------------
def train_loop(data, optimizer, criterion, model, clip=5):
    model.train()
    loss_array = []
    number_of_tokens = []
    
    for sample in data:
        optimizer.zero_grad()  # Zeroing the gradient
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward()  # Compute the gradient, deleting the computational graph

        # Clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step()  # Update the weights
        
    return sum(loss_array) / sum(number_of_tokens)

# -------------------- Evaluation loop function --------------------
def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_array = []
    number_of_tokens = []

    with torch.no_grad():  # Avoid the creation of computational graph
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])
    
    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return





