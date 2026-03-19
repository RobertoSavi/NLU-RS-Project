# -\-\-\ Define the hyperparameters for the model /-/-/-
# -------------------- Import libraries --------------------
import torch
import torch.optim as optim
import torch.nn as nn
import math
import itertools
# -------------------- Import functions from other files --------------------
from models import *
from utils import *
from functions import *

# -------------------- Define hyperparameters for the model --------------------
n_epochs = 100  # Number of epochs
patience_value = 10    # Early stopping patience
hid_size = 250  # Hidden layer size
emb_size = 250  # Embedding layer size
lr = 1  # Learning rate
clip = 5  # Gradient clipping
vocab_len = len(lang.word2id)  # Vocabulary size
lr_values = [ 1, 5, 10, 15, 20]  # Learning rates to try
""" hid_size_values = [250, 350]  # Hidden layer sizes to try
emb_size_values = [250, 350]  # Embedded layer sizes to try """

# Different learning rates to try
hyperparams_to_try = [
    {"lr": lr, "hid_size": hid_size, "emb_size": emb_size}
    for lr in lr_values
] 

# Create all combinations of hyperparameters using itertools.product
""" hyperparams_to_try = [
    {"lr": lr, "hid_size": hid_size, "emb_size": emb_size}
    for lr, hid_size, emb_size in itertools.product(lr_values, hid_size_values, emb_size_values)
    if hid_size == emb_size and hid_size in [250, 350]
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
for hyperparam in hyperparams_to_try:
    model = LM_LSTM_VAR_DROPOUT(hyperparam["emb_size"], hyperparam["hid_size"], vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
    model.apply(init_weights)
    models.append(model)
    optimizer = optim.SGD(model.parameters(), lr=hyperparam["lr"])
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