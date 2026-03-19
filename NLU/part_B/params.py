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
from functions import *

# -------------------- Define hyperparameters for the model --------------------
n_epochs = 65
lr = 1e-4
batch_size = 128
dropout = 0.1
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
    model.slots_out.apply(init_weights)
    model.int_out.apply(init_weights)
    models.append(model)
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['lr'])
    optimizers.append(optimizer)
    
criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token
