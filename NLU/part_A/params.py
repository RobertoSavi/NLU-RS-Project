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
n_epochs = 200
runs = 5
patience_value = 3

""" hid_size = 200
emb_size = 300

lr = 0.0001 # learning rate
clip = 5 # Clip the gradient """

hid_size_values = [200]
emb_size_values = [300]
lr_values = [1e-4, 5e-4, 1e-3,]

# Create all combinations of hyperparameters using itertools.product
hyperparams_to_try = [
    {"lr": lr, "hid_size": hid_size, "emb_size": emb_size}
    for lr, hid_size, emb_size in itertools.product(lr_values, hid_size_values, emb_size_values)
]

out_slot = len(lang.slot2id)
out_int = len(lang.intent2id)
vocab_len = len(lang.word2id)

models = []
optimizers = []

# For each combination of hyperparameters, create model and optimizer
for hyperparams in hyperparams_to_try:
    model = ModelIAS_Bi_Drop(hyperparams['hid_size'], out_slot, out_int, 
                     hyperparams['emb_size'], vocab_len, 
                     pad_index=PAD_TOKEN).to(DEVICE)
    model.apply(init_weights)
    models.append(model)
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['lr'])
    optimizers.append(optimizer)
    
criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token