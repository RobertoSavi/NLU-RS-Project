# -\-\-\ Run the training of the model and save the results /-/-/-
# -------------------- Import libraries --------------------
import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import logging
# -------------------- Import functions from other files --------------------
from functions import *
from utils import *
from model import *

# -------------------- Training process --------------------

# Multi training parameters
best_ppls = []
best_ppl_overall = math.inf
best_model_overall = None

# For each model and optimizer
for model, optimizer, hyperparams in zip(models, optimizers, hyperparams_to_try):
    # Single training parameters
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1, n_epochs))

    learning_rate = hyperparams['lr']
    hidden_size = hyperparams['hid_size']
    embedding_size = hyperparams['emb_size']
    optimizer_name = type(optimizer).__name__
    model_params = f"[Optimizer: {optimizer_name}, Hidden-size: {hidden_size}, Embedding-size: {embedding_size}, Learning-rate: {learning_rate}]\n"

    # Create a file on which to store the results
    filename = f"opt-{optimizer_name}_hid-{hidden_size}_emb-{embedding_size}_lr-{learning_rate:.0e}.txt"
    # Create 'results' folder if it doesn't exist
    os.makedirs("results", exist_ok=True)
    # Full path to the file
    path = os.path.join("results", filename)
    # Create the file (empty or write something if you want)
    with open(path, 'w') as f:
        f.write(model_params)
    print(model_params)

    # For each epoch in each model and optimizer
    for epoch in pbar:

        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)    
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())

            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            losses_dev.append(np.asarray(loss_dev).mean())
            pbar.set_description("PPL: %f" % ppl_dev) 

            if ppl_dev < best_ppl:  # The lower, the better
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = patience  # Reset patience if we get a new best model
            else:
                patience -= 1

            # Create and write the log line
            log = f"[Epoch: {epoch}, PPL: {ppl_dev:.4f}, Best PPL: {best_ppl:.4f}]\n"
            with open(path, 'a') as f:
                f.write(log)
                
            if patience <= 0:  # Early stopping with patience
                break  # Clean exit when training stops

    # -------------------- Final evaluation --------------------
    best_model.to(DEVICE)
    final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
    print('Test PPL:', final_ppl)
    final_log = f"[Final PPL: {final_ppl:.4f}]\n"
    with open(path, 'a') as f:
        f.write(log)

    # Store the best ppl of this configuration
    best_ppls.append(final_ppl)
    
    # Track the best overall model
    if final_ppl < best_ppl_overall:
        best_ppl_overall = final_ppl
        best_model_overall = copy.deepcopy(best_model)

    # -------------------- Model saving --------------------
    # Create 'models' folder if it doesn't exist
    os.makedirs("models", exist_ok=True)
    # Full path to the file
    path = os.path.join("models", filename)
    torch.save(best_model.state_dict(), path)

# -------------------- Save best PPL results --------------------
with open('results/overall_training_results.txt', 'w') as f:
    for i, (ppl, model, optimizer, hyperparams) in enumerate(zip(best_ppls, models, optimizers, hyperparams_to_try)):
        entry = f"Model {i}: [Best PPL: {ppl:.4f}, Hidden-size: {hyperparams['hid_size']}, Embedding-size: {hyperparams['emb_size']}, Optimizer: {type(optimizer).__name__}, Model: {model}]\n"
        f.write(entry)

# To load the model:
# model = LM_RNN(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
# model.load_state_dict(torch.load(path))