# -\-\-\ Run the training of the model and save the results /-/-/-
# -------------------- CLI Argument Parsing --------------------
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--part", type=int, choices=[0, 1, 2, 3], required=True, help="Which part of the assignment to run")
parser.add_argument("--eval", action="store_true", help="Evaluate the best saved model for this part")
args = parser.parse_args()

import builtins
builtins.PART = args.part
builtins.EVAL = args.eval
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
best_model_filename = None

# For each model and optimizer
#for model, optimizer, hyperparams in zip(models, optimizers, hyperparams_to_try): --- Uncomment this line to train multiple models with different hyperparameters
for model, optimizer in zip(models, optimizers):

    learning_rate = hyperparams['lr'] if 'hyperparams' in locals() else lr
    hidden_size = hyperparams['hid_size'] if 'hyperparams' in locals() else hid_size
    embedding_size = hyperparams['emb_size'] if 'hyperparams' in locals() else emb_size
    optimizer_name = type(optimizer).__name__
    model_name = type(model).__name__
    model_params = f"[Model: {model_name}{', With dropout' if model_name == 'LM_LSTM_DROPOUT' else ''}, Optimizer: {optimizer_name}, Hidden-size: {hidden_size}, Embedding-size: {embedding_size}, Learning-rate: {learning_rate}]"
    #model_params = f"[Model: {model_name}, With dropout, Optimizer: {optimizer_name}, Hidden-size: {hidden_size}, Embedding-size: {embedding_size}, Learning-rate: {learning_rate}]"
    print(model_params)
    
    if EVAL:
        print("Evaluating pre-trained model...")
        model.to(DEVICE)
        final_ppl, _ = eval_loop(test_loader, criterion_eval, model)
        print(f"[Final PPL: {final_ppl:.4f}]")

    else:
        print("Training model...")
        # Single training parameters
        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_ppl = math.inf
        best_model = None
        pbar = tqdm(range(1, n_epochs))
        patience = patience_value  # Reset patience for each model

        """ # Create a file on which to store the results
        filename = f"mod-{model_name}{'_drop-yes' if model_name == 'LM_LSTM_DROPOUT' else ''}_opt-{optimizer_name}_hid-{hidden_size}_emb-{embedding_size}_lr-{learning_rate:.0e}"
        # Create 'results' folder if it doesn't exist
        os.makedirs("results", exist_ok=True)
        # Full path to the file
        path = os.path.join("results", filename + ".txt")
        # Create the file (empty or write something if you want)
        with open(path, 'w') as f:
            f.write(model_params + '\n') """

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
                    patience = patience_value  # Reset patience if we get a new best model
                else:
                    patience -= 1

                """ # Create and write the log line
                log = f"[Epoch: {epoch}, PPL: {ppl_dev:.4f}, Best PPL: {best_ppl:.4f}]\n"
                with open(path, 'a') as f:
                    f.write(log) """
                    
                if patience <= 0:  # Early stopping with patience
                    break  # Clean exit when training stops

        # -------------------- Final evaluation --------------------
        best_model.to(DEVICE)
        final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
        print(f"[Final PPL: {final_ppl:.4f}]")
        """ final_log = f"[Final PPL: {final_ppl:.4f}]\n"
        with open(path, 'a') as f:
            f.write(final_log) """
        
        print(f"Finished training model with {model_params.strip()} | Final PPL: {final_ppl:.4f}")

        # Store the best ppl of this configuration
        best_ppls.append(final_ppl)
        
        # Track the best overall model
        if final_ppl < best_ppl_overall:
            best_ppl_overall = final_ppl
            best_model_overall = copy.deepcopy(best_model)
            #best_model_filename = filename

        # Log GPU memory before cleanup
        allocated_before = torch.cuda.memory_allocated() / 1024**2
        reserved_before = torch.cuda.memory_reserved() / 1024**2
        print(f"[Memory before cleanup] Allocated: {allocated_before:.2f} MB, Reserved: {reserved_before:.2f} MB")

        # Release GPU memory right after evaluation
        del best_model  
        model.to("cpu")
        del model
        del optimizer
        torch.cuda.empty_cache()

        # Log GPU memory after cleanup
        allocated_after = torch.cuda.memory_allocated() / 1024**2
        reserved_after = torch.cuda.memory_reserved() / 1024**2
        print(f"[Memory after cleanup] Allocated: {allocated_after:.2f} MB, Reserved: {reserved_after:.2f} MB")
        print("\n ----------------------------------- \n")

# -------------------- Model saving --------------------
""" # Create 'models' folder if it doesn't exist
os.makedirs("models", exist_ok=True)
# Full path to the file
path = os.path.join("models", f"best_LSTM_dropout_adamw_{best_model_filename}.pt")
torch.save(best_model_overall.state_dict(), path) """

# -------------------- Save best PPL results --------------------
""" with open('results/overall_training_results_LSTM_dropout_adamw2.txt', 'w') as f:
    for i, (ppl, model, optimizer, hyperparams) in enumerate(zip(best_ppls, models, optimizers, hyperparams_to_try)):
        entry = f"Model {i}: [Best PPL: {ppl:.4f}, Optimizer: {type(optimizer).__name__}, Hidden-size: {hyperparams['hid_size']}, Embedding-size: {hyperparams['emb_size']}, Learning-rate: {hyperparams['lr']}, Model: {model}]\n"
        f.write(entry) """