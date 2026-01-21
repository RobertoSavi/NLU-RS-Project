# -\-\-\ Run the training of the model and save the results /-/-/-
# -------------------- Import libraries --------------------
import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import logging
# -------------------- Import functions from other files --------------------
from models import *
from utils import *
from params import *

# -------------------- Training process --------------------

# Multi training parameters
best_ppls = []
best_ppl_overall = math.inf
best_model_overall = None
best_model_filename = None

print(hyperparams_to_try)

# For each model and optimizer
for model, optimizer, hyperparams in zip(models, optimizers, hyperparams_to_try):
    # Single training parameters
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1, n_epochs+1))
    patience = patience_value  # Reset patience for each model
    
    # NT-AvSGD parameter initialization
    logs = []       # List to store validation perplexity 'v'
    t = 0          # Logging counter
    T = 0          # Trigger iteration/epoch index
    n = 5          # Non-monotone interval (Paper suggestion: n=5)

    learning_rate = hyperparams['lr']
    hidden_size = hyperparams['hid_size']
    embedding_size = hyperparams['emb_size']
    optimizer_name = type(optimizer).__name__
    model_name = type(model).__name__
    model_params = f"[Model: {model_name}, Optimizer: {optimizer_name}, Hidden-size: {hidden_size}, Embedding-size: {embedding_size}, Learning-rate: {learning_rate}]"

    # Create a file on which to store the results
    filename = f"NT-AvSGD-mod-{model_name}_opt-{optimizer_name}_hid-{hidden_size}_emb-{embedding_size}_lr-{learning_rate:.1e}"
    # Create 'results' folder if it doesn't exist
    os.makedirs("results", exist_ok=True)
    # Full path to the file
    path = os.path.join("results", filename + ".txt")
    # Create the file (empty or write something if you want)
    with open(path, 'w') as f:
        f.write(model_params + '\n')
    print(model_params)

    # For each epoch in each model and optimizer
    for epoch in pbar:

        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)    
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())

            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            losses_dev.append(np.asarray(loss_dev).mean())
            
            # NT-AvSGD logic
            # If the trigger is not yet active, check the non-monotonic condition
            if T == 0:
                # Condition: Current PPL is worse than the minimum PPL recorded n steps ago
                if t > n and ppl_dev > min(logs[:t - n]):
                    T = epoch
                    print(f"Trigger activated at epoch {epoch}. Switching to ASGD.")
                    # Note: ASGD in PyTorch implements the averaging logic described in the paper [3, 7]
                    # We switch to ASGD with t0=0 to start averaging from this point forward
                    optimizer = torch.optim.ASGD(model.parameters(), lr=learning_rate, t0=0, lambd=0.)
                
                logs.append(ppl_dev)
                t += 1
            # ----------------------------------------

            
            pbar.set_description("PPL: %f" % ppl_dev) 

            if ppl_dev < best_ppl:  # The lower, the better
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = patience_value  # Reset patience if we get a new best model
            else:
                patience -= 1

            # Create and write the log line
            log = f"[Epoch: {epoch}, PPL: {ppl_dev:.4f}, Best PPL: {best_ppl:.4f}]\n"
            with open(path, 'a') as f:
                f.write(log)
                
            if patience <= 0:  # Early stopping with patience
                break  # Clean exit when training stops
            
    print("Starting Fine-tuning phase...")
    model.load_state_dict(best_model.state_dict()) # Load best weights
    model.to(DEVICE)
    # Set T=0: Start averaging immediately by using ASGD from the start
    optimizer = torch.optim.ASGD(model.parameters(), lr=learning_rate, t0=0, lambd=0.)
    
    # 3. Reset variables for the non-monotonic stopping criterion [1, 3]
    logs_ft = []
    t_ft = 0
    best_ppl_ft = math.inf
    best_model_ft = None

    # Fine-tuning loop: Uses the same non-monotonic criterion to terminate [1]
    for epoch_ft in range(1, n_epochs + 1):
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)
        ppl_dev, _ = eval_loop(dev_loader, criterion_eval, model)
        
        # Termination Logic: "terminate the run using the same non-monotonic criterion" [1]
        # If current PPL is worse than the best PPL from 'n' epochs ago, stop [3]
        if t_ft > n and ppl_dev > min(logs_ft[:t_ft - n]):
            print(f"Fine-tuning terminated by non-monotonic trigger at epoch {epoch_ft}")
            break
        
        logs_ft.append(ppl_dev)
        t_ft += 1
        
        # Keep track of the absolute best weights during fine-tuning
        if ppl_dev < best_ppl_ft:
            best_ppl_ft = ppl_dev
            best_model_ft = copy.deepcopy(model).to('cpu')


            

    # -------------------- Final evaluation --------------------
    best_model_ft.to(DEVICE)
    final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model_ft)
    print('Test PPL:', final_ppl)
    final_log = f"[Final PPL: {final_ppl:.4f}]\n"
    with open(path, 'a') as f:
        f.write(final_log)
    
    print(f"Finished training model with {model_params.strip()} | Final PPL: {final_ppl:.4f}")

    # Store the best ppl of this configuration
    best_ppls.append(final_ppl)
    
    # Track the best overall model
    if final_ppl < best_ppl_overall:
        best_ppl_overall = final_ppl
        best_model_overall = copy.deepcopy(best_model)
        best_model_filename = filename

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
# Create 'models' folder if it doesn't exist
os.makedirs("models", exist_ok=True)
# Full path to the file
path = os.path.join("models", f"best-NT-AvSGD-{best_model_filename}.pt")
torch.save(best_model_overall.state_dict(), path)

# -------------------- Save best PPL results --------------------
with open('results/overall_training_results_LSTM_NT-AvSGD.txt', 'w') as f:
    for i, (ppl, model, optimizer, hyperparams) in enumerate(zip(best_ppls, models, optimizers, hyperparams_to_try)):
        entry = f"Model {i}: [Best PPL: {ppl:.4f}, Optimizer: {type(optimizer).__name__}, Hidden-size: {hyperparams['hid_size']}, Embedding-size: {hyperparams['emb_size']}, Learning-rate: {hyperparams['lr']}, Model: {model}]\n"
        f.write(entry)

# To load the model:
# model = LM_RNN(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
# model.load_state_dict(torch.load(path))