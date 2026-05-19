# Training, evaluation, and pipeline utilities
import os
import json
import copy
import math
import gc
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
from omegaconf import OmegaConf
import optuna

# Import model architectures and device configuration
from models import LM_RNN, LM_LSTM, LM_LSTM_DROPOUT
from utils import DEVICE

# Initialize recurrent and linear layer weights
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

# Build model architecture and optimizer from configuration
def build_model_and_optim(config, vocab_len, pad_index) -> Tuple[nn.Module, optim.Optimizer]:
    if config.part == "1a0":
        model = LM_RNN(
            config.emb_size,
            config.hid_size,
            vocab_len,
            pad_index=pad_index
        )
    elif config.part == "1a1":
        model = LM_LSTM(
            config.emb_size,
            config.hid_size,
            vocab_len,
            pad_index=pad_index
        )
    elif config.part in ["1a2", "1a3"]:
        model = LM_LSTM_DROPOUT(
            config.emb_size,
            config.hid_size,
            vocab_len,
            pad_index=pad_index,
            emb_dropout=config.emb_dropout,
            out_dropout=config.out_dropout
        )
    else:
        raise ValueError(f"Unknown part {config.part}")

    model = model.to(DEVICE)
    model.apply(init_weights)

    if config.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=config.lr)
    elif config.optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    else:
        raise ValueError(f"Unknown optimizer {config.optimizer}")

    return model, optimizer

# Execute one full training epoch
def train_loop(data, optimizer, criterion, model, clip=5) -> float:
    model.train()
    loss_array = []
    number_of_tokens = []
    
    for sample in data:
        optimizer.zero_grad()
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
    return sum(loss_array) / sum(number_of_tokens)

# Evaluate model performance on validation or test data
def eval_loop(data, eval_criterion, model) -> Tuple[float, float]:
    model.eval()
    loss_array = []
    number_of_tokens = []

    with torch.no_grad():
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])
    
    loss_to_return = sum(loss_array) / sum(number_of_tokens)

    ppl = math.exp(loss_to_return)

    return ppl, loss_to_return

# Train model with validation monitoring and early stopping
def train_model(config, model, optimizer, train_loader, dev_loader, pad_index) -> Tuple[nn.Module, list, list]:
    criterion_train = nn.CrossEntropyLoss(ignore_index=pad_index)
    criterion_eval = nn.CrossEntropyLoss(ignore_index=pad_index, reduction='sum')
    
    losses_train = []
    losses_dev = []
    best_ppl = float('inf')
    best_model = None
    patience = config.patience_value
    
    pbar = tqdm(range(1, config.n_epochs + 1))
    for epoch in pbar:
        loss_t = train_loop(train_loader, optimizer, criterion_train, model, config.clip)
        losses_train.append(loss_t)
        ppl_dev, loss_d = eval_loop(dev_loader, criterion_eval, model)
        losses_dev.append(loss_d)
        pbar.set_description(f"Dev PPL: {ppl_dev:.4f}")
        
        if ppl_dev < best_ppl:
            best_ppl = ppl_dev
            best_model = copy.deepcopy(model).cpu()
            patience = config.patience_value
        else:
            patience -= 1
        if patience <= 0:
            print("Early stopping triggered.")
            break
            
    return best_model.to(DEVICE), losses_train, losses_dev

# Evaluate final model perplexity on the test set
def eval_model(model, test_loader, pad_index, model_path=None) -> float:
    if model_path:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Loaded saved model weights from {model_path}")

    criterion_eval = nn.CrossEntropyLoss(ignore_index=pad_index, reduction='sum')
    final_ppl, _ = eval_loop(test_loader, criterion_eval, model)
    print(f"[Final Test PPL: {final_ppl:.4f}]")
    return final_ppl

# Save trained model weights to disk
def save_model(model, save_path) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# Load model weights from disk for evaluation
def load_model(model, model_path) -> nn.Module:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print(f"Model loaded from {model_path}")
    return model
    
# Save training and validation losses to a JSON file for later analysis
def save_losses(losses_train, losses_dev, save_path) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump({'losses_train': losses_train, 'losses_dev': losses_dev}, f)
    print(f"Losses saved to {save_path}")

# Generates and saves training plots when testing, plot a saved training plot when evaluating
def plot_losses(losses_train=None, losses_dev=None, save_path=None, testing=True) -> None:
    if not testing:
        if not save_path or not os.path.exists(save_path):
            print(f"Saved plot not found at: {save_path}")
            return
            
        print(f"Loading saved plot from {save_path}...")
        img = mpimg.imread(save_path)
        plt.figure(figsize=(12, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        return
    
    if not losses_train or not losses_dev:
        print("No loss data available to plot.")
        return
    
    plt.figure(figsize=(10, 5))
    plt.plot(
        range(1, len(losses_train) + 1),
        losses_train,
        label='Train Loss'
    )
    plt.plot(
        range(1, len(losses_dev) + 1),
        losses_dev,
        label='Dev Loss'
    )
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Loss plot successfully generated and saved to {save_path}")
    plt.close()

# Safely removes models and optimizers from memory and clears CUDA cache
def free_memory(model, best_model, optimizer) -> None:
    if torch.cuda.is_available():
        allocated_before = torch.cuda.memory_allocated() / 1024**2
        reserved_before = torch.cuda.memory_reserved() / 1024**2
        print(f"[Memory before cleanup] Allocated: {allocated_before:.2f} MB, Reserved: {reserved_before:.2f} MB")

    # Release GPU memory
    if best_model is not None:
        del best_model
    if model is not None:
        model.to("cpu")
        del model
    if optimizer is not None:
        del optimizer

    # Force garbage collection and empty CUDA cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated_after = torch.cuda.memory_allocated() / 1024**2
        reserved_after = torch.cuda.memory_reserved() / 1024**2
        print(f"[Memory after cleanup] Allocated: {allocated_after:.2f} MB, Reserved: {reserved_after:.2f} MB")
        print("-" * 50)
        
# Append trial results to a central JSON log file
def update_sweep_log(trial_number, params, ppl, log_path) -> None:
    log_data = []
    # Load existing data if the file already exists
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            try:
                log_data = json.load(f)
            except json.JSONDecodeError:
                pass # If file is empty or corrupted, start fresh              
    # Append the new trial's results
    log_data.append({
        "trial_number": trial_number,
        "parameters": params,
        "eval_ppl": ppl
    })
    # Ensure the parent directory exists before writing the log
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    # Write back to disk
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=4)
           
def run_sweep(config, active_params, train_loader, dev_loader, test_loader, vocab_len, pad_index, current_hydra_dir) -> None:
    print("\n================ RUNNING OPTUNA SWEEP ================")
    # Use active_params to build the folder naming keys
    base_params_dict = OmegaConf.to_container(active_params, resolve=True)
    varying_keys = [k for k, v in base_params_dict.items() if isinstance(v, (list, tuple)) and len(v) > 1]
    print(f"Varying parameters for folder naming: {varying_keys}")
    
    best_sweep_loss = float('inf') 
    def objective(trial):
        nonlocal best_sweep_loss
        trial_params = {}
        for key, value in active_params.items():
            if isinstance(value, (list, tuple)) or type(value).__name__ == "ListConfig":
                if len(value) > 1:
                    trial_params[key] = trial.suggest_categorical(key, list(value))
                else:
                    trial_params[key] = value[0]
            else:
                trial_params[key] = value
        # Aggressive duplicate check to force new hyperparameter combinations to be explored
        for past_trial in trial.study.trials:
            if past_trial.state == optuna.trial.TrialState.COMPLETE and past_trial.params == trial_params:
                print(f"\n--- Trial {trial.number} ---")
                print(f"Duplicate hyperparameters found: {trial_params}")
                print("Rejecting this trial to force Optuna to explore new parameters")
                raise optuna.exceptions.TrialPruned()              
                
        trial_config = OmegaConf.merge(config, trial_params)
        
        print(f"\n--- Trial {trial.number} ---")
        print(f"Testing params: {trial_params}")
        folder_name = "_".join([f"{k}={trial_params[k]}" for k in varying_keys]) if varying_keys else f"trial_{trial.number}"
        trial_folder_path = os.path.join(current_hydra_dir, folder_name)
        os.makedirs(trial_folder_path, exist_ok=True)
        
        # Build and train
        model, optimizer = build_model_and_optim(trial_config, vocab_len, pad_index)
        best_model, losses_train, losses_dev = train_model(trial_config, model, optimizer, train_loader, dev_loader, pad_index)
        
        # Evaluate this trial's best model
        trial_ppl = eval_model(best_model, test_loader, pad_index)
        
        # Log and save trial data
        update_sweep_log(trial.number, trial_params, trial_ppl, os.path.join(trial_folder_path, "log.json"))
        save_losses(losses_train, losses_dev, os.path.join(trial_folder_path, "losses.json"))
        plot_losses(losses_train, losses_dev, os.path.join(trial_folder_path, "loss_plot.png"), testing=True)
        
        # Check overall best
        best_val_loss = min(losses_dev)
        if best_val_loss < best_sweep_loss:
            best_sweep_loss = best_val_loss
            best_dir = os.path.join(current_hydra_dir, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            print(f"\nNew best model found! Saving files to {best_dir}...")
            save_model(best_model, os.path.join(best_dir, "model.pt"))
            update_sweep_log(trial.number, trial_params, trial_ppl, os.path.join(best_dir, "log.json"))
            save_losses(losses_train, losses_dev, os.path.join(best_dir, "losses.json"))
            plot_losses(losses_train, losses_dev, os.path.join(best_dir, "loss_plot.png"), testing=True)        
        free_memory(model, best_model, optimizer)
        return best_val_loss
    
    study = optuna.create_study(direction="minimize")
    # Force Optuna to keep trying until it gets 20 unique, successful completions
    target_trials = 20
    print(f"\nSearching for {target_trials} unique hyperparameter combinations...")
    while len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) < target_trials:
        study.optimize(objective, n_trials=1)
        
    print("\n================ SWEEP COMPLETE ================")
    print("Best hyperparameters found:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"Best validation loss: {study.best_value}")
    print(f"Best model and plots have been saved to: {os.path.join(current_hydra_dir, 'best_model')}")

def evaluate_best_model(config, test_loader, vocab_len, pad_index, original_cwd) -> None:
    part_name = (
        f"part={config.name}\n"
        f"hid_size={config.hid_size}\n"
        f"emb_size={config.emb_size}\n"
        f"emb_dropout={config.get('emb_dropout', 'N/A')}\n"
        f"out_dropout={config.get('out_dropout', 'N/A')}\n"
        f"optimizer={config.optimizer}\n"
        f"lr={config.lr}\n"
        f"train_bs={config.train_batch_size}\n"
        f"eval_bs={config.eval_batch_size}"
    )
    print(f"\n================ EVALUATING PART ================\n{part_name}\n=================================================")
    best_dir = os.path.join(original_cwd, "results", f"part{config.name}", "best_model")
    model, _ = build_model_and_optim(config, vocab_len, pad_index)
    
    print("\n--- Loading Saved Model ---")
    model = load_model(model, os.path.join(best_dir, "model.pt"))
    
    print("\n--- Evaluating Best Model ---")
    eval_model(model, test_loader, pad_index)
    
    print("\n--- Displaying Loss Plot ---")
    plot_losses(save_path=os.path.join(best_dir, "loss_plot.png"), testing=False)