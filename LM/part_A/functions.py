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
def save_model(model, save_dir) -> None:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "model.pt")
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
    
# Load model weights from disk for evaluation
def load_model(model, model_path) -> nn.Module:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print(f"Model loaded from {model_path}")
    return model
    
# Save training and validation losses to a JSON file for later analysis
def save_losses(losses_train, losses_dev, save_dir) -> None:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "losses.json")
    with open(path, 'w') as f:
        json.dump({'losses_train': losses_train, 'losses_dev': losses_dev}, f)
    print(f"Losses saved to {path}")

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
def free_memory(model, best_model, optimizer):
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
def update_sweep_log(log_path, trial_number, params, ppl):
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
    
    # Write back to disk
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=4)