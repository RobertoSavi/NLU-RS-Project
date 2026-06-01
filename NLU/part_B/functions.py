# Training, evaluation, and pipeline utilities
import copy
import gc
import json
import os
import logging
from pathlib import Path
from typing import Dict, Tuple, List
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from tqdm import tqdm
from omegaconf import OmegaConf
import optuna
from conll import evaluate

# Import model architectures and device configuration
from models import ModelIAS, ModelIAS_Bi, ModelIAS_Bi_Drop
from utils import DEVICE

logger = logging.getLogger(__name__)

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
def build_model_and_optim(config, vocab_len, out_slot, out_int, pad_index) -> Tuple[nn.Module, optim.Optimizer]:
    if config.part == "2a0":
        model = ModelIAS(
            config.emb_size,
            config.hid_size,
            vocab_len,
            out_slot,
            out_int,
            pad_index
        )
    elif config.part == "2a1":
        model = ModelIAS_Bi(
            config.emb_size,
            config.hid_size,
            vocab_len,
            out_slot,
            out_int,
            pad_index
        )
    elif config.part == "2a2":
        model = ModelIAS_Bi_Drop(
            config.emb_size,
            config.hid_size,
            vocab_len,
            out_slot,
            out_int,
            pad_index,
            emb_dropout=config.emb_dropout,
            out_dropout=config.out_dropout
        )
    else:
        raise ValueError(f"Unknown part {config.part}")

    model = model.to(DEVICE)
    model.apply(init_weights)

    if config.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
    else:
        raise ValueError(f"Unknown optimizer {config.optimizer}")

    return model, optimizer
                    
# Execute one full training epoch
def train_loop(data, optimizer, pad_index, model, clip=5) -> List[float]:
    model.train()
    criterion_slots = nn.CrossEntropyLoss(ignore_index=pad_index)
    criterion_intents = nn.CrossEntropyLoss()
    loss_array = []
    for sample in data:
        optimizer.zero_grad()
        slots, intent = model(sample['utterances'], sample['slots_len'])
        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        loss = loss_intent + loss_slot
        loss_array.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step()
    return loss_array

# Evaluate model performance on validation or test data
def eval_loop(data, pad_index, model, lang) -> Tuple[Dict, Dict, List[float]]:
    model.eval()
    criterion_slots = nn.CrossEntropyLoss(ignore_index=pad_index)
    criterion_intents = nn.CrossEntropyLoss()
    loss_array = []
    
    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []
    
    with torch.no_grad():
        for sample in data:
            slots, intents = model(sample['utterances'], sample['slots_len'])
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot 
            loss_array.append(loss.item())
            out_intents = [lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()] 
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
    try:            
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}
        
    report_intent = classification_report(ref_intents, hyp_intents, 
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array

# Main training loop that executes multiple runs and computes average performance
def train_model(config, train_loader, dev_loader, test_loader, lang, vocab_len, out_slot, out_int, pad_index) -> Tuple[nn.Module, float, List[float], List[float], List[List[float]], List[List[float]], List[List[int]]]:
    slot_f1s, intent_accs, best_models = [], [], []
    dev_f1s = []
    
    all_losses_train = []
    all_losses_dev = []
    all_sampled_epochs = []

    # Outer Loop: Execute N independent runs
    for run in tqdm(range(0, config.runs), desc="Runs"):
        model, optimizer = build_model_and_optim(config, vocab_len, out_slot, out_int, pad_index)
        patience = config.patience_value
        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_f1 = float('-inf')
        best_model = None

        # Inner Loop: Train the model for N epochs
        for epoch in tqdm(range(1, config.n_epochs + 1), desc=f"Run {run+1} Training"):
            loss_t_array = train_loop(train_loader, optimizer, pad_index, model, config.clip)
            
            # Evaluate every 5 epochs
            if epoch % 5 == 0: 
                sampled_epochs.append(epoch)
                losses_train.append(np.asarray(loss_t_array).mean())
                
                results_dev, _, loss_d_array = eval_loop(dev_loader, pad_index, model, lang)
                losses_dev.append(np.asarray(loss_d_array).mean())
                
                f1 = results_dev['total']['f']
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = copy.deepcopy(model).cpu()
                    patience = config.patience_value
                else:
                    patience -= 1
                    
                if patience <= 0:
                    break 

        if best_model is None:
            best_model = copy.deepcopy(model).cpu()
            
        # Evaluate the best model of this run on the Test set
        best_model.to(DEVICE)
        results_test, intent_test, _ = eval_loop(test_loader, pad_index, best_model, lang)   
        
        # Store the results for this run
        intent_accs.append(intent_test['accuracy'])
        slot_f1s.append(results_test['total']['f'])
        
        dev_f1s.append(best_f1) 
        
        best_models.append((best_model.cpu(), best_f1))
        
        all_losses_dev.append(losses_dev)
        all_losses_train.append(losses_train)
        all_sampled_epochs.append(sampled_epochs)

        print(f'\nRun {run+1} | Dev F1: {best_f1:.4f} | Test Slot F1: {results_test["total"]["f"]:.4f} | Test Intent Acc: {intent_test["accuracy"]:.4f}')
        
        free_memory(model, best_model, optimizer)
    
    # Compute mean F1 across all runs for the dev set    
    mean_dev_f1 = round(np.asarray(dev_f1s).mean(), 4)
    # Compute and store mean and standard deviation across all runs
    slot_f1s = np.asarray(slot_f1s)
    intent_accs = np.asarray(intent_accs)
    mean_slot_f1 = round(slot_f1s.mean(), 4)
    mean_int_acc = round(intent_accs.mean(), 4)
    
    # Find the absolute best model from all the runs
    best_overall_model, _ = max(best_models, key=lambda x: x[1])
    final_best_model = copy.deepcopy(best_overall_model).cpu()

    print(f'Dev F1:     {mean_dev_f1:.4f}')
    print(f'Test F1:    {mean_slot_f1:.4f} ± {round(slot_f1s.std(), 4)}')
    print(f'Intent Acc: {mean_int_acc:.4f} ± {round(intent_accs.std(), 4)}')

    return final_best_model, mean_dev_f1, slot_f1s.tolist(), intent_accs.tolist(), all_losses_train, all_losses_dev, all_sampled_epochs

# Save trained model weights to disk
def save_model(model, w2id, slot2id, intent2id, save_path) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model_data_save = {
        "model": model.state_dict(), 
        "w2id": w2id, 
        "slot2id": slot2id, 
        "intent2id": intent2id
    }
    torch.save(model_data_save, save_path)
    logger.info(f"Model saved to {save_path}")

# Load model weights and vocabularies from disk for evaluation
def load_model(model, model_path) -> Tuple[nn.Module, Dict, Dict, Dict]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at: {model_path}")
        
    state = torch.load(model_path, map_location=DEVICE)
    
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
        w2id = state.get("w2id", {})
        slot2id = state.get("slot2id", {})
        intent2id = state.get("intent2id", {})
    else:
        model.load_state_dict(state)
        w2id, slot2id, intent2id = {}, {}, {}
        
    logger.info(f"Model and vocabularies loaded from {model_path}")
    return model, w2id, slot2id, intent2id


# Save parameters, training and validation losses to a JSON file for later analysis
def save_losses(trial_number, part_name, model_name, params, dev_f1, test_f1_mean, test_f1_std, test_acc_mean, test_acc_std, all_losses_train, all_losses_dev, all_sampled_epochs, save_path) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True) 
    data = {
        "trial_number": trial_number,
        "part_name": part_name,
        "model_name": model_name,
        "parameters": params,
        "metrics": {
            "mean_dev_f1": dev_f1,
            "test_slot_f1_mean": test_f1_mean,
            "test_slot_f1_std": test_f1_std,
            "test_intent_acc_mean": test_acc_mean,
            "test_intent_acc_std": test_acc_std
        },
        "all_losses_train": all_losses_train,
        "all_losses_dev": all_losses_dev,
        "all_sampled_epochs": all_sampled_epochs
    }
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info(f"Losses and parameters saved to {save_path}")
    
# Generate and save a plot of training and validation losses over sampled epochs   
def save_loss_plot(all_sampled_epochs, all_losses_train, all_losses_dev, save_path) -> None:
    plt.figure(figsize=(10, 5))
    
    # Plot a line for each run in the sweep trial
    for run_idx, (epochs, t_loss, d_loss) in enumerate(zip(all_sampled_epochs, all_losses_train, all_losses_dev)):
        plt.plot(epochs, t_loss, label=f'Run {run_idx+1} Train Loss', marker='o', alpha=0.8)
        plt.plot(epochs, d_loss, label=f'Run {run_idx+1} Dev Loss', marker='x', linestyle='--', alpha=0.8)
        
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses Across Runs')
    
    # Place the legend slightly outside the plot so it doesn't cover the lines
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    logger.info(f"Loss plot successfully generated and saved to {save_path}")
    plt.close()
    
# Display a saved loss plot for visual analysis
def display_loss_plot(plot_path) -> None:
    if not plot_path or not os.path.exists(plot_path):
        logger.info(f"Saved plot not found at: {plot_path}")
        return
        
    logger.info(f"Loading saved plot from {plot_path}...")
    img = mpimg.imread(plot_path)
    plt.figure(figsize=(12, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    return

# Safely removes models and optimizers from memory and clears CUDA cache
def free_memory(model, best_model, optimizer) -> None:
    if torch.cuda.is_available():
        allocated_before = torch.cuda.memory_allocated() / 1024**2
        reserved_before = torch.cuda.memory_reserved() / 1024**2
        logger.info(f"[Memory before cleanup] Allocated: {allocated_before:.2f} MB, Reserved: {reserved_before:.2f} MB")

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
        logger.info(f"[Memory after cleanup] Allocated: {allocated_after:.2f} MB, Reserved: {reserved_after:.2f} MB")
        logger.info("-" * 50)

# Append trial results to a central JSON log file
def update_sweep_log(trial_number, part_name, model_name, params, dev_f1, test_f1_mean, test_f1_std, test_acc_mean, test_acc_std, log_path) -> None:
    log_data = []
    # Load existing data if the file already exists
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            try:
                log_data = json.load(f)
            except json.JSONDecodeError:
                pass
    # Append the new trial's results
    log_data.append({
        "trial_number": trial_number,
        "part_name": part_name,
        "model_name": model_name,
        "parameters": params,
        "metrics": {
            "mean_dev_f1": dev_f1,
            "test_slot_f1_mean": test_f1_mean,
            "test_slot_f1_std": test_f1_std,
            "test_intent_acc_mean": test_acc_mean,
            "test_intent_acc_std": test_acc_std
        }
    })
    
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=4)


def run_sweep(config, active_params, train_loader, dev_loader, test_loader, lang, vocab_len, out_slot, out_int, pad_index, current_hydra_dir) -> None:
    logger.info("\n================ RUNNING OPTUNA SWEEP ================")
    # Use active_params to build the folder naming keys
    base_params_dict = OmegaConf.to_container(active_params, resolve=True)
    varying_keys = [k for k, v in base_params_dict.items() if isinstance(v, (list, tuple)) and len(v) > 1]
    logger.info(f"Varying parameters for folder naming: {varying_keys}")
    
    # Determine search strategy based on the number of combinations in the grid search space
    grid_search_space = {}
    total_combinations = 1
    for key, value in base_params_dict.items():
        if isinstance(value, (list, tuple)) and len(value) > 1:
            grid_search_space[key] = value
            total_combinations *= len(value)
    # If the total combinations are small, use Grid Search to exhaustively explore. If large, use TPE to intelligently sample 20 unique combinations.        
    if total_combinations <= 20:
        logger.info(f"Small search space detected ({total_combinations} max combinations). Switching to Exhaustive Grid Search.")
        sampler = optuna.samplers.GridSampler(grid_search_space)
        target_trials = total_combinations
    else:
        logger.info(f"Large search space detected ({total_combinations} max combinations). Using Smart TPE Search for 20 trials.")
        sampler = optuna.samplers.TPESampler()
        target_trials = 20

    best_sweep_f1 = float('-inf')
    def objective(trial):
        nonlocal best_sweep_f1
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
        current_suggested_params = trial.params
        for past_trial in trial.study.trials:
            is_complete = past_trial.state == optuna.trial.TrialState.COMPLETE
            is_pruned = past_trial.state == optuna.trial.TrialState.PRUNED
            if (is_complete or is_pruned) and past_trial.params == current_suggested_params:
                logger.info(f"\n--- Trial {trial.number} ---")
                logger.info(f"Duplicate hyperparameters found: {trial_params}")
                logger.info("Rejecting this trial to force Optuna to explore new parameters")
                raise optuna.exceptions.TrialPruned()              
                
        trial_config = OmegaConf.merge(config, trial_params)
        
        logger.info(f"\n--- Trial {trial.number} ---")
        logger.info(f"Testing params for {trial_config.name}: {trial_params}")
        folder_name = "_".join([f"{k}={trial_params[k]}" for k in varying_keys]) if varying_keys else f"trial_{trial.number}"
        trial_folder_path = os.path.join(current_hydra_dir, folder_name)
        os.makedirs(trial_folder_path, exist_ok=True)

        # Build and train, train_model does both
        best_model, mean_dev_f1, slot_f1s, intent_accs, all_losses_train, all_losses_dev, all_sampled_epochs = train_model(
            trial_config, train_loader, dev_loader, test_loader, lang, vocab_len, out_slot, out_int, pad_index
        )
        
        # Save part and model name for logging
        part_name = trial_config.name
        model_name = type(best_model).__name__

        # Calculate metrics for logging
        test_f1_mean = np.mean(slot_f1s)
        test_f1_std = np.std(slot_f1s)
        test_acc_mean = np.mean(intent_accs)
        test_acc_std = np.std(intent_accs)
        model_name = best_model.__class__.__name__

        # Log and save trial data
        save_losses(trial.number, part_name, model_name, trial_params, mean_dev_f1, test_f1_mean, test_f1_std, test_acc_mean, test_acc_std, all_losses_train, all_losses_dev, all_sampled_epochs, os.path.join(trial_folder_path, "losses.json"))
        save_loss_plot(all_sampled_epochs, all_losses_train, all_losses_dev, os.path.join(trial_folder_path, "loss_plot.png"))
        update_sweep_log(trial.number, part_name, model_name, trial_params, mean_dev_f1, test_f1_mean, test_f1_std, test_acc_mean, test_acc_std, os.path.join(current_hydra_dir, "sweep_summary.json"))

        # Check overall best
        if mean_dev_f1 > best_sweep_f1:
            best_sweep_f1 = mean_dev_f1
            best_dir = os.path.join(current_hydra_dir, "best_model")
            logger.info(f"\nNew best model found! Saving files to {best_dir}...")
            save_model(best_model, lang.word2id, lang.slot2id, lang.intent2id, os.path.join(best_dir, "model.pt"))
            save_losses(trial.number, part_name, model_name, trial_params, mean_dev_f1, test_f1_mean, test_f1_std, test_acc_mean, test_acc_std, all_losses_train, all_losses_dev, all_sampled_epochs, os.path.join(best_dir, "losses.json"))
            save_loss_plot(all_sampled_epochs, all_losses_train, all_losses_dev, os.path.join(best_dir, "loss_plot.png"))
        return mean_dev_f1

    study = optuna.create_study(direction="maximize", sampler=sampler)
    # Force Optuna to keep trying until it gets 20 unique, successful completions or the entire grid is exhausted (whichever comes first)
    logger.info(f"\nSearching for {target_trials} unique hyperparameter combinations...")
    while len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) < target_trials:
        study.optimize(objective, n_trials=1)

    logger.info("\n================ SWEEP COMPLETE ================")
    logger.info("Best hyperparameters found:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
    logger.info(f"Best validation Dev F1: {study.best_value}")
    logger.info(f"Best model and plots have been saved to: {os.path.join(current_hydra_dir, 'best_model')}")
    
# A simple container to reconstruct the 'lang' object from saved dictionaries
class LangNamespace:
    def __init__(self, w2id, slot2id, intent2id):
        # Create the reverse mappings (ID to string) required by eval_loop
        self.id2word = {v: k for k, v in w2id.items()}
        self.id2slot = {v: k for k, v in slot2id.items()}
        self.id2intent = {v: k for k, v in intent2id.items()}


def evaluate_best_model(config, test_loader, vocab_len, out_slot, out_int, pad_index, original_cwd) -> None:
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
    logger.info(f"\n================ EVALUATING PART ================\n{part_name}\n=================================================")
    best_dir = os.path.join(original_cwd, "results", f"part{config.part}", "best_model")
    model, _ = build_model_and_optim(config, vocab_len, out_slot, out_int, pad_index)
    
    logger.info("\n--- Loading Saved Model ---")
    model, w2id, slot2id, intent2id = load_model(model, os.path.join(best_dir, "model.pt"))
    
    lang = LangNamespace(w2id, slot2id, intent2id)
    
    logger.info("\n--- Evaluating Best Model ---")
    results, report_intent, _ = eval_loop(test_loader, pad_index, model, lang)
    slot_f1 = results['total']['f']
    intent_acc = report_intent['accuracy'] 
    logger.info(f"[Final Test Results] Slot F1: {slot_f1:.4f} | Intent Accuracy: {intent_acc:.4f}")
    
    logger.info("\n--- Displaying Loss Plot ---")
    display_loss_plot(os.path.join(best_dir, "loss_plot.png"))