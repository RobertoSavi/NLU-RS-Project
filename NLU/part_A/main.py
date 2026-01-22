# -\-\-\ Run the training of the model and save the results /-/-/-
# -------------------- Import libraries --------------------
import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# -------------------- Import functions from other files --------------------
from models import *
from utils import *
from params import *

best_f1_overall = 0
best_model_overall = None
best_optimizer_overall = None
best_model_filename = None
results = []

# For each model and optimizer
for model, optimizer, hyperparams in zip(models, optimizers, hyperparams_to_try):

    slot_f1s = []
    intent_accs = []
    
    best_f1_config = 0
    best_model_config = None
    best_optimizer_config = None
    
    lr = hyperparams['lr']
    hid_size = hyperparams['hid_size']
    emb_size = hyperparams['emb_size']
    
    model_name = type(model).__name__
    optimizer_name = type(optimizer).__name__
    model_params = f"[Model: {model_name}, Optimizer: {optimizer_name}, Hidden-size: {hid_size}, Embedding-size: {emb_size}, Learning-rate: {lr}]"
    # Create a file on which to store the results
    filename = f"mod-{model_name}_opt-{optimizer_name}_hid-{hid_size}_emb-{emb_size}_lr-{lr:.1e}"
    # Create 'results' folder if it doesn't exist
    os.makedirs("results", exist_ok=True)
    # Full path to the file
    path = os.path.join("results", filename + ".txt")
    # Create the file (empty or write something if you want)
    with open(path, 'w') as f:
        f.write(model_params + '\n')
    print(model_params)

    for x in tqdm(range(0, runs)):    
        patience = patience_value  # Reset patience for each model
        losses_train = []
        losses_dev = []
        sampled_epochs = []
        
        best_f1_run = 0
        best_model_run = None
        best_optimizer_run = None
        
        model = ModelIAS(hid_size, out_slot, out_int, emb_size, 
                     vocab_len, pad_index=PAD_TOKEN).to(DEVICE)
        model.apply(init_weights)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        pbar = tqdm(range(1, n_epochs+1))
        for epoch in pbar:
            loss = train_loop(train_loader, optimizer, criterion_slots, 
                        criterion_intents, model)
             
            if epoch % 5 == 0: # We check the performance every 5 epochs
                sampled_epochs.append(epoch)
                losses_train.append(np.asarray(loss).mean())
                results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, 
                                                            criterion_intents, model, lang)
                losses_dev.append(np.asarray(loss_dev).mean())
                
                f1 = results_dev['total']['f']
                pbar.set_description("f1: %f" % f1)
                # For decreasing the patience you can also use the average between slot f1 and intent accuracy
                if f1 > best_f1_run:
                    best_f1_run = f1
                    best_model_run = copy.deepcopy(model).to('cpu')
                    best_optimizer_run = copy.deepcopy(optimizer)
                    patience = patience_value  # Reset patience
                else:
                    patience -= 1                
                if patience <= 0: # Early stopping with patience
                    break # Not nice but it keeps the code clean

        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, 
                                                criterion_intents, best_model_run, lang)
        slot_f1s.append(results_test['total']['f'])
        intent_accs.append(intent_test['accuracy'])
        
        # Keep best run model for this hyperparam configuration
        if results_test['total']['f'] > best_f1_config:
            best_f1_config = results_test['total']['f']
            best_model_config = best_model_run
            
        # Log GPU memory before cleanup
        allocated_before = torch.cuda.memory_allocated() / 1024**2
        reserved_before = torch.cuda.memory_reserved() / 1024**2
        print(f"[Memory before cleanup] Allocated: {allocated_before:.2f} MB, Reserved: {reserved_before:.2f} MB")

        # Release GPU memory right after evaluation
        del model
        del optimizer
        torch.cuda.empty_cache()

        # Log GPU memory after cleanup
        allocated_after = torch.cuda.memory_allocated() / 1024**2
        reserved_after = torch.cuda.memory_reserved() / 1024**2
        print(f"[Memory after cleanup] Allocated: {allocated_after:.2f} MB, Reserved: {reserved_after:.2f} MB")
        print("\n ----------------------------------- \n")
            
    slot_f1s = np.asarray(slot_f1s)
    intent_accs = np.asarray(intent_accs)

    log = f"Slot F1: {slot_f1s.mean():.3f} ± {slot_f1s.std():.3f}\n Intent Acc: {intent_accs.mean():.3f} ± {intent_accs.std():.3f}\n"
    # Add to overall results
    results.append(log)
    # Print it
    print(log)
    # Append it to file
    with open(path, 'a') as f:
        f.write(log )
        
    
    # Track the best overall model
    if best_f1_config > best_f1_overall:
        best_f1_overall = best_f1_config
        best_model_filename = filename
        best_model_overall = copy.deepcopy(best_model_config).to('cpu')
        best_optimizer_overall = copy.deepcopy(best_optimizer_config)
        
    model_data_save = {
        "model": best_model_overall.state_dict(), 
        "optimizer": best_optimizer_overall.state_dict(), 
        "w2id": w2id, 
        "slot2id": slot2id, 
        "intent2id": intent2id
    }
    

# -------------------- Model saving --------------------
# Create 'models' folder if it doesn't exist
os.makedirs("models", exist_ok=True)
# Full path to the file
path = os.path.join("models", f"best-{best_model_filename}.pt")
torch.save(model_data_save, path)

with open('results/overall_training_results_baseline_IAS.txt', 'w') as f:
    for i, (result, model, optimizer, hyperparams) in enumerate(zip(results, models, optimizers, hyperparams_to_try)):
        entry = f"Model {i}: [Result: {result}, Optimizer: {type(optimizer).__name__}, Hidden-size: {hyperparams['hid_size']}, Embedding-size: {hyperparams['emb_size']}, Learning-rate: {hyperparams['lr']}, Model: {model}]\n"
        f.write(entry)

""" # Plotting train and dev losses
plt.figure(num = 3, figsize=(8, 5)).patch.set_facecolor('white')
plt.title('Train and Dev Losses')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.plot(sampled_epochs, losses_train, label='Train loss')
plt.plot(sampled_epochs, losses_dev, label='Dev loss')
plt.legend()
plt.show() """