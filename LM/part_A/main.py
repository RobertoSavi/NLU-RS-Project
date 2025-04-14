# -\-\-\ Run the training of the model and save the results /-/-/-
# -------------------- Import libraries --------------------
import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
# -------------------- Import functions from other files --------------------
from functions import *
from utils import *
from model import *

# -------------------- Training process --------------------

# Multi training parameters
best_ppls = []
best_ppl_overall = math.inf
best_model_overall = None

print(models, optimizers)
# For each model and optimizer
for model, optimizer in zip(models, optimizers):
    # Single training parameters
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1, n_epochs))
    print(model)
    print(optimizer)

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
                patience = 3
            else:
                patience -= 1
                
            if patience <= 0:  # Early stopping with patience
                break  # Clean exit when training stops

    # -------------------- Final evaluation --------------------
    best_model.to(DEVICE)
    final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
    print('Test PPL:', final_ppl)

    # Store the best ppl of this configuration
    best_ppls.append(final_ppl)
    
    # Track the best overall model
    if final_ppl < best_ppl_overall:
        best_ppl_overall = final_ppl
        best_model_overall = copy.deepcopy(best_model)

# -------------------- Model saving --------------------
path = 'models/LSTM_best_lr.pt'
torch.save(best_model_overall.state_dict(), path)

# To load the model:
# model = LM_RNN(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
# model.load_state_dict(torch.load(path))

# -------------------- Save best PPL results --------------------
with open('results/models_LSTM_best_lr', 'w') as f:
    for i, (ppl, model) in enumerate(zip(best_ppls, models)):
        f.write(f'Model {i}: PPL = {ppl}, Model = {model}, Optimizer = {optimizer}\n')

# -------------------- Plot PPL results --------------------
plt.figure(figsize=(10, 5))
plt.plot(range(len(best_ppls)), best_ppls, marker='o', linestyle='-', label='Best PPL per Model')
plt.xlabel('Model index')
plt.ylabel('Perplexity (PPL)')
plt.title('Best PPL for each Model-Optimizer configuration')
plt.legend()
plt.savefig('images/models_LSTM_best_lr.jpg')  # Save the plot as an image
plt.show()
