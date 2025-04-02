# -\-\-\ Run the training of the model and save the results
# -------------------- Import libraries --------------------
import copy
import numpy as np
from tqdm import tqdm
# -------------------- Import functions from other files --------------------
from functions import *
from utils import *
from model import *

# -------------------- Training process --------------------

losses_train = []
losses_dev = []
sampled_epochs = []
best_ppl = math.inf
best_model = None

pbar = tqdm(range(1, n_epochs))
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

# -------------------- Model saving --------------------
path = 'models/standard.pt'
torch.save(model.state_dict(), path)

# To load the model:
# model = LM_RNN(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
# model.load_state_dict(torch.load(path))
