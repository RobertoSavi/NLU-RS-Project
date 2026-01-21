# -\-\-\ Define the architecture of the model /-/-/-
# -------------------- Import libraries --------------------
import torch
import torch.nn as nn

# -------------------- RNN Elman version --------------------
# We are not going to use this since, for efficiency purposes, 
# it's better to use the RNN layer provided by PyTorch  
class RNN_cell(nn.Module):
    def __init__(self, hidden_size, input_size, output_size, vocab_size, dropout=0.1):
        super(RNN_cell, self).__init__()
        
        self.W = nn.Linear(input_size, hidden_size, bias=False)
        self.U = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, prev_hidden, word):
        input_emb = self.W(word)
        prev_hidden_rep = self.U(prev_hidden)

        # ht = σ(Wx + Uht-1 + b)
        hidden_state = self.sigmoid(input_emb + prev_hidden_rep)

        # yt = σ(Vht + b)
        output = self.output(hidden_state)
        return hidden_state, output

# -------------------- RNN-based language model --------------------
class LM_RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, 
                 out_dropout=0.1, emb_dropout=0.1, n_layers=1):
        super(LM_RNN, self).__init__()

        # Token IDs to vectors (embedding layer)
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)

        # PyTorch's RNN layer: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.rnn = nn.RNN(emb_size, hidden_size, n_layers, 
                          bidirectional=False, batch_first=True)

        self.pad_token = pad_index
        
        # Linear layer to project the hidden layer to output space
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        rnn_out, _ = self.rnn(emb)
        output = self.output(rnn_out).permute(0, 2, 1)
        return output
    
# -------------------- LSTM langauge model --------------------
class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, 
                 out_dropout=0.1, emb_dropout=0.1, n_layers=1):
        super(LM_LSTM, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)

        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, 
                           bidirectional=False, batch_first=True)

        self.pad_token = pad_index
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)

        rnn_out, _ = self.rnn(emb)

        output = self.output(rnn_out).permute(0, 2, 1)
        return output

# --- 1. Apply Weight Tying
class LM_LSTM_WEIGHT_TYING(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, 
                 out_dropout=0.1, emb_dropout=0.1, n_layers=1):
        super(LM_LSTM_WEIGHT_TYING, self).__init__()
        assert emb_size == hidden_size, "Weight tying requires emb_size == hidden_size"
        
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)

        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, 
                           bidirectional=False, batch_first=True)

        self.pad_token = pad_index
        # Commented as output will be computed via tied embedding weights
        #self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)

        rnn_out, _ = self.rnn(emb)

        output = torch.matmul(rnn_out, self.embedding.weight.T).permute(0, 2, 1)
        return output

# --- 2. Apply Variational Dropout (no DropConnect)
# This class applies the same dropout mask every time dropout is performed.
class LockedDropout(nn.Module):
    #Applies the same dropout mask.
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.1):
        if not self.training or dropout == 0:
            return x
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = mask.div_(1 - dropout)
        mask = mask.expand_as(x)
        return x * mask
    
class LM_LSTM_VAR_DROPOUT(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, 
                 out_dropout=0.1, emb_dropout=0.1, n_layers=1):
        super(LM_LSTM_VAR_DROPOUT, self).__init__()
        assert emb_size == hidden_size, "Weight tying requires emb_size == hidden_size"

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.emb_dropout = LockedDropout()
        self.out_dropout = LockedDropout()

        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, 
                           bidirectional=False, batch_first=True)

        self.pad_token = pad_index

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb = self.emb_dropout(emb, dropout=0.1)

        rnn_out, _ = self.rnn(emb)
        rnn_out = self.out_dropout(rnn_out, dropout=0.1)

        output = torch.matmul(rnn_out, self.embedding.weight.T).permute(0, 2, 1)
        return output
    
# --- 3. Apply Non-monotonically Triggered AvSGD
""" best_model_overall = None
    best_ppl_overall = float('inf')
    best_model_filename = ""
    best_ppls = []

    for model, optimizer, hyperparams in zip(models, optimizers, hyperparams_to_try):
        losses_train = []
        losses_dev = []
        sampled_epochs = []
        logs = []
        t = 0
        T = None
        k = 0
        avg_params = None
        avg_count = 0
        best_model = None

        learning_rate = hyperparams['lr']
        hidden_size = hyperparams['hid_size']
        embedding_size = hyperparams['emb_size']
        optimizer_name = type(optimizer).__name__
        model_name = type(model).__name__
        model_params = f"[Model: {model_name}, Optimizer: {optimizer_name}, Hidden-size: {hidden_size}, Embedding-size: {embedding_size}, Learning-rate: {learning_rate}]"

        filename = f"mod-{model_name}_opt-{optimizer_name}_hid-{hidden_size}_emb-{embedding_size}_lr-{learning_rate:.1e}"
        os.makedirs("results", exist_ok=True)
        path = os.path.join("results", filename + ".txt")
        with open(path, 'w') as f:
            f.write(model_params + '\n')
        print(model_params)

        pbar = tqdm(range(1, n_epochs + 1))

        for epoch in pbar:
            model.train()
            loss = train_loop(train_loader, optimizer, criterion_train, model, clip)

            # NT-AvSGD logic
            if epoch % L == 0 and T is None:
                model.eval()
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                v = ppl_dev

                if t > n and v > min(logs[-n:]):
                    T = k
                    avg_params = {name: p.clone().detach() for name, p in model.state_dict().items()}
                    avg_count = 1
                logs.append(v)
                t += 1

                sampled_epochs.append(epoch)
                losses_train.append(np.asarray(loss).mean())
                losses_dev.append(np.asarray(loss_dev).mean())
                pbar.set_description("PPL: %f" % ppl_dev)

                log = f"[Epoch: {epoch}, PPL: {ppl_dev:.4f}]\n"
                with open(path, 'a') as f:
                    f.write(log)
            
            elif T is not None:
                # Accumulate for parameter averaging
                for name, param in model.state_dict().items():
                    avg_params[name] += param.clone().detach()
                avg_count += 1

            k += 1

        # Apply averaged parameters if applicable
        if avg_params is not None:
            for name in avg_params:
                avg_params[name] /= avg_count
            model.load_state_dict(avg_params)

        best_model = copy.deepcopy(model).to('cpu')

        # -------------------- Final evaluation --------------------
        best_model.to(DEVICE)
        final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
        print('Test PPL:', final_ppl)
        final_log = f"[Final PPL: {final_ppl:.4f}]\n"
        with open(path, 'a') as f:
            f.write(final_log)

        print(f"Finished training model with {model_params.strip()} | Final PPL: {final_ppl:.4f}")

        best_ppls.append(final_ppl)

        if final_ppl < best_ppl_overall:
            best_ppl_overall = final_ppl
            best_model_overall = copy.deepcopy(best_model)
            best_model_filename = filename

        # Log GPU memory before cleanup
        allocated_before = torch.cuda.memory_allocated() / 1024**2
        reserved_before = torch.cuda.memory_reserved() / 1024**2
        print(f"[Memory before cleanup] Allocated: {allocated_before:.2f} MB, Reserved: {reserved_before:.2f} MB")

        del best_model  
        model.to("cpu")
        del model
        del optimizer
        torch.cuda.empty_cache()

        allocated_after = torch.cuda.memory_allocated() / 1024**2
        reserved_after = torch.cuda.memory_reserved() / 1024**2
        print(f"[Memory after cleanup] Allocated: {allocated_after:.2f} MB, Reserved: {reserved_after:.2f} MB")
        print("\n ----------------------------------- \n")

    # -------------------- Model saving --------------------
    os.makedirs("models", exist_ok=True)
    path = os.path.join("models", f"best{best_model_filename}.pt")
    torch.save(best_model_overall.state_dict(), path)

    return best_model_overall, best_ppl_overall, best_model_filename """