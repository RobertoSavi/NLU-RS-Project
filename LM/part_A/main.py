# Configuration management imports
from logging import config

import hydra
from omegaconf import DictConfig, OmegaConf
import optuna
import os

# Import data pipeline and training utilities
from utils import init_data_pipeline
from functions import (
    build_model_and_optim,
    train_model,
    eval_model,
    save_model,
    load_model,
    save_losses,
    plot_losses,
    free_memory,
    update_sweep_log
)

# Main part pipeline
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    
    print(f"\n================ STARTING PIPELINE ================")
    # Prepare datasets and dataloaders
    original_cwd = hydra.utils.get_original_cwd()
    lang, train_loader, dev_loader, test_loader = init_data_pipeline(
        os.path.join(original_cwd, "dataset/PennTreeBank/ptb.train.txt"),
        os.path.join(original_cwd, "dataset/PennTreeBank/ptb.valid.txt"),
        os.path.join(original_cwd, "dataset/PennTreeBank/ptb.test.txt"),
        train_batch_size=cfg.part.train_batch_size,
        eval_batch_size=cfg.part.eval_batch_size
    )
    pad_index = lang.word2id["<pad>"]
    vocab_len = len(lang.word2id)
    
    if cfg.testing:
        print("\n================ RUNNING OPTUNA SWEEP ================")
        
        # 1. Figure out which parameters are actually varying
        params_node = cfg.part.parameters
        base_params_dict = OmegaConf.to_container(params_node, resolve=True)
        varying_keys = []
        for k, v in base_params_dict.items():
            # If it's a list with more than 1 item, it's a varying parameter
            if isinstance(v, (list, tuple)) and len(v) > 1:
                varying_keys.append(k)
        
        print(f"Varying parameters for folder naming: {varying_keys}")

        best_sweep_loss = float('inf')
        
        def objective(trial):
            nonlocal best_sweep_loss
            
            # Dynamically suggest parameters
            trial_params = {}
            for key, value in cfg.part.parameters.items():
                if isinstance(value, (list, tuple)) or type(value).__name__ == "ListConfig":
                    # Only suggest from the list if it has multiple items, otherwise just use it
                    if len(value) > 1:
                        trial_params[key] = trial.suggest_categorical(key, list(value))
                    else:
                        trial_params[key] = value[0]
                else:
                    trial_params[key] = value

            config = OmegaConf.merge(cfg.part, trial_params)
            
            print(f"\n--- Trial {trial.number} ---")
            print(f"Testing params: {trial_params}")
            
            # Build and train
            model, optimizer = build_model_and_optim(config, vocab_len, pad_index)
            best_model, losses_train, losses_dev = train_model(
                config, model, optimizer, train_loader, dev_loader, pad_index
            )
            
            # Evaluate this trial's best model
            print(f"\n--- Evaluating Trial {trial.number} Model ---")
            trial_ppl = eval_model(best_model, test_loader, pad_index)
            
            # Log the results of this trial
            update_sweep_log("log.json", trial.number, trial_params, trial_ppl)
            
            # Create the dynamic folder name based on the varying parameters
            if varying_keys:
                folder_name = "_".join([f"{k}={trial_params[k]}" for k in varying_keys])
            else:
                folder_name = f"trial_{trial.number}"
                
            # Hydra already put us in the sweep folder, so we just use the dynamic name
            save_losses(losses_train, losses_dev, folder_name)
            plot_losses(losses_train, losses_dev, save_path=os.path.join(folder_name, "loss_plot.png"), testing=True)
            
            # Check if this is the best overall model
            best_val_loss = min(losses_dev)
            if best_val_loss < best_sweep_loss:
                best_sweep_loss = best_val_loss
                
                # Save the absolute best artifacts to a dedicated folder in this sweep directory
                best_dir = "best_model"
                print(f"\nNew best model found! Saving files to {best_dir}...")
                
                save_model(best_model, best_dir)
                save_losses(losses_train, losses_dev, best_dir)
                plot_losses(losses_train, losses_dev, save_path=os.path.join(best_dir, "loss_plot.png"), testing=True)
            
            # Free memory before the next trial
            free_memory(model, best_model, optimizer)
            
            return best_val_loss

        # Create and run the study
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=20) 
        
        print("\n================ SWEEP COMPLETE ================")
        print("Best hyperparameters found:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        print(f"Best validation loss: {study.best_value}")
        print(f"Best model and plots have been saved to: {os.path.join(os.getcwd(), 'best_model')}")
        
    else:
        active_params = cfg.part.best_parameters
        config = OmegaConf.merge(cfg.part, active_params)
        
        part_name = (
            f"part={config.get('part', 'N/A')}\n"
            f"testing={config.get('testing', 'N/A')}\n"
            f"hid_size={config.get('hid_size', 'N/A')}\n"
            f"emb_size={config.get('emb_size', 'N/A')}\n"
            f"optimizer={config.get('optimizer', 'N/A')}\n"
            f"lr={config.get('lr', 'N/A')}\n"
            f"train_bs={config.get('train_batch_size', 'N/A')}\n"
            f"eval_bs={config.get('eval_batch_size', 'N/A')}"
        )

        print(f"\n================ EVALUATING PART ================\n{part_name}\n=================================================")

        # Build dummy model architecture to load the saved weights into
        model, _ = build_model_and_optim(config, vocab_len, pad_index)

        # Load the pre-trained model weights
        best_dir = os.path.join(os.getcwd(), "best_model")
        model_path = os.path.join(best_dir, "model.pt")
        plot_path = os.path.join(best_dir, "loss_plot.png")

        # Load weights
        print("\n--- Loading Saved Model ---")
        model = load_model(model, model_path)

        # Evaluate the model
        print("\n--- Evaluating Best Model ---")
        eval_model(model, test_loader, pad_index)

        # Display the previously saved plot
        print("\n--- Displaying Loss Plot ---")
        plot_losses(save_path=plot_path, testing=False)
        
# Run pipeline entry point
if __name__ == "__main__":
    main()