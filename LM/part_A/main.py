# Configuration management imports
import hydra
from omegaconf import DictConfig, OmegaConf
import os
# Import data pipeline and training utilities
from utils import init_data_pipeline
from functions import run_sweep, evaluate_best_model

# Main pipeline
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(f"\n================ STARTING PIPELINE ================")
    original_cwd = hydra.utils.get_original_cwd()
    current_hydra_dir = os.getcwd()
    
    OmegaConf.set_struct(cfg, False)
    active_params = cfg.part.parameters if cfg.testing else cfg.part.best_parameters
    config = OmegaConf.merge(cfg.part, active_params)
    
    # Prepare datasets and dataloaders
    lang, train_loader, dev_loader, test_loader = init_data_pipeline(
        os.path.join(original_cwd, "dataset/PennTreeBank/ptb.train.txt"),
        os.path.join(original_cwd, "dataset/PennTreeBank/ptb.valid.txt"),
        os.path.join(original_cwd, "dataset/PennTreeBank/ptb.test.txt"),
        train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size
    )
    pad_index = lang.word2id["<pad>"]
    vocab_len = len(lang.word2id)
    
    if cfg.testing:
        run_sweep(config, active_params, train_loader, dev_loader, test_loader, vocab_len, pad_index, current_hydra_dir)
    else:
        evaluate_best_model(config, test_loader, vocab_len, pad_index, original_cwd)
        
if __name__ == "__main__":
    main()