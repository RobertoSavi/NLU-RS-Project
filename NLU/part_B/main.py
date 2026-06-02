# Configuration management imports
import hydra
from omegaconf import DictConfig, OmegaConf
import os
# Import data pipeline and training utilities
from utils import init_data_pipeline
from functions import run_sweep, evaluate_best_model
import logging
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Main pipeline
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    logger.info(f"\n================ STARTING PIPELINE ================")
    original_cwd = hydra.utils.get_original_cwd()
    current_hydra_dir = os.getcwd()

    OmegaConf.set_struct(cfg, False)
    active_params = cfg.part.parameters if cfg.testing else cfg.part.best_parameters
    config = OmegaConf.merge(cfg.part, active_params)

    train_loader, dev_loader, test_loader, out_slot, out_int, lang, pad_index = init_data_pipeline(
        os.path.join(original_cwd, "dataset/ATIS/train.json"),
        os.path.join(original_cwd, "dataset/ATIS/test.json"),
        train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size
    )
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    if cfg.testing:
        run_sweep(config, active_params, train_loader, dev_loader, test_loader, lang, tokenizer, out_slot, out_int, pad_index, current_hydra_dir)
    else:
        evaluate_best_model(config, test_loader, out_slot, out_int, tokenizer, pad_index, original_cwd)

if __name__ == "__main__":
    main()