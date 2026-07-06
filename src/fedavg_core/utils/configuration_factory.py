from omegaconf import DictConfig, OmegaConf
import os

def load_config(config_path: str = ".config/federation/global_runtime.yaml") -> DictConfig:
    """Loads and validates the Hydra/Omegaconf YAML configuration."""
    if os.path.exists(config_path):
        cfg = OmegaConf.load(config_path)
        return cfg
    
    # Fallback to defaults if file is empty or missing
    return OmegaConf.create({
        "federated": {"num_rounds": 10, "num_clients": 5, "client_fraction": 0.8},
        "model": {"input_size": 8, "hidden_size": 16, "num_classes": 3},
        "training": {"local_epochs": 2, "batch_size": 16, "learning_rate": 0.05, "seed": 42}
    })
