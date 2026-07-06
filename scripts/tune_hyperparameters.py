import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import optuna
import torch
from src.fedavg_core.utils.configuration_factory import load_config
from src.fedavg_core.optimization.hyperparameter_tuner import create_study
from src.fedavg_core.data.dataset_downloader import build_synthetic_dataset
from src.fedavg_core.data.processing_pipeline import train_test_split_dataset
from scripts.orchestrate_simulation import run_fedavg_iid

def optimize():
    cfg = load_config()
    
    # Static Data Preparation
    features, labels = build_synthetic_dataset(
        1000, cfg.model.input_size, cfg.model.num_classes, cfg.training.seed
    )
    train_feat, train_lab, test_feat, test_lab = train_test_split_dataset(
        features, labels, 0.2, cfg.training.seed
    )
    
    def objective(trial: optuna.Trial) -> float:
        # Hyperparameters to tune
        lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
        
        # Run standard FedAvg
        accuracies = run_fedavg_iid(
            train_feat, train_lab, test_feat, test_lab,
            model_config=dict(cfg.model),
            num_clients=cfg.federated.num_clients,
            num_rounds=5, # Quick simulation for tuning
            client_fraction=cfg.federated.client_fraction,
            local_epochs=cfg.training.local_epochs,
            batch_size=batch_size,
            learning_rate=lr,
            seed=cfg.training.seed
        )
        return accuracies[-1]

    # Run the study
    study = create_study("hivetorch-tuning")
    study.optimize(objective, n_trials=5)
    print("Best hyperparameters:", study.best_params)

if __name__ == "__main__":
    optimize()
