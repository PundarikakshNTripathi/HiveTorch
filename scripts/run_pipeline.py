import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from src.fedavg_core.utils.configuration_factory import load_config
from src.fedavg_core.data.dataset_downloader import build_synthetic_dataset
from src.fedavg_core.data.processing_pipeline import train_test_split_dataset
from scripts.orchestrate_simulation import run_fedavg_iid

def main():
    # 1. Load config dynamically using the factory we built
    cfg = load_config()
    print("Loaded Configuration:", cfg)
    
    # 2. Prepare Data
    print("Generating Synthetic Data...")
    features, labels = build_synthetic_dataset(
        num_samples=1000, 
        input_size=cfg.model.input_size, 
        num_classes=cfg.model.num_classes, 
        seed=cfg.training.seed
    )
    
    train_feat, train_lab, test_feat, test_lab = train_test_split_dataset(
        features, labels, test_fraction=0.2, seed=cfg.training.seed
    )
    
    # 3. Run Pipeline
    print(f"Starting Federated Pipeline with {cfg.federated.num_clients} clients...")
    accuracies = run_fedavg_iid(
        train_features=train_feat,
        train_labels=train_lab,
        test_features=test_feat,
        test_labels=test_lab,
        model_config=dict(cfg.model),
        num_clients=cfg.federated.num_clients,
        num_rounds=cfg.federated.num_rounds,
        client_fraction=cfg.federated.client_fraction,
        local_epochs=cfg.training.local_epochs,
        batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.learning_rate,
        seed=cfg.training.seed
    )
    print("Pipeline Complete! Final Accuracy:", accuracies[-1])

if __name__ == "__main__":
    main()
