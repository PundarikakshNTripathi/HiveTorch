import optuna
import torch

def create_study(study_name: str) -> optuna.Study:
    """Initializes an Optuna study for hyperparameter tuning."""
    return optuna.create_study(study_name=study_name, direction="maximize")

def objective(trial: optuna.Trial) -> float:
    """Example objective function for Optuna. Should be injected with actual FedAvg loop."""
    # Placeholder for sweeping
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    # Returns a dummy score to be replaced in production
    return 0.95 
