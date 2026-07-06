
class ConceptDriftDetector:
    """Detects statistical drift in client updates."""
    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold
        
    def check_accuracy_drift(self, baseline_acc: float, current_acc: float) -> bool:
        """Returns True if the current accuracy drops significantly below the baseline."""
        # If accuracy drops by more than the threshold fraction from the baseline
        deviation = (baseline_acc - current_acc) / (baseline_acc + 1e-9)
        return deviation > self.threshold
