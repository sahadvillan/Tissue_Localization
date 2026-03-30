import torch

class MCDropoutWrapper:
    """
    A utility class to inject MC-Dropout into a PyTorch model 
    and perform stochastic inference.
    """
    
    def __init__(self, model, dropout_p=0.1):
        self.model = model
        self.dropout_p = dropout_p
        self.enabled = False

    def enable_mc_dropout(self):
        """
        Recursively finds all Dropout layers in the model and enables them
        during inference (sets to train mode).
        """
        count = 0
        for m in self.model.modules():
            # Use class name string for better robustness against module variations
            if m.__class__.__name__ == "Dropout":
                m.p = self.dropout_p
                m.train() # Force active during eval
                count += 1
        self.enabled = True
        return count

    def disable_mc_dropout(self):
        """
        Disables dropout by returning those layers to eval mode.
        """
        count = 0
        for m in self.model.modules():
            if m.__class__.__name__ == "Dropout":
                m.eval()
                count += 1
        self.enabled = False
        return count


    def stochastic_forward(self, *args, n_samples=10, **kwargs):
        """
        Runs multiple forward passes and collects results.
        Returns a list of outputs.
        """
        if not self.enabled:
            # Optionally enable if not already done, or just warn
            self.enable_mc_dropout()
            
        outputs = []
        for _ in range(n_samples):
            with torch.no_grad():
                out = self.model(*args, **kwargs)
                # Ensure we clone/detach if necessary, though model(video) 
                # for CoTracker usually returns (tracks, visibility)
                outputs.append(out)
        return outputs
