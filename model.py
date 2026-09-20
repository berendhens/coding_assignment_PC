# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelPC(nn.Module):
    def __init__(self, num_classes_per_task=3, attributes=None):
        super().__init__()
        self.attributes = attributes or ["outlier", "seasonality", "trend", "volatility"] # Take hardcoded attributes if None are given

        # Shared 1D CNN backbone
        self.backbone = nn.Sequential(
            # First layer
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Second layer
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Last backbone layer --> need length of 1 to go to last layer
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # Define feature dimension (hardcoded as layers are hardcoded as well, simplicity)
        feature_dim = 128

        # One independent softmax head per attribute (each head 3 neurons --> due to 3 classes)
        self.heads = nn.ModuleDict({
            attr: nn.Linear(feature_dim, num_classes_per_task) 
            for attr in self.attributes
        })

    def forward(self, x):
        # Create the path of data through the model
        # x: (batch, seq_len) -> unsqueeze -> (batch, 1, seq_len)
        x = x.unsqueeze(1)
        features = self.backbone(x).squeeze(-1) # (batch, feature_dim, 1) -> unsqueeze -> (batch, feature_dim)

        # Raw logits per head (softmax applied later, inside the loss / at inference)
        # Each attr will have (batch, 3) tensor -> only not all 'rows' in tensor have meaning
        # Only the rows from entries designated in the beginning to this attr are meaningful
        return {attr: head(features) for attr, head in self.heads.items()}

def masked_loss(outputs, batch_labels, batch_tasks):
    """Compute cross-entropy loss per attribute head, using only matching samples.

    Each sample in a batch has ground truth for exactly one attribute (its
    'Task'). For every head, this masks the batch down to just the samples
    belonging to that task and computes standard cross-entropy on them,
    then sums the per-head losses into a single scalar, the batch loss.

    Args:
        outputs: dict[str, Tensor], one (batch, 3) logits tensor per attribute head.
        batch_labels: Tensor of shape (batch,), integer class index per sample. Only index for assigned 'Task'.
        batch_tasks: list/array of shape (batch,), the 'Task' name per sample.

    Returns:
        total_loss: summed cross-entropy loss across all matched heads.
    """
    total_loss = 0.0
    matched_head = False
    for attr in outputs:
        # Create mask for attr in batch, all correct attr indicated as True
        mask = [t == attr for t in batch_tasks]
        if not any(mask):
            continue # Don't calculate loss for attr if this attr doesn't occur in batch
        matched_head = True
        # Convert mask into Torch Tensor for loss calculation
        idx = torch.tensor(mask, dtype=torch.bool)
        # Calculate loss for this attribute (cross entropy) by comparing outputs with GT
        # outputs[attr][idx] selects only the outputs of attr that actually have attr as listed attribute in database
        # shape of tensor goes from (B, 3) to (X, 3) with X amount of attr designated time-series in the batch
        # Add this loss to total batch loss
        total_loss += F.cross_entropy(outputs[attr][idx], batch_labels[idx])

    # Check if there is any match in attribute in the batch --> this can't be not True
    if not matched_head:
        raise ValueError(f"No task in batch matched any head. Heads: {list(outputs.keys())}, batch tasks: {set(batch_tasks)}")

    return total_loss