# train.py
import torch
from torch.utils.data import Dataset, DataLoader
from model import ModelPC, masked_loss

class CreateDataset(Dataset):
    # Allows to transform created pd.dataframe into torch.Dataset
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        # Model only needs time-series, task assigned to series, and the GT label
        series = torch.tensor(row["Series"], dtype=torch.float32)
        label_idx = torch.tensor(row["label_idx"], dtype=torch.long)
        task = row["Task"].lower() # Make task lower cases, to match earlier made lowercases
        return series, label_idx, task

def train_model(df_train, df_val, config, labels):
    """TODO: Create docstring"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create Dataset that can be loaded by PyTorch for both train and val
    # and load data
    train_ds = CreateDataset(df_train)
    val_ds = CreateDataset(df_val)
    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"], shuffle=False)

    # Load the model architecture to device
    model = ModelPC(
        attributes=labels["attributes"],
    ).to(device)

    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    history = []
    best_val_loss = float("inf")
    best_state = None

    # Run for as much epochs are defined in configuration
    for epoch in range(config["training"]["epochs"]):
        # Go to training mode, reset training loss to 0 (for history)
        model.train()
        train_loss = 0.0
        # Run through dataset in batches (created by DataLoader)
        for series, label_idx, task in train_loader:
            series, label_idx = series.to(device), label_idx.to(device)
            # Clear gradients, run model on batch, calculate masked loss, perform backward propagation, update weights
            optimizer.zero_grad()
            outputs = model(series)
            loss = masked_loss(outputs, label_idx, task)
            loss.backward()
            optimizer.step()
            # Updates train_loss with batch loss (mean batch loss * batch length)
            train_loss += loss.item() * len(series)

        # After all batches, divide total loss by number of samples --> average loss of epoch 
        train_loss /= len(train_ds)

        # Calculate mean validation loss after this epoch
        val_loss = evaluate_loss(model, val_loader, device)

        # Create history point, print results at this epoch (only based on loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch+1}/{config['training']['epochs']}: train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

        # Check if loss this epoch is lower, lowest loss selected as best state (and kept)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

    # Restore best checkpoint (as final model) before returning
    model.load_state_dict(best_state)  
    return model, history

def evaluate_loss(model, loader, device):
    """TODO: Create docstring"""
    # Go to evaluation mode, reset loss to 0
    model.eval() # No computation of gradients/adaptation of weights possible
    total_loss = 0.0
    with torch.no_grad():
        # Run through validation data in batches
        for series, label_idx, task in loader:
            series, label_idx = series.to(device), label_idx.to(device)
            # Forward pass of data through model
            outputs = model(series)
            # Calculate (masked) validation loss
            loss = masked_loss(outputs, label_idx, task)
            # Add loss of batch to total validation loss
            total_loss += loss.item() * len(series)
    return total_loss / len(loader.dataset)