import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from train import CreateDataset  

def validate_model(model, df_val, labels, config, device=None):
    """TODO: Fix docstring.
    Computes per-task accuracy, F1, and a random-baseline comparison.
    Returns a metrics dict ready to be saved to disk.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    val_ds = CreateDataset(df_val)
    val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"], shuffle=False)

    # Collect predictions per task
    per_task_preds = {attr: [] for attr in labels["attributes"]}
    per_task_labels = {attr: [] for attr in labels["attributes"]}
    per_task_confidence = {attr: [] for attr in labels["attributes"]}

    with torch.no_grad():
        # Run through validation data in batches
        for series, label_idx, task in val_loader:
            series = series.to(device)
            outputs = model(series) # Caculate outputs of a forward progression through the model

            # Run through each task, as output is only relevant for specific assigned task
            for attr in labels["attributes"]:
                # Create mask to indicate which entries have the assigned task
                mask = np.array([t == attr for t in task])
                if not mask.any():
                    continue

                # Only take logits of the attr head for the entries that have that attr assigned as task
                logits = outputs[attr][torch.tensor(mask)]
                # Using softmax activation, make from (X, 3) logits probabilities for each of 3 classes within task
                probs = torch.softmax(logits, dim=-1)
                # Create predicted class and confidence - no more prob for each of the classes in attr, but only one assigned label and its confidence
                confidences, preds = probs.max(dim=-1)

                # Assign output to collecting dictionary (at correct key for the attribution)
                per_task_preds[attr].extend(preds.cpu().tolist())
                per_task_labels[attr].extend(label_idx[mask].tolist()) # GT, with mask only correct entries are selected
                per_task_confidence[attr].extend(confidences.cpu().tolist())

    # Compute metrics per task
    metrics = {"per_task": {}, "macro_avg": {}}
    accs, f1s = [], []

    for attr in labels["attributes"]:
        # Get the GT and prediction for the attr
        y_true = per_task_labels[attr]
        y_pred = per_task_preds[attr]

        if len(y_true) == 0:
            continue  # no validation samples for this task, skip

        # Calculate acc and f1 as metric, also check the mean confidence of each assigned class
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        mean_conf = float(np.mean(per_task_confidence[attr]))

        # Random-classifier baseline for comparison (uniform over 3 classes)
        rng = np.random.default_rng(config["seed"])
        random_preds = rng.integers(0, len(labels["classes"][attr]), size=len(y_true)) # Choose random integers between 0 and 2 for the amount of true examples there are --> random guessing class
        random_acc = accuracy_score(y_true, random_preds) # Calculate accuracy of random guessed classes to GT

        metrics["per_task"][attr] = {
            "accuracy": acc,
            "f1_macro": f1,
            "mean_confidence": mean_conf,
            "n_samples": len(y_true),
            "random_baseline_accuracy": random_acc,
        }

        # Add the acc and f1 for the attr, to calculate overall acc and f1
        accs.append(acc)
        f1s.append(f1)

    # Calculate averages of metrics over all attributes, add them to results
    metrics["macro_avg"]["accuracy"] = float(np.mean(accs))
    metrics["macro_avg"]["f1"] = float(np.mean(f1s))

    return metrics