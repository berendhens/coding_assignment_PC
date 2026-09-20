import os
import json
import shutil
import torch
from datetime import datetime

def save_run(model, history, metrics, config, labels, base_dir="outputs"):
    # Create run ID based on date
    run_id = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
    # Create path to output directory, and make the directory
    run_dir = os.path.join(base_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Save final model weights
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))

    # Save evaluation metrics of final model (as JSON)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save config (as JSON)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Save tTraining history (as JSON)
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Labels definition (needed to interpret the model's output)
    shutil.copy(config["paths"]["label_path"], os.path.join(run_dir, "labels.json"))

    print(f"Run saved to: {run_dir}")
    return run_dir