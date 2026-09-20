import json
import pandas as pd
from datasets import load_dataset


def build_labels(df: pd.DataFrame, out_path: str="data/labels.json"):
    """
    TODO: FIX DOCSTRING TO ONE FORMAT
    Inspect the dataset once, derive the class list per task,
    and persist it as the single source of truth for label <-> index mapping.
    """
    labels = {}
    # Run through all tasks (only once) --> sorted() for reproducibility
    for task in sorted(df["Task"].unique()):
        # All classes in "Task" are the unique "Label"
        classes = sorted(df.loc[df["Task"] == task, "Label"].unique())
        # Create key of "Task" with all its classes
        labels[task.lower()] = classes

    labels_json = {"attributes": list(labels.keys()), "classes": labels}

    # Write away as JSON in data as labels.json
    with open(out_path, "w") as f:
        json.dump(labels_json, f, indent=2)

    return labels_json

if __name__ == "__main__":
    df = load_dataset("ChengsenWang/TSQA")["train"].to_pandas()
    labels = build_labels(df)
    print(labels)