import json
import pandas as pd
from datasets import load_dataset


def build_labels(df: pd.DataFrame, out_path: str="data/labels.json"):
    """Derive the class vocabulary per task from the raw dataset and persist it to disk.
    
    Inspects the unique (Task, Label) combinations once and builds a fixed,
    ordered class list per task. This vocabulary is the single source of
    truth for label <-> index conversion.

    Args:
        df: Raw TSQA dataframe with 'Task' and 'Label' columns.
        out_path: Where to write the resulting labels.json.

    Returns:
        dict with keys 'attributes' (list of task names) and
        'classes' (dict mapping each task to its ordered list of class labels)
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