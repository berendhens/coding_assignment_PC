import numpy as np
import pandas as pd

from use_labels import label_to_index

def preprocess(df, labels):
    # Convert 'Series': str to np.array of floats
    df["Series"] = [np.fromstring(s.strip('[]'), sep=',', dtype=np.float32) if pd.notna(s) else s for s in df["Series"]]

    # Only retain the series with size 64 --> easy shortcut to get same inputsizes
    df = df[df["Size"] == 64].copy().reset_index(drop=True)

    # Normalize each series individually 
    eps = 1e-8 
    df["Series"] = [(arr - np.mean(arr)) / (np.std(arr) + eps) for arr in df["Series"]]

    # Create extra column with indices for 'Label' in each 'Task'
    df = add_label_index(df, labels)

    return df

def add_label_index(df, labels):
    """TODO: NEED EXPLANATION"""
    df["label_idx"] = df.apply(
        lambda row: label_to_index(labels, row["Task"], row["Label"]),
        axis=1
    )
    return df