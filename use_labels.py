import json

def load_label_vocab(path="data/labels.json"):
    with open(path) as f:
        return json.load(f)

def label_to_index(labels, task, label):
    return labels["classes"][task.lower()].index(label)

def index_to_label(labels, task, idx):
    return labels["classes"][task.lower()][idx]