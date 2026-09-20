from datasets import load_dataset

def fetch_tsqa(dataset_name: str, cache_dir: str="./data/raw"):
    """Retrieve Dataset ChengsenWang/TSQA."""
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    return dataset

if __name__ == "__main__":
    ds = fetch_tsqa()
    print(ds['train']['Size'])