from fetch_data import fetch_tsqa
from build_labels import build_labels
from preprocessing import preprocess
from use_config import load_config
from split_data import split_data
#from train import train_model
#from validate import validate_model

if __name__ == "__main__":
    config = load_config("config.yaml")

    # Get data and transform to dataframe
    ds = fetch_tsqa(config["data"]["dataset_name"], cache_dir=config["data"]["cache_dir"])
    df = ds["train"].to_pandas()

    # Get labels (build JSON), preprocess, and use labels to add correct indices
    labels = build_labels(df, out_path=config["paths"]["label_path"])
    processed = preprocess(df, labels)

    # Split dataframe into train and test (stratified)
    df_train, df_val = split_data(
        processed,
        test_size=config["data"]["val_split"],
        random_state=config["seed"],
    )

    # model = train_model(processed, config)
    # validate_model(model, processed, vocab)