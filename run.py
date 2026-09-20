from fetch_data import fetch_tsqa
from build_labels import build_labels
from preprocessing import preprocess
from use_config import load_config
from split_data import split_data
from train import train_model
from validate import validate_model
from save_run import save_run

if __name__ == "__main__":
    config = load_config("config.yaml")

    # Get data and transform to dataframe
    ds = fetch_tsqa(config["data"]["dataset_name"], cache_dir=config["data"]["cache_dir"])
    df = ds["train"].to_pandas()
    print('Data gathered. Transformed to pd.Dataframe.')

    # Get labels (build JSON), preprocess, and use labels to add correct indices
    labels = build_labels(df, out_path=config["paths"]["label_path"])
    print('Finished build_labels.')
    processed = preprocess(df, labels)
    print('Finished preprocessing.')

    # Split dataframe into train and test (stratified)
    df_train, df_val = split_data(
        processed,
        test_size=config["data"]["val_split"],
        random_state=config["seed"],
    )

    model, history = train_model(df_train, df_val, config, labels)
    print(f"Training complete. Final val_loss: {history[-1]['val_loss']:.4f}")

    metrics = validate_model(model, df_val, labels, config)
    print(f"\nOverall accuracy: {metrics['macro_avg']['accuracy']:.3f}, \n Overall F1: {metrics['macro_avg']['f1']:.3f}")

    run_directory = save_run(model, history, metrics, config, labels, base_dir="outputs")