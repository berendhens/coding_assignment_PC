from sklearn.model_selection import train_test_split

def split_data(df, test_size=0.2, random_state=67):
    """TODO: NEED EXPLANATION
    Stratified split by Task + Label combined, so every task's every class
    is represented proportionally in both train and validation sets.
    """
    # Stratify on Task + Label, ensuring enough cases in both train and test
    stratify = df["Task"] + "_" + df["Label"]

    df_train, df_val = train_test_split(
        df,
        test_size=test_size,
        stratify=stratify,
        random_state=random_state,
    )

    return df_train.reset_index(drop=True), df_val.reset_index(drop=True)