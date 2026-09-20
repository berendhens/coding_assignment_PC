from sklearn.model_selection import train_test_split

def split_data(df, test_size=0.2, random_state=67):
    """Split into train/validation sets, stratified by Task + Label combined.

    Stratifying on the combined key (rather than Task or Label alone)
    ensures every task's every class is represented proportionally.

    Args:
        df: Preprocessed dataframe with 'Task' and 'Label' columns.
        test_size: Fraction of rows to hold out for validation.
        random_state: Seed for reproducible splitting.

    Returns:
        (df_train, df_val), both with reset integer indices.
    """
    # Stratify on Task + Label, ensuring enough cases in both train and test
    # as it gives proportional division.
    stratify = df["Task"] + "_" + df["Label"]

    df_train, df_val = train_test_split(
        df,
        test_size=test_size,
        stratify=stratify,
        random_state=random_state,
    )

    return df_train.reset_index(drop=True), df_val.reset_index(drop=True)