import pandas as pd
from thesis_graph.data import load_thesis_csv

from thesis_graph.utils import base_data_path


def train_test_split_thesis_df(df: pd.DataFrame, train_ratio: float, val_ratio: float):
    df = df.sort_values(by="application_date", ascending=True)
    df = df.reset_index(drop=True)

    train_cut = int(df.shape[0] * train_ratio)
    val_cut = int(df.shape[0] * (train_ratio + val_ratio))

    train_df, val_df, test_df = (
        df.iloc[:train_cut],
        df.iloc[train_cut:val_cut],
        df.iloc[val_cut:],
    )

    return train_df, val_df, test_df


def main():
    thesis_df = load_thesis_csv(base_data_path / "committee.csv")
    train_df, val_df, test_df = train_test_split_thesis_df(
        thesis_df, train_ratio=0.8, val_ratio=0.1
    )

    train_df.to_csv(base_data_path / "thesis_train.csv", index=False)
    val_df.to_csv(base_data_path / "thesis_val.csv", index=False)
    test_df.to_csv(base_data_path / "thesis_test.csv", index=False)


if __name__ == "__main__":
    main()
