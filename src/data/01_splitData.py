from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
import pathlib


def main():
    srcDataPath = "data/raw_data/raw.csv"
    destDataPath = pathlib.Path("data/processed_data/")
    targetColumn = "silica_concentrate"

    df = pd.read_csv(srcDataPath)

    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1)
    idx_train, idx_test = next(splitter.split(df, groups=df[targetColumn]))

    df_train = df.iloc[idx_train]
    df_test = df.iloc[idx_test]

    X_train = df_train.drop(columns=[targetColumn])
    Y_train = df_train[targetColumn]

    X_test = df_test.drop(columns=["silica_concentrate"])
    Y_test = df_test["silica_concentrate"]

    X_train.to_csv(destDataPath / "X_train.csv", sep=",", index=False)
    Y_train.to_csv(destDataPath / "Y_train.csv", sep=",", index=False)

    X_test.to_csv(destDataPath / "X_test.csv", sep=",", index=False)
    Y_test.to_csv(destDataPath / "Y_test.csv", sep=",", index=False)


if __name__ == "__main__":
    main()
