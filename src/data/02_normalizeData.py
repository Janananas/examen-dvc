from sklearn.preprocessing import normalize
import pandas as pd
import pathlib


def main(dataFileSrc: str, dataFileDest: str):
    srcDataPath = pathlib.Path("../../data/processed_data/") / dataFileSrc
    destDataPath = pathlib.Path("../../data/processed_data/")

    df = pd.read_csv(srcDataPath)
    df_noDate = df.drop(columns=["date"])

    df_noDate_norm = pd.DataFrame(normalize(df_noDate))
    df_noDate_norm.columns = df_noDate.columns

    df_norm: pd.DataFrame = pd.concat([df["date"].to_frame(), df_noDate_norm], axis=1)
    df_norm.to_csv(destDataPath / dataFileDest, sep=",", index=False)


if __name__ == "__main__":
    main("X_test.csv", "X_test_scaled.csv")
    main("X_train.csv", "X_train_scaled.csv")
