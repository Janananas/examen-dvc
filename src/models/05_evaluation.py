import pathlib
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def main():
    srcDataPath = pathlib.Path("../../data/processed_data/")
    modelPath = pathlib.Path("../../models/")
    metricsPath = pathlib.Path("../../metrics/")

    X_test = pd.read_csv(srcDataPath / "X_test_scaled.csv")
    y_test = pd.read_csv(srcDataPath / "Y_test.csv")
    model = joblib.load(modelPath / "model.pkl")

    X = X_test.drop(columns=["date"])
    y_pred = model.predict(X)

    out = {
        "mean_squared_error": mean_squared_error(y_test, y_pred),
        "r2_score": r2_score(y_test, y_pred),
    }

    df = pd.DataFrame.from_dict(out.items())
    df.columns = ["method", "score"]
    df.to_csv(metricsPath / "scores.json", sep=",", index=False)

    df_pred = pd.DataFrame(y_pred)
    df_pred.columns = ["y_pred"]
    df_pred.to_csv(srcDataPath / "predictions.csv", sep=",", index=False)


if __name__ == "__main__":
    main()
