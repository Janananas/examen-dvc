import joblib
from sklearn.linear_model import BayesianRidge
import pathlib
import pandas as pd


def main():
    srcDataPath = pathlib.Path("data/processed_data/")
    modelPath = pathlib.Path("models/")
    X_train = pd.read_csv(srcDataPath / "X_train_scaled.csv")
    y_train = pd.read_csv(srcDataPath / "Y_train.csv")

    params = joblib.load(modelPath / "best_params.pkl")

    X = X_train.drop(columns=["date"])
    y = y_train.to_numpy().ravel()

    model = BayesianRidge(**params)
    model.fit(X, y)

    joblib.dump(model, modelPath / "model.pkl")


if __name__ == "__main__":
    main()
