from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import GridSearchCV
import pandas as pd
import pathlib
import joblib


def main():
    srcDataPath = pathlib.Path("data/processed_data/")
    modelPath = pathlib.Path("models/")
    X_train = pd.read_csv(srcDataPath / "X_train_scaled.csv")
    y_train = pd.read_csv(srcDataPath / "Y_train.csv")

    params = {
        "kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
        "C": [0.1, 0, 5, 1, 2, 5, 10, 20, 40, 80, 100],
    }

    params = {
        "alpha_1": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        "alpha_2": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        "lambda_1": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        "lambda_2": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
    }

    X = X_train.drop(columns=["date"])
    y = y_train.to_numpy().ravel()

    estimator = BayesianRidge()
    gridSearch = GridSearchCV(
        estimator=estimator, param_grid=params, scoring="neg_mean_squared_error"
    )
    gridSearch.fit(X, y)

    joblib.dump(gridSearch.best_params_, modelPath / "best_params.pkl")


if __name__ == "__main__":
    main()
