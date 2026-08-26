import os
import pickle
import pandas as pd
import yaml

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

params = yaml.safe_load(open("params.yaml"))
tp = params["train"]
seed = params["split"]["seed"]

IN = "data/split/train.csv"
OUT = "models"

def build_model():
    kind = tp["model_type"]
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=tp["n_estimators"],
            max_depth=tp["max_depth"],
            min_samples_leaf=tp["min_samples_leaf"],
            random_state=seed
        )
    elif kind == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=tp["n_estimators"],
            max_depth=tp["max_depth"],
            learning_rate=tp["learning_rate"],
            random_state=seed
        )
    elif kind == "logistic_regression":
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=tp["C"], max_iter=1000, random_state=seed)),
        ])

    raise ValueError(f"unknown model_type: {kind}")

def main() -> None:
    df = pd.read_csv(IN)
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    model = build_model()
    model.fit(X, y)

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/model.pkl", "wb") as f:
        pickle.dump({"model": model, "features": list(X.columns)}, f)
    print(f"trained {tp['model_type']} on {len(X)} rows")

if __name__ == "__main__":
    main()