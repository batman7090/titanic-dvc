import json
import os
import pickle
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    roc_curve, confusion_matrix,
)

MODEL = "models/model.pkl"
TEST = "data/split/test.csv"


def main() -> None:
    with open(MODEL, "rb") as f:
        bundle = pickle.load(f)
    model, features = bundle["model"], bundle["features"]

    df = pd.read_csv(TEST)
    X, y = df[features], df["Survived"]

    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred)), 4),
        "recall": round(float(recall_score(y, pred)), 4),
        "f1": round(float(f1_score(y, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
    }

    os.makedirs("evaluation", exist_ok=True)
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ROC curve — DVC renders this as a linear plot
    fpr, tpr, _ = roc_curve(y, proba)
    with open("evaluation/roc.json", "w") as f:
        json.dump([{"fpr": float(a), "tpr": float(b)} for a, b in zip(fpr, tpr)], f)

    # Confusion matrix — DVC renders this from actual/predicted pairs
    with open("evaluation/confusion.json", "w") as f:
        json.dump(
            [{"actual": str(a), "predicted": str(p)} for a, p in zip(y, pred)], f
        )


    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    if hasattr(clf, "feature_importances_"):
        vals = clf.feature_importances_
    else:
        vals = abs(clf.coef_[0])
        vals = vals / vals.sum()
    imp = sorted(zip(features, vals), key=lambda t: -t[1])
    with open("evaluation/importance.json", "w") as f:
        json.dump(
            [{"feature": n, "importance": round(float(v), 4)} for n, v in imp], f
        )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
