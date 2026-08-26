import os
import pandas as pd
import yaml

from sklearn.model_selection import train_test_split

params = yaml.safe_load(open("params.yaml"))["split"]

IN = "data/prepared/clean.csv"
OUT = "data/split"

def main() -> None:
    df = pd.read_csv(IN)
    train, test = train_test_split(
        df,
        test_size=params["test_size"],
        random_state=params["seed"],
        stratify=df["Survived"],
    )

    os.makedirs(OUT, exist_ok=True)
    train.to_csv(f"{OUT}/train.csv")
    test.to_csv(f"{OUT}/test.csv")

    print(f"train={len(train)} test={len(test)}")

if __name__ == "__main__":
    main()