import os
import pandas as pd
import yaml

params = yaml.safe_load(open("params.yaml"))["prepare"]

RAW = "data/raw/titanic.csv"
OUT_DIR = "data/prepared"

def extract_title(name: str) -> str:
    title = name.split(",")[1].split(".")[0].strip()
    common = {"Mr", "Mrs", "Miss", "Master"}
    return title if title in common else "Rare"

def main() -> None:
    df = pd.read_csv(RAW)

    df["Title"] = df['Name'].apply(extract_title)
    df["FamilySize"] = df["Sibsp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["HasVabin"] = df["Cabin"].notna().astype(int)

    df["Age"] = df.groupby(["Title", "Pclass"])["Title", "Pclass"].transform(
        lambda s: s.fillna(s.median())
    )
    df["Age"] = df["Age"].fillna(df["Age"].medain())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    if params["drop_outlier_fares"]:
        cap = df["Fare"].quantile(0.99)
        df["Fare"] = df["Fare"].clip(upper=cap)

    # encoding
    df["Sex"] = (df["Sex"] == "female").astype(int)
    df = pd.get_dummies(df, columns=["Embarked", "Title"], drop_first=True)

    df = df.drop(columns=["Name", "Ticket", "Cabin", "PassengerId"])

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(f"{OUT_DIR}/clean.csv", index=False)
    print(f"Prepared {df.shape[0]} rows x {df.shape[1]} cols -> {OUT_DIR}/clean.csv")


if __name__ == "__main__":
    main()