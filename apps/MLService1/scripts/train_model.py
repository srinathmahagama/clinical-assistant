
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def main():
    # Load dataset
    data = pd.read_csv("../data/Training.csv")

    # Features and labels
    X = data.drop(columns=["prognosis"])
    y = data["prognosis"]

    # Encode target labels
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save artifacts inside app/models/
    os.makedirs("../app/models", exist_ok=True)
    joblib.dump(model, "../app/models/rf_model.pkl")
    joblib.dump(le, "../app/models/label_encoder.pkl")
    joblib.dump(list(X.columns), "../app/models/features.pkl")

    print("✅ Model trained and saved inside app/models/")


if __name__ == "__main__":
    main()
