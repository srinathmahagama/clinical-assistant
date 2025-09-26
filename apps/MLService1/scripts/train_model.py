import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

DATA_DIR = "../data"
MODEL_DIR = "../app/models"

def load_data(filename):
    df = pd.read_csv(os.path.join(DATA_DIR, filename))
    df = df.drop(columns=["Unnamed: 133"], errors="ignore")
    return df

def main():
    # Load training data
    train_data = load_data("Training.csv")
    X_train = train_data.drop(columns=["prognosis"])
    y_train = train_data["prognosis"]

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    # Model definition
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Perform 5-fold cross-validation, split into 5 parts, train on 4, validate on 1, repeat 5 times
    cv_scores = cross_val_score(model, X_train, y_train_encoded, cv=5)
    print("Cross-validation scores:", cv_scores)
    print(f"Mean CV Accuracy: {cv_scores.mean():.2f}")

    # Train on full data
    model.fit(X_train, y_train_encoded)

    # External test evaluation
    test_data = load_data("Testing.csv")
    X_test = test_data.drop(columns=["prognosis"])
    y_test = label_encoder.transform(test_data["prognosis"])
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nExternal Test Accuracy: {accuracy:.2f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Save artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "rf_model.pkl"))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    joblib.dump(list(X_train.columns), os.path.join(MODEL_DIR, "features.pkl"))

    print("\nModel trained and saved inside app/models/")

if __name__ == "__main__":
    main()
