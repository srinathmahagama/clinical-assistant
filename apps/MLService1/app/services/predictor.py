import joblib
import numpy as np
from app.utils.severity import get_severity
from app.schemas.symptom_input import SymptomInput

# Load trained artifacts once at startup
model = joblib.load("app/models/rf_model.pkl")
label_encoder = joblib.load("app/models/label_encoder.pkl")
feature_names = joblib.load("app/models/features.pkl")

def make_prediction(input_data: SymptomInput):
    # Build feature vector, assign 0 for missing symptoms
    feature_vector = [input_data.symptoms.get(feat, 0) for feat in feature_names]
    feature_vector = np.array(feature_vector).reshape(1, -1)

    # Predict probabilities
    probabilities = model.predict_proba(feature_vector)[0]
    top_indices = np.argsort(probabilities)[::-1][:3]

    top_predictions = [
        {
            "disease": label_encoder.inverse_transform([i])[0],
            "confidence": f"{round(probabilities[i] * 100, 2)}%"
        }
        for i in top_indices
    ]

    best_prediction = top_predictions[0]
    severity = get_severity(input_data.symptoms)

    return {
        "best_prediction": best_prediction,
        "top_3_predictions": top_predictions,
        "severity": severity
    }
