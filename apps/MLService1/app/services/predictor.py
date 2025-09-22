import joblib
import numpy as np
from app.utils.severity import get_severity
from app.schemas.symptom_input import SymptomInput

# Load trained artifacts once at startup
model = joblib.load("app/models/rf_model.pkl")
label_encoder = joblib.load("app/models/label_encoder.pkl")
feature_names = joblib.load("app/models/features.pkl")

def make_prediction(input_data: SymptomInput):
    # Ensure input matches features
    x = [input_data.symptoms.get(feat, 0) for feat in feature_names]
    x = np.array(x).reshape(1, -1)

    # Predict probabilities
    proba = model.predict_proba(x)[0]

    # Get top 3 predictions
    top_indices = np.argsort(proba)[::-1][:3]
    top_diseases = [
        {
            "disease": label_encoder.inverse_transform([i])[0],
            "confidence": f"{round(proba[i]*100, 2)}%"
        }
        for i in top_indices
    ]

    # Best prediction
    best_pred = top_diseases[0]["disease"]
    best_conf = top_diseases[0]["confidence"]

    # Severity
    severity = get_severity(input_data.symptoms)

    return {
        "best_prediction": {"disease": best_pred, "confidence": best_conf},
        "top_3_predictions": top_diseases,
        "severity": severity
    }
