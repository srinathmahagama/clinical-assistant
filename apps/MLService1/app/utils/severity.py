def get_severity(symptoms_dict):
    # red flag symptoms indicating severe condition as of WHO guidelines
    red_flag_symptoms = {
        "high_fever", "chest_pain", "breathlessness", "altered_sensorium", "coma",
        "dehydration", "sunken_eyes", "stomach_bleeding", "bloody_stool", "blood_in_sputum",
        "abdominal_pain", "distention_of_abdomen", "slurred_speech", "weakness_of_one_body_side",
        "loss_of_balance", "headache", "stiff_neck", "yellowing_of_eyes", "acute_liver_failure",
        "fast_heart_rate", "palpitations", "weight_loss", "fatigue"
    }


    
    # If any red flag symptom is present → Severe
    for symptom in red_flag_symptoms:
        if symptoms_dict.get(symptom, 0) == 1:
            return "Severe"
    
    # Otherwise use count logic
    count = sum(symptoms_dict.values())
    if count <= 3:
        return "Mild"
    elif 4 <= count <= 7:
        return "Moderate"
    else:
        return "Severe"

