def get_severity(symptoms_dict):
    count = sum(symptoms_dict.values())
    if count <= 3:
        return "Mild"
    elif 4 <= count <= 7:
        return "Moderate"
    else:
        return "Severe"
