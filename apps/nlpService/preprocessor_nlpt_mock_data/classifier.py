symptoms_dict = {
    'fever': ['fever', 'temperature', 'hot'],
    'headache': ['headache', 'migraine', 'pain in head'],
    'cough': ['cough', 'throat irritation']
}

def classify_symptoms(words):
    detected = []
    for symptom, keywords in symptoms_dict.items():
        if any(word in words for word in keywords):
            detected.append(symptom)
    return detected
