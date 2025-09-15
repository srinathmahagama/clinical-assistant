import json
import random
from copy import deepcopy

# Absolute paths to the files
input_file = r"C:\Users\imtha\Documents\Semester 3\TRP\clinical-assitant\clinical-assistant\apps\nlpService\data_generating_script\noongar_clinical_dataset_500.json"
output_file = r"C:\Users\imtha\Documents\Semester 3\TRP\clinical-assitant\clinical-assistant\apps\nlpService\data_generating_script\noongar_clinical_dataset_5000.json"

# Load the original dataset
with open(input_file, 'r', encoding='utf-8') as f:
    original_data = json.load(f)

# Vocabulary lists
body_parts = ['koort', 'mooly', 'miyal', 'waarngk', 'kaat', 'djen', 'ngoorndiny', 'korbol', 'woort']
symptoms = ['kalyakal', 'moorditj', 'yoowart', 'wara', 'moorn', 'nyidiny']
qualifiers = ['boola', 'kwop']
negations = ['kadak', 'kwop yoowart']

# Common patterns observed in the original data
patterns = [
    lambda: f"Ngaitj {random.choice(body_parts)} {random.choice(symptoms)}",
    lambda: f"Ngaitj {random.choice(body_parts)} {random.choice(qualifiers)} {random.choice(symptoms)}",
    lambda: f"Ngaitj {random.choice(body_parts)} {random.choice(negations)}",
    lambda: f"{random.choice(body_parts)} ngaitj {random.choice(symptoms)}",
    lambda: f"{random.choice(body_parts)} ngaitj {random.choice(qualifiers)} {random.choice(symptoms)}",
    lambda: f"Ngaitj {random.choice(symptoms)}",
    lambda: f"Ngaitj {random.choice(qualifiers)} {random.choice(symptoms)}",
    lambda: f"Ngaitj {random.choice(body_parts)} {random.choice(symptoms)} {random.choice(qualifiers)} {random.choice(symptoms)}",
]

# Translations mapping
symptom_translations = {
    'kalyakal': 'tired',
    'moorditj': 'strong/severe',
    'yoowart': 'fever',
    'wara': 'bad/sick',
    'moorn': 'sick/unwell',
    'nyidiny': 'cold'
}

body_part_translations = {
    'koort': 'heart',
    'mooly': 'nose',
    'miyal': 'eye',
    'waarngk': 'mouth',
    'kaat': 'head',
    'djen': 'foot',
    'ngoorndiny': 'ear',
    'korbol': 'stomach',
    'woort': 'throat'
}

qualifier_translations = {
    'boola': 'very',
    'kwop': 'good/well'
}

negation_translations = {
    'kadak': 'no/not',
    'kwop yoowart': 'no fever / not good health'
}

# Function to generate translations
def generate_translation(text):
    words = text.split()
    translation_parts = []
    i = 0
    while i < len(words):
        if i < len(words)-1 and f"{words[i]} {words[i+1]}" in negation_translations:
            translation_parts.append(negation_translations[f"{words[i]} {words[i+1]}"])
            i += 2
            continue
        word = words[i]
        if word in body_part_translations:
            translation_parts.append(body_part_translations[word])
        elif word in symptom_translations:
            translation_parts.append(symptom_translations[word])
        elif word in qualifier_translations:
            translation_parts.append(qualifier_translations[word])
        elif word in negation_translations:
            translation_parts.append(negation_translations[word])
        elif word == 'ngaitj':
            translation_parts.append('my')
        else:
            translation_parts.append(word)
        i += 1
    return ' '.join(translation_parts)

# Function to generate entity labels
def generate_entities(text):
    entities = []

    # Handle multi-word negations first
    if 'kwop yoowart' in text:
        start = text.find('kwop yoowart')
        entities.append({'start': start, 'end': start + len('kwop yoowart'), 'label': 'NEGATION'})
    if 'kadak' in text:
        start = text.find('kadak')
        entities.append({'start': start, 'end': start + len('kadak'), 'label': 'NEGATION'})

    # Single words
    words = text.split()
    start_pos = 0
    for word in words:
        end_pos = start_pos + len(word)
        if word in body_parts:
            entities.append({'start': start_pos, 'end': end_pos, 'label': 'BODY_PART'})
        elif word in symptoms:
            entities.append({'start': start_pos, 'end': end_pos, 'label': 'SYMPTOM'})
        elif word in qualifiers:
            entities.append({'start': start_pos, 'end': end_pos, 'label': 'QUALITY'})
        start_pos = end_pos + 1  # +1 for space
    return entities

# Generate new data
new_data = deepcopy(original_data)

for i in range(501, 5001):
    pattern = random.choice(patterns)
    text = pattern()
    translation = generate_translation(text)
    entities = generate_entities(text)
    
    new_data.append({
        'id': i,
        'text': text,
        'translation': translation,
        'entities': entities
    })

# Save the expanded dataset
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(new_data, f, indent=2, ensure_ascii=False)

print("Expanded dataset with 5000 entries created successfully!")
