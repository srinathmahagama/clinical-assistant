from preprocess import preprocess_text
from classifier import classify_symptoms

text = input("Enter patient symptoms: ")
words = preprocess_text(text)
detected = classify_symptoms(words)
print("Detected symptoms:", detected)
