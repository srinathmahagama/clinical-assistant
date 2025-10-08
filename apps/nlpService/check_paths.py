# check_paths.py
from pathlib import Path
import os

BASE_DIR = Path(__file__).parent
print(f"Current directory: {BASE_DIR}")

# Check different possible model locations
possible_paths = [
    BASE_DIR / "models" / "noongar-clinical-ner-model-finetuned",
    BASE_DIR / "models" / "noongar-clinical-ner-model-final", 
    BASE_DIR.parent / "models" / "noongar-clinical-ner-model-finetuned",
    BASE_DIR.parent / "models" / "noongar-clinical-ner-model-final",
]

for path in possible_paths:
    exists = os.path.exists(str(path))
    print(f"Model path: {path} -> {'EXISTS' if exists else 'NOT FOUND'}")
    if exists:
        # List contents
        items = list(path.glob("*"))
        print(f"  Contents: {[item.name for item in items]}")