# apps/nlpService/testing/test_api_json.py

import requests
import json

BASE_URL = "http://127.0.0.1:8000"  # Your FastAPI server URL

def test_single_text():
    print("=== Testing Single Text ===")
    payload = {
        "text": "Ngaitj koort kalyakal",
        "language": "noongar",
        "return_tokens": False
    }
    print("Single Text Request:", json.dumps(payload, indent=2))
    
    response = requests.post(f"{BASE_URL}/analyze", json=payload)
    print("Status Code:", response.status_code)
    
    try:
        data = response.json()
        print("Response:", json.dumps(data, indent=2))
    except Exception as e:
        print("Error parsing response:", e)

def test_batch_texts():
    print("\n=== Testing Batch Texts ===")
    payload = [
        {"text": "Ngaitj koort kalyakal", "language": "noongar"},
        {"text": "Ngaitj miyal yoowart kadak", "language": "noongar"},
        {"text": "Ngaitj kaat wara", "language": "noongar"}
    ]
    print("Batch Request:", json.dumps(payload, indent=2))
    
    response = requests.post(f"{BASE_URL}/analyze-batch", json=payload)
    print("Status Code:", response.status_code)
    
    try:
        data = response.json()
        print("Response:", json.dumps(data, indent=2))
        
        # Print readable summaries for each batch item
        for idx, item in enumerate(data.get("results", []), 1):
            print(f"\n--- Result {idx} ---")
            print("Original Text:", item["original_text"])
            print("Clinical Summary:", item["clinical_summary"])
            print("Processing Method:", item["processing_method"])
            print("Confidence:", item["confidence"])
            print("Entities:")
            for ent in item["entities"]:
                source_icon = "🧠" if "ml" in ent["source"] else "📚"
                english = ent.get("english", "")
                english_text = f" ({english})" if english else ""
                print(f"  {source_icon} {ent['word']} -> {ent['entity_group']}{english_text} ({ent['confidence']:.4f})")
                
    except Exception as e:
        print("Error parsing response:", e)

if __name__ == "__main__":
    test_single_text()
    test_batch_texts()
