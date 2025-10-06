# apps/nlpService/testing/test_api_json.py

import requests
import json

BASE_URL = "http://127.0.0.1:8000"  # Your FastAPI server URL

def test_single_text():
    print("=== Testing Single Text ===")
    payload = {
        "text": "Ngaitj koort kalyakal",
        "language": "noongar"
    }
    print("Single Text Request:", json.dumps(payload, indent=2))
    
    response = requests.post(f"{BASE_URL}/analyze", json=payload)
    print("Status Code:", response.status_code)
    
    try:
        data = response.json()
        print("Response:", json.dumps(data, indent=2))
        
        # Print readable summary
        print("\n--- Clinical Summary ---")
        print(f"Text: {data['text']}")
        print(f"Entities Found: {data['entity_count']}")
        print(f"English Interpretation: {data['english_interpretation']}")
        print(f"Has Negation: {data['clinical_summary']['has_negation']}")
        print(f"Symptoms: {len(data['clinical_summary']['symptoms'])}")
        print(f"Body Parts: {len(data['clinical_summary']['body_parts'])}")
        
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
        print("Batch Response Summary:")
        
        # Print readable summaries for each batch item
        for idx, item in enumerate(data.get("results", []), 1):
            print(f"\n--- Result {idx} ---")
            print(f"Original Text: {item['text']}")
            print(f"Success: {item['success']}")
            print(f"Entities Found: {item['entity_count']}")
            print(f"Method Used: {item.get('method_used', 'N/A')}")
            print(f"English Interpretation: {item['english_interpretation']}")
            
            # Print entities in readable format
            print("Entities:")
            for ent in item["entities"]:
                english = ent.get('english_translation', '')
                english_text = f" ({english})" if english else ""
                print(f"  📍 {ent['word']} -> {ent['entity']}{english_text} (confidence: {ent['confidence']:.2f})")
            
            # Print clinical summary
            cs = item["clinical_summary"]
            print(f"Clinical Summary:")
            print(f"  • Body Parts: {[bp['word'] for bp in cs['body_parts']]}")
            print(f"  • Symptoms: {[s['word'] for s in cs['symptoms']]}")
            print(f"  • Negations: {[n['word'] for n in cs['negations']]}")
            print(f"  • Has Negation: {cs['has_negation']}")
                
    except Exception as e:
        print("Error parsing response:", e)

def test_health():
    print("\n=== Testing Health Endpoint ===")
    response = requests.get(f"{BASE_URL}/health")
    print("Status Code:", response.status_code)
    print("Response:", json.dumps(response.json(), indent=2))

def test_root():
    print("\n=== Testing Root Endpoint ===")
    response = requests.get(f"{BASE_URL}/")
    print("Status Code:", response.status_code)
    print("Response:", json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    test_health()
    test_root()
    test_single_text()
    test_batch_texts()