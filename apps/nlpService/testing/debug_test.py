# debug_test.py
import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def debug_response():
    print("🔍 Debugging API response...")
    
    # Test single analysis with raw response
    payload = {
        "text": "Ngaitj koort kalyakal",
        "language": "noongar"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/analyze", json=payload)
        
        print(f"Status Code: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type')}")
        print(f"Response text (first 500 chars): {response.text[:500]}")
        
        # Try to parse as JSON
        try:
            json_data = response.json()
            print("✅ JSON response:")
            print(json.dumps(json_data, indent=2))
        except:
            print("❌ Response is not valid JSON")
            
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    debug_response()