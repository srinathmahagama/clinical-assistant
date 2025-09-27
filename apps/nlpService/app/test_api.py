import requests

# API URL (adjust port if needed)
BASE_URL = "http://127.0.0.1:8000"

# Paragraph of Noongar sentences
paragraph = """
Ngaitj koort kalyakal. Ngaitj miyal yoowart kadak. Ngaitj kaat wara. 
Koort moorditj. Mooly nyidiny. Woort moorn. Kaat boola kwop.
"""

# Split paragraph into sentences for batch processing
sentences = [s.strip() for s in paragraph.split('.') if s.strip()]

# Prepare batch request payload
batch_payload = [{"text": sentence, "language": "noongar"} for sentence in sentences]

# Send batch request
response = requests.post(f"{BASE_URL}/analyze-batch", json=batch_payload)

# Print response
try:
    print("=== Batch Response ===")
    print(response.status_code)
    print(response.json())
except Exception as e:
    print(f"Error decoding JSON: {e}")
    print(response.text)
