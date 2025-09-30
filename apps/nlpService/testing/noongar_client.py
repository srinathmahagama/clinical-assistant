#!/usr/bin/env python3
"""
Python client library for Noongar Clinical NER API
"""

import requests
from typing import List, Dict, Optional

class NoongarClinicalClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def analyze_text(self, text: str, language: str = "noongar") -> Optional[Dict]:
        """
        Analyze a single Noongar clinical text
        """
        payload = {
            "text": text,
            "language": language
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/analyze",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed: {e}")
            return None
    
    def analyze_batch(self, texts: List[str], language: str = "noongar") -> Optional[Dict]:
        """
        Analyze multiple Noongar clinical texts
        """
        payload = [
            {"text": text, "language": language}
            for text in texts
        ]
        
        try:
            response = requests.post(
                f"{self.base_url}/analyze-batch",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed: {e}")
            return None
    
    def health_check(self) -> bool:
        """Check if API is healthy"""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False

# Usage example
def main():
    client = NoongarClinicalClient()
    
    if client.health_check():
        print("✅ API is healthy")
        
        # Single analysis
        result = client.analyze_text("Ngaitj koort kalyakal")
        if result:
            print(f"📝 {result['original_text']}")
            print(f"🏥 {result['clinical_summary']}")
        
        # Batch analysis
        texts = [
            "Ngaitj koort kalyakal",
            "Ngaitj miyal yoowart kadak",
            "Ngaitj kaat wara"
        ]
        
        batch_result = client.analyze_batch(texts)
        if batch_result:
            print(f"\n📊 Processed {batch_result['processed_count']} texts")
            
    else:
        print("❌ API is not available")

if __name__ == "__main__":
    main()