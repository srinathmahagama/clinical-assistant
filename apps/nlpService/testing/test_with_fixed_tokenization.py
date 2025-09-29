# production_test.py
import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_production_scenarios():
    print("🏥 PRODUCTION SCENARIOS - NOONGAR CLINICAL NER")
    print("=" * 60)
    
    # Common clinical scenarios in Noongar
    clinical_cases = [
        {
            "name": "Cardiac fatigue",
            "text": "Ngaitj koort kalyakal",
            "expected_entities": ["POSSESSIVE", "BODY_PART", "SYMPTOM"]
        },
        {
            "name": "Ocular fever with negation", 
            "text": "Ngaitj miyal yoowart kadak",
            "expected_entities": ["POSSESSIVE", "BODY_PART", "SYMPTOM", "NEGATION"]
        },
        {
            "name": "Cephalic illness",
            "text": "Ngaitj kaat wara", 
            "expected_entities": ["POSSESSIVE", "BODY_PART", "SYMPTOM"]
        },
        {
            "name": "Gastric fatigue with severity",
            "text": "Ngaitj korbol boola kalyakal",
            "expected_entities": ["POSSESSIVE", "BODY_PART", "QUALITY", "SYMPTOM"]
        },
        {
            "name": "Negated cardiac symptoms",
            "text": "Kadak ngaitj koort kalyakal",
            "expected_entities": ["NEGATION", "POSSESSIVE", "BODY_PART", "SYMPTOM"]
        }
    ]
    
    print("\n📋 CLINICAL CASE ANALYSIS:")
    print("-" * 40)
    
    all_passed = True
    
    for case in clinical_cases:
        print(f"\n🔬 Case: {case['name']}")
        print(f"   Noongar: '{case['text']}'")
        
        payload = {"text": case["text"], "language": "noongar"}
        response = requests.post(f"{BASE_URL}/analyze", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract actual entities found
            actual_entities = [entity["entity"] for entity in data["entities"]]
            
            # Check if all expected entities are found
            missing_entities = set(case["expected_entities"]) - set(actual_entities)
            extra_entities = set(actual_entities) - set(case["expected_entities"])
            
            if not missing_entities and not extra_entities:
                print(f"   ✅ PASS - All entities correctly identified")
                print(f"   📊 Entities: {', '.join(actual_entities)}")
            else:
                print(f"   ⚠️  PARTIAL - Entity mismatch")
                if missing_entities:
                    print(f"      Missing: {', '.join(missing_entities)}")
                if extra_entities:
                    print(f"      Extra: {', '.join(extra_entities)}")
                all_passed = False
            
            # Show clinical interpretation
            print(f"   🏥 Interpretation: {data['english_interpretation']}")
            print(f"   📈 Stats: {data['clinical_summary']['symptom_count']} symptoms, "
                  f"{data['clinical_summary']['body_part_count']} body parts, "
                  f"negation: {data['clinical_summary']['has_negation']}")
                  
        else:
            print(f"   ❌ FAIL - API error: {response.status_code}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 EXCELLENT! All clinical cases processed successfully!")
        print("🚀 API is ready for production use!")
    else:
        print("⚠️  Some issues detected. Review the results above.")

def generate_clinical_report():
    """Generate a sample clinical report"""
    print("\n📄 SAMPLE CLINICAL REPORT GENERATION:")
    print("-" * 40)
    
    patient_text = "Ngaitj koort kalyakal miyal yoowart boola"
    
    payload = {"text": patient_text, "language": "noongar"}
    response = requests.post(f"{BASE_URL}/analyze", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        
        print(f"Patient complaint: {patient_text}")
        print(f"Clinical analysis: {data['english_interpretation']}")
        print("\nCLINICAL FINDINGS:")
        print(f"- Affected body parts: {len(data['clinical_summary']['body_parts'])}")
        for bp in data['clinical_summary']['body_parts']:
            print(f"  • {bp['word']} ({bp['translation']})")
        
        print(f"- Symptoms reported: {len(data['clinical_summary']['symptoms'])}")
        for symptom in data['clinical_summary']['symptoms']:
            print(f"  • {symptom['word']} ({symptom['translation']})")
        
        print(f"- Severity qualifiers: {len(data['clinical_summary']['qualifiers'])}")
        for qual in data['clinical_summary']['qualifiers']:
            print(f"  • {qual['word']} ({qual['translation']})")
        
        print(f"- Negation present: {data['clinical_summary']['has_negation']}")

if __name__ == "__main__":
    test_production_scenarios()
    generate_clinical_report()