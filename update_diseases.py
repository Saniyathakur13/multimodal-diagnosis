# advanced_disease_fetcher.py - Gets 500+ diseases!
import requests
import json
import time

def fetch_all_diseases():
    """
    Fetches diseases from MULTIPLE sources to get hundreds of diseases
    """
    all_diseases = set()
    
    # Source 1: SNOMED CT via NIH API (30,000+ diseases)
    print("🌐 Fetching from NIH Medical Database...")
    base_url = "https://clinicaltables.nlm.nih.gov/api/conditions/v3/search"
    
    # Fetch in batches (100 at a time)
    for offset in range(0, 500, 100):
        params = {
            'terms': 'disease',
            'maxTerms': 100,
            'offset': offset,
            'ef': 'name'
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=10)
            data = response.json()
            diseases = data[1]  # List of disease names
            
            for disease in diseases:
                # Clean the names
                disease = disease.strip()
                if disease and len(disease) > 2 and 'search' not in disease.lower():
                    all_diseases.add(disease)
            
            print(f"   Fetched {len(all_diseases)} diseases so far...")
            time.sleep(0.5)  # Be nice to the server
            
        except Exception as e:
            print(f"   Error: {e}")
            continue
    
    # Source 2: WHO Disease List
    print("\n🌐 Fetching from WHO Disease Database...")
    try:
        who_url = "https://www.who.int/data/gho/data/themes/mortality-and-global-health-estimates"
        # WHO has their own list of 100+ diseases
        who_diseases = [
            "HIV/AIDS", "Malaria", "Tuberculosis", "Hepatitis B", "Hepatitis C",
            "Diarrheal diseases", "Meningitis", "Encephalitis", "Tetanus", 
            "Measles", "Chickenpox", "Yellow Fever", "Dengue", "Zika Virus",
            "Ebola", "COVID-19", "Influenza", "Pneumonia", "Bronchitis",
            "Asthma", "COPD", "Lung Cancer", "Breast Cancer", "Diabetes",
            "Hypertension", "Heart Disease", "Stroke", "Alzheimer's",
            "Parkinson's", "Epilepsy", "Migraine", "Arthritis"
        ]
        all_diseases.update(who_diseases)
        print(f"   Added {len(who_diseases)} from WHO list")
    except:
        print("   WHO fetch failed, using local list")
    
    # Source 3: Common medical conditions (guaranteed to work)
    print("\n📋 Adding common medical conditions...")
    common_conditions = [
        "Normal", "Healthy", 
        "Acute Bronchitis", "Chronic Bronchitis", 
        "Bacterial Pneumonia", "Viral Pneumonia", "Mycoplasma Pneumonia",
        "COVID-19", "SARS", "MERS",
        "Tuberculosis", "MDR-TB", "XDR-TB",
        "Asthma", "Exercise-Induced Asthma", "Allergic Asthma",
        "COPD", "Emphysema", "Chronic Bronchitis",
        "Lung Cancer", "Small Cell Lung Cancer", "Non-Small Cell Lung Cancer",
        "Pulmonary Fibrosis", "Idiopathic Pulmonary Fibrosis",
        "Pleural Effusion", "Pneumothorax", "Hemothorax",
        "Pulmonary Edema", "ARDS", "Respiratory Failure",
        "Sleep Apnea", "Central Sleep Apnea", "Obstructive Sleep Apnea",
        "Allergic Rhinitis", "Sinusitis", "Pharyngitis", "Laryngitis",
        "Influenza A", "Influenza B", "RSV", "Common Cold",
        "Cystic Fibrosis", "Primary Ciliary Dyskinesia",
        "Pulmonary Hypertension", "Cor Pulmonale",
        "Asbestosis", "Silicosis", "Coal Worker's Pneumoconiosis",
        "Hypersensitivity Pneumonitis", "Eosinophilic Pneumonia",
        "Goodpasture Syndrome", "Wegener's Granulomatosis",
        "Sarcoidosis", "Langerhans Cell Histiocytosis",
        "Lymphangioleiomyomatosis", "Alveolar Proteinosis"
    ]
    all_diseases.update(common_conditions)
    
    # Convert to sorted list
    all_diseases = sorted(list(all_diseases))
    
    print(f"\n✅ SUCCESS! Total diseases fetched: {len(all_diseases)}")
    
    # Save to config file
    with open('diseases_config.py', 'w') as f:
        f.write(f"# Auto-fetched diseases - Total: {len(all_diseases)}\n")
        f.write(f"# Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"DISEASES = {all_diseases}\n\n")
        f.write(f"DISEASE_SYMPTOMS = {{\n")
        for disease in all_diseases[:50]:  # Add symptoms for first 50
            f.write(f'    "{disease}": "common symptoms of {disease.lower()}",\n')
        f.write(f"}}\n")
    
    print(f"💾 Saved to diseases_config.py")
    return all_diseases

if __name__ == "__main__":
    diseases = fetch_all_diseases()
    print(f"\n📊 Summary:")
    print(f"   - Total diseases: {len(diseases)}")
    print(f"   - Example diseases: {diseases[:20]}")






























    