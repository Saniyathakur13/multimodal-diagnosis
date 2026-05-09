import pandas as pd
import numpy as np
import os
from PIL import Image, ImageDraw
import random

def create_realistic_medical_images():
    """
    Creates REALISTIC synthetic medical images with disease-specific patterns
    """
    
    # Create directories
    diseases = ['normal', 'pneumonia', 'covid', 'tuberculosis']
    for disease in diseases:
        os.makedirs(f'data/train/{disease}', exist_ok=True)
    
    for disease in diseases:
        for i in range(200):  # 200 images per disease
            # Create a realistic-looking medical image
            img = Image.new('RGB', (224, 224), color='black')
            draw = ImageDraw.Draw(img)
            
            # Add disease-specific patterns
            if disease == 'pneumonia':
                # White patches (opacity) in lungs area
                for _ in range(random.randint(10, 30)):
                    x = random.randint(60, 160)
                    y = random.randint(80, 180)
                    size = random.randint(10, 40)
                    intensity = random.randint(180, 255)
                    draw.rectangle([x, y, x+size, y+size], fill=(intensity, intensity, intensity))
                    
            elif disease == 'covid':
                # Ground-glass opacity pattern (diffuse white patches)
                for _ in range(random.randint(20, 50)):
                    x = random.randint(40, 180)
                    y = random.randint(60, 200)
                    size = random.randint(15, 35)
                    intensity = random.randint(160, 230)
                    draw.ellipse([x, y, x+size, y+size], fill=(intensity, intensity, intensity))
                    
            elif disease == 'tuberculosis':
                # Cavitary lesions (dark spots with white rings)
                for _ in range(random.randint(5, 15)):
                    x = random.randint(50, 170)
                    y = random.randint(70, 190)
                    size = random.randint(8, 25)
                    # Dark center
                    draw.ellipse([x, y, x+size, y+size], fill=(50, 50, 50))
                    # White ring
                    draw.ellipse([x-3, y-3, x+size+3, y+size+3], outline=(255, 255, 255), width=2)
                    
            else:  # normal
                # Clear lungs - just some normal structure lines
                for _ in range(random.randint(5, 15)):
                    x = random.randint(50, 170)
                    y = random.randint(70, 190)
                    draw.line([x, y, x+random.randint(10, 30), y+random.randint(-10, 10)], 
                             fill=(100, 100, 100), width=2)
            
            img.save(f'data/train/{disease}/{disease}_{i}.jpg')
    
    print(f"✅ Created {200*len(diseases)} realistic medical images")

def create_symptom_disease_mapping():
    """
    Creates REAL symptom-disease mapping with proper patterns
    """
    
    # Define disease-specific symptom patterns
    disease_patterns = {
        'pneumonia': {
            'keywords': ['fever', 'cough', 'mucus', 'phlegm', 'chest pain', 'shortness of breath', 'chills', 'sweating'],
            'temperature_range': (101, 104),
            'oxygen_range': (88, 94),
            'cough_type': 'productive',
            'duration_days': (3, 10)
        },
        'covid': {
            'keywords': ['fever', 'dry cough', 'loss of taste', 'loss of smell', 'fatigue', 'body aches', 'headache'],
            'temperature_range': (100, 103),
            'oxygen_range': (90, 96),
            'cough_type': 'dry',
            'duration_days': (2, 14)
        },
        'tuberculosis': {
            'keywords': ['chronic cough', 'blood in sputum', 'night sweats', 'weight loss', 'fatigue', 'chest pain'],
            'temperature_range': (99, 101),
            'oxygen_range': (92, 98),
            'cough_type': 'chronic',
            'duration_days': (14, 60)
        },
        'bronchitis': {
            'keywords': ['cough', 'mucus', 'wheezing', 'chest discomfort', 'fatigue', 'mild fever'],
            'temperature_range': (99, 101),
            'oxygen_range': (94, 98),
            'cough_type': 'wet',
            'duration_days': (3, 10)
        },
        'normal': {
            'keywords': ['no symptoms', 'healthy', 'routine checkup', 'no fever', 'no cough'],
            'temperature_range': (97, 99),
            'oxygen_range': (96, 100),
            'cough_type': 'none',
            'duration_days': (0, 0)
        }
    }
    
    data = []
    
    for disease, pattern in disease_patterns.items():
        for i in range(500):  # 500 samples per disease
            # Generate symptoms text
            selected_keywords = random.sample(pattern['keywords'], min(3, len(pattern['keywords'])))
            temperature = random.uniform(*pattern['temperature_range'])
            oxygen = random.uniform(*pattern['oxygen_range'])
            duration = random.randint(*pattern['duration_days'])
            
            symptom_text = f"Patient presents with {', '.join(selected_keywords)}. "
            symptom_text += f"Temperature: {temperature:.1f}°F. "
            symptom_text += f"Oxygen saturation: {oxygen:.0f}%. "
            symptom_text += f"Symptoms lasting for {duration} days. "
            
            if pattern['cough_type'] != 'none':
                symptom_text += f"Cough is {pattern['cough_type']}. "
            
            # Generate 20 features
            features = {
                'fever_duration': duration if 'fever' in symptom_text else random.uniform(0, 2),
                'cough_type': {'productive': 2, 'dry': 1, 'wet': 1, 'chronic': 1.5, 'none': 0}[pattern['cough_type']],
                'breathing_difficulty': random.uniform(3, 9) if 'breathing' in symptom_text else random.uniform(0, 3),
                'chest_pain': 1 if 'chest' in symptom_text else 0,
                'fatigue_level': random.uniform(5, 9) if 'fatigue' in symptom_text else random.uniform(0, 4),
                'oxygen_saturation': oxygen,
                'heart_rate': random.uniform(70, 110) if temperature > 100 else random.uniform(60, 85),
                'blood_pressure': random.uniform(110, 140),
                'age_risk': random.uniform(0.2, 0.8),
                'comorbidity_index': random.uniform(0, 0.5),
                'opacity_density': 0.7 if disease == 'pneumonia' else (0.8 if disease == 'covid' else (0.3 if disease == 'tuberculosis' else 0.1)),
                'consolidation_pattern': 0.6 if disease == 'pneumonia' else (0.2 if disease == 'normal' else 0.4),
                'nodule_size': random.uniform(1, 4) if disease == 'tuberculosis' else random.uniform(0, 1),
                'pleural_effusion': 1 if disease == 'pneumonia' else 0,
                'lung_symmetry': 0.9 if disease == 'normal' else random.uniform(0.4, 0.7),
                'bronchial_thickening': 0.7 if disease in ['pneumonia', 'bronchitis'] else random.uniform(0.1, 0.3),
                'ground_glass_opacity': 0.8 if disease == 'covid' else random.uniform(0, 0.2),
                'calcification': 0.1 if disease == 'tuberculosis' else 0,
                'lymph_node': random.uniform(0.2, 0.5) if disease == 'tuberculosis' else random.uniform(0, 0.2),
                'vascular_anomaly': 0
            }
            
            data.append({
                'image_path': f"{disease}_{i}.jpg",
                'symptoms_description': symptom_text,
                'disease': disease,
                **features
            })
    
    df = pd.DataFrame(data)
    df.to_csv('data/symptoms.csv', index=False)
    print(f"✅ Created {len(df)} symptom-disease mappings")
    return df

if __name__ == "__main__":
    os.makedirs('data/train', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    print("Creating realistic training data...")
    create_realistic_medical_images()
    create_symptom_disease_mapping()
    print("\n✅ Training data ready!")