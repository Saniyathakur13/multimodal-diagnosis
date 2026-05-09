import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import os

class MedicalMultimodalDataset(Dataset):
    def __init__(self, image_dir, symptoms_csv, tokenizer, max_length=128, transform=None):
        self.image_dir = image_dir
        self.symptoms_df = pd.read_csv(symptoms_csv)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Image transformations (with augmentation)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Define diseases (4 classes)
        self.disease_map = {
            'normal': 0,
            'pneumonia': 1,
            'covid': 2,
            'tuberculosis': 3
        }
        
        # 20 features mapping
        self.feature_names = [
            'fever_duration', 'cough_type', 'breathing_difficulty', 'chest_pain', 
            'fatigue_level', 'oxygen_saturation', 'heart_rate', 'blood_pressure',
            'age_risk', 'comorbidity_index', 'opacity_density', 'consolidation_pattern',
            'nodule_size', 'pleural_effusion', 'lung_symmetry', 'bronchial_thickening',
            'ground_glass_opacity', 'calcification', 'lymph_node', 'vascular_anomaly'
        ]
        
    def __len__(self):
        return len(self.symptoms_df)
    
    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.image_dir, self.symptoms_df.iloc[idx]['image_path'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # Process symptoms text
        symptoms_text = self.symptoms_df.iloc[idx]['symptoms_description']
        encoded = self.tokenizer(
            symptoms_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get label
        label = self.disease_map[self.symptoms_df.iloc[idx]['disease']]
        
        # Get 20 features (for training feature extractor)
        # Convert to numeric first, then to a float32 numpy array, then to a tensor
        features_data = pd.to_numeric(self.symptoms_df.iloc[idx][self.feature_names], errors='coerce').fillna(0).values
        features_20 = torch.tensor(features_data, dtype=torch.float32)

        
        return {
            'image': image,
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'features_20': features_20
        }

# Function to create synthetic dataset for demonstration
def create_synthetic_dataset():
    """Create a small synthetic dataset for testing"""
    import random
    
    diseases = ['normal', 'pneumonia', 'covid', 'tuberculosis']
    symptoms_templates = [
        "Patient has fever of {fever}°C for {days} days, {cough} cough, and {difficulty} breathing difficulty",
        "Reports {fatigue} fatigue, oxygen saturation {oxygen}%, heart rate {hr} BPM",
        "Chest {pain} pain present, blood pressure {bp}, {comorbidities} comorbidities",
        "Age {age} years, shows {symptom1} and {symptom2} with {severity} severity"
    ]
    
    data = []
    for i in range(500):  # 500 samples
        disease = random.choice(diseases)
        
        # Generate 20 features
        features = {
            'fever_duration': random.uniform(0, 10),
            'cough_type': random.choice([0, 1, 2]),  # 0:dry, 1:wet, 2:productive
            'breathing_difficulty': random.uniform(1, 10),
            'chest_pain': random.choice([0, 1]),
            'fatigue_level': random.uniform(1, 10),
            'oxygen_saturation': random.uniform(85, 100),
            'heart_rate': random.uniform(60, 120),
            'blood_pressure': random.uniform(90, 160),
            'age_risk': random.uniform(0, 1),
            'comorbidity_index': random.uniform(0, 1),
            'opacity_density': random.uniform(0, 1),
            'consolidation_pattern': random.uniform(0, 1),
            'nodule_size': random.uniform(0, 5),
            'pleural_effusion': random.choice([0, 1]),
            'lung_symmetry': random.uniform(0, 1),
            'bronchial_thickening': random.uniform(0, 1),
            'ground_glass_opacity': random.uniform(0, 1),
            'calcification': random.uniform(0, 1),
            'lymph_node': random.uniform(0, 1),
            'vascular_anomaly': random.uniform(0, 1)
        }
        
        # Generate symptoms text
        text = f"Disease: {disease}. " + symptoms_templates[0].format(
            fever=round(features['fever_duration'], 1),
            days=random.randint(1, 14),
            cough=['dry', 'wet', 'productive'][int(features['cough_type'])],
            difficulty=round(features['breathing_difficulty'], 1)
        )
        
        data.append({
            'image_path': f"{disease}_{i}.jpg",
            'symptoms_description': text,
            'disease': disease,
            **features
        })
    
    df = pd.DataFrame(data)
    df.to_csv('data/symptoms.csv', index=False)
    print(f"Created synthetic dataset with {len(df)} samples")
    return df

# Create dummy images (for testing without real dataset)
def create_dummy_images():
    import numpy as np
    from PIL import Image
    
    os.makedirs('data/train/normal', exist_ok=True)
    os.makedirs('data/train/pneumonia', exist_ok=True)
    os.makedirs('data/train/covid', exist_ok=True)
    os.makedirs('data/train/tuberculosis', exist_ok=True)
    
    # TO THIS:
os.makedirs('data/train', exist_ok=True) # Ensure the main folder exists
for disease in ['normal', 'pneumonia', 'covid', 'tuberculosis']:
    for i in range(500): # Match your 500 samples in create_synthetic_dataset
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        # Save directly to data/train so the path matches symptoms.csv
        img.save(f'data/train/{disease}_{i}.jpg')

    print("Created dummy images for testing")