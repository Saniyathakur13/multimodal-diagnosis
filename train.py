import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import random
from tqdm import tqdm
import json

from model import LightweightMedicalModel, extract_text_features

class SimpleBalancedDataset(Dataset):
    def __init__(self, samples_per_disease=5):
        self.diseases = ['normal', 'pneumonia', 'covid', 'tuberculosis', 'bronchitis']
        self.samples_per_disease = samples_per_disease
        
        # UNIQUE symptoms for each disease - NO OVERLAP
        self.symptoms_db = {
            'normal': [
                "No symptoms, feeling healthy, routine checkup",
                "Patient has no fever, no cough, normal breathing",
                "Routine physical exam, no complaints",
                "Asymptomatic, vitals normal"
            ],
            'pneumonia': [
                "High fever 103F, productive cough with green mucus, chest pain",
                "Fever 102F, coughing up yellow phlegm, shortness of breath",
                "Bacterial pneumonia: fever, productive cough, crackles in lungs",
                "Fever 104F, rust colored sputum, pleuritic chest pain"
            ],
            'covid': [
                "Dry cough, loss of taste and smell, fever 101F, fatigue",
                "Dry persistent cough, no mucus, loss of smell, body aches",
                "COVID positive: dry hacking cough, anosmia, fever",
                "Dry cough, fever 100.5F, extreme fatigue, no phlegm"
            ],
            'tuberculosis': [
                "Cough for 6 weeks, coughing up blood, night sweats, weight loss",
                "Blood in sputum, night sweats, chronic cough, weight loss 15lbs",
                "Hemoptysis, night sweats, fever evenings, cough >3 weeks",
                "TB symptoms: blood streaks in sputum, night sweats, cachexia"
            ],
            'bronchitis': [
                "Cough with clear mucus, wheezing, chest tightness, no high fever",
                "Productive cough with white phlegm, wheezing, mild fever",
                "Clear mucus production, wheezing upon expiration, chest discomfort",
                "Bronchitis: cough with clear sputum, wheezing, normal oxygen"
            ]
        }
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.data = []
        self.create_data()
    
    def create_data(self):
        for disease in self.diseases:
            for i in range(self.samples_per_disease):
                # Pick random symptom template
                symptom = random.choice(self.symptoms_db[disease])
                
                # Add random variations
                temp = random.choice([98.6, 99.5, 100.5, 101.5, 102.5, 103.5])
                oxygen = random.choice([99, 98, 97, 96, 95, 94, 93, 92, 91, 90])
                
                if disease == 'normal':
                    temp = random.choice([97.5, 98.0, 98.6])
                    oxygen = random.choice([98, 99, 100])
                
                symptoms_text = f"{symptom} Temperature {temp}F. Oxygen {oxygen}%."
                
                # Create image with disease pattern
                img = self.create_image(disease)
                
                # Create features
                features = self.create_features(disease, oxygen)
                
                self.data.append({
                    'image': img,
                    'symptoms': symptoms_text,
                    'disease': disease,
                    'features': features,
                    'label': self.diseases.index(disease)
                })
        
        random.shuffle(self.data)
        print(f"✅ Created {len(self.data)} samples ({self.samples_per_disease} per disease)")
    
    def create_image(self, disease):
        img = np.random.randint(0, 50, (224, 224, 3), dtype=np.uint8)
        
        if disease == 'pneumonia':
            # White patches
            for _ in range(25):
                x, y = random.randint(50, 150), random.randint(60, 160)
                img[y:y+40, x:x+40] = [220, 220, 220]
        elif disease == 'covid':
            # Diffuse white areas
            for _ in range(35):
                x, y = random.randint(40, 160), random.randint(50, 170)
                img[y:y+30, x:x+30] = [200, 200, 200]
        elif disease == 'tuberculosis':
            # Dark spots with white rings
            for _ in range(15):
                x, y = random.randint(60, 140), random.randint(70, 150)
                img[y:y+25, x:x+25] = [30, 30, 30]
                img[y-3:y+28, x-3:x+28] = [250, 250, 250]
        elif disease == 'bronchitis':
            # Lines
            for _ in range(20):
                x, y = random.randint(50, 160), random.randint(60, 160)
                img[y:y+3, x:x+40] = [150, 150, 150]
        else:  # normal
            for _ in range(8):
                x, y = random.randint(60, 150), random.randint(70, 150)
                img[y:y+2, x:x+30] = [80, 80, 80]
        
        return Image.fromarray(img)
    
    def create_features(self, disease, oxygen):
        if disease == 'normal':
            return [0, 0, 0, 0, 1, oxygen, 72, 118, 0.2, 0.1, 0.1, 0.1, 0, 0, 0.9, 0.1, 0, 0, 0, 0]
        elif disease == 'pneumonia':
            return [5, 2, 7, 1, 8, oxygen, 100, 125, 0.5, 0.3, 0.8, 0.7, 2, 0.8, 0.4, 0.7, 0.2, 0.1, 0.3, 0.1]
        elif disease == 'covid':
            return [3, 1, 6, 0, 8, oxygen, 95, 120, 0.6, 0.4, 0.7, 0.4, 1, 0.1, 0.5, 0.3, 0.8, 0.1, 0.2, 0.1]
        elif disease == 'tuberculosis':
            return [30, 1.5, 5, 1, 9, oxygen, 85, 115, 0.7, 0.5, 0.3, 0.4, 3, 0.2, 0.6, 0.4, 0.2, 0.2, 0.6, 0.1]
        else:  # bronchitis
            return [4, 1, 4, 0, 6, oxygen, 90, 125, 0.4, 0.2, 0.2, 0.3, 0, 0, 0.8, 0.6, 0.1, 0.1, 0.1, 0.1]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = self.transform(item['image'])
        text_features = torch.tensor(extract_text_features(item['symptoms']), dtype=torch.float32)
        label = torch.tensor(item['label'], dtype=torch.long)
        features = torch.tensor(item['features'], dtype=torch.float32)
        return {'image': image, 'text_features': text_features, 'label': label, 'features': features}

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using: {device}")
    
    # Create balanced dataset
    dataset = SimpleBalancedDataset(samples_per_disease=500)
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    # Model
    model = LightweightMedicalModel(num_diseases=5)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    
    best_acc = 0
    
    for epoch in range(20):
        # Train
        model.train()
        correct = 0
        total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/20"):
            images = batch['image'].to(device)
            text = batch['text_features'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            logits, _ = model(images, text)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            _, pred = torch.max(logits, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
        
        train_acc = 100 * correct / total
        
        # Validate
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                text = batch['text_features'].to(device)
                labels = batch['label'].to(device)
                
                logits, _ = model(images, text)
                _, pred = torch.max(logits, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
        
        val_acc = 100 * correct / total
        
        print(f"Epoch {epoch+1}: Train Acc = {train_acc:.1f}%, Val Acc = {val_acc:.1f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'models/best_model.pth')
            print(f"✅ Saved! Accuracy: {best_acc:.1f}%")
    
    # Save mapping
    with open('models/class_mapping.json', 'w') as f:
        json.dump({'normal':0, 'pneumonia':1, 'covid':2, 'tuberculosis':3, 'bronchitis':4}, f)
    
    print(f"\n🎉 Done! Best accuracy: {best_acc:.1f}%")

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    train()