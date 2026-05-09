import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import re

class LightweightMedicalModel(nn.Module):
    def __init__(self, num_diseases=5):
        super(LightweightMedicalModel, self).__init__()
        
        # CNN for images
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        
        # Text encoder
        self.text_encoder = nn.Sequential(
            nn.Linear(512, 256),  # Increased from 768 to handle our features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_diseases)
        )
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
            nn.Sigmoid()
        )
        
    def forward(self, images, text_features):
        img_features = self.cnn(images)
        text_feat = self.text_encoder(text_features)
        combined = torch.cat([img_features, text_feat], dim=1)
        disease_logits = self.classifier(combined)
        features_20 = self.feature_extractor(combined)
        return disease_logits, features_20

def extract_text_features(symptoms_text):
    """Enhanced text feature extraction for disease classification"""
    features = np.zeros(512)
    symptoms_lower = symptoms_text.lower()
    
    # Disease-specific keyword scoring (UNIQUE per disease)
    disease_keywords = {
        'pneumonia': ['productive cough', 'green mucus', 'yellow mucus', 'phlegm', 'crackles', 'rust colored', 'pleuritic', 'bacterial pneumonia', 'thick sputum'],
        'covid': ['dry cough', 'loss of taste', 'loss of smell', 'anosmia', 'ageusia', 'dry hacking', 'no mucus', 'no phlegm', 'sars-cov-2'],
        'tuberculosis': ['blood in sputum', 'hemoptysis', 'night sweats', 'weight loss', 'chronic cough weeks', 'evening fever', 'ppd positive', 'cavitary'],
        'bronchitis': ['clear mucus', 'white mucus', 'wheezing', 'clear phlegm', 'bronchial', 'afebrile', 'no blood', 'acute bronchitis'],
        'normal': ['no symptoms', 'healthy', 'routine checkup', 'asymptomatic', 'normal vitals', 'no fever', 'no cough']
    }
    
    # Score each disease
    disease_scores = {}
    for disease, keywords in disease_keywords.items():
        score = sum(2 if kw in symptoms_lower else 0 for kw in keywords)  # Weighted scoring
        disease_scores[disease] = score
    
    # Encode disease scores into feature vector
    for i, (disease, score) in enumerate(disease_scores.items()):
        if score > 0:
            features[i*50:(i+1)*50] = min(1.0, score / 10)
    
    # Extract temperature
    temp_match = re.search(r'(\d{2,3}(?:\.\d+)?)\s*°?\s*f', symptoms_lower)
    if temp_match:
        temp = float(temp_match.group(1))
        if temp > 101:
            features[250:260] = 0.9  # High fever
        elif temp > 100:
            features[250:260] = 0.6
        elif temp > 99:
            features[250:260] = 0.3
    
    # Extract oxygen
    o2_match = re.search(r'oxygen\s*(\d{2})', symptoms_lower)
    if o2_match:
        o2 = int(o2_match.group(1))
        if o2 < 92:
            features[260:270] = 0.8  # Low oxygen
        elif o2 < 95:
            features[260:270] = 0.4
    
    # Extract duration
    duration_match = re.search(r'(\d+)\s*days', symptoms_lower)
    if duration_match:
        days = int(duration_match.group(1))
        if days > 14:
            features[270:280] = 0.9  # Chronic
        elif days > 7:
            features[270:280] = 0.6
    
    return features