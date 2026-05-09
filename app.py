from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import json
import os

app = Flask(__name__)
CORS(app)

from model import LightweightMedicalModel, extract_text_features

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Disease mapping
idx_to_disease = {0: 'Normal', 1: 'Pneumonia', 2: 'COVID-19', 3: 'Tuberculosis', 4: 'Bronchitis'}

# Load model
model = LightweightMedicalModel(num_diseases=5)
if os.path.exists('models/best_model.pth'):
    model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
    model.to(device)
    model.eval()
    print("✅ Model loaded!")
else:
    print("⚠️ Train model first: python train.py")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        symptoms = request.form.get('symptoms', '')
        if not symptoms:
            return jsonify({'error': 'Please provide symptoms'}), 400
        
        # Process image if provided
        image_file = request.files.get('image')
        if image_file and image_file.filename:
            image = Image.open(image_file).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
        else:
            dummy = Image.new('RGB', (224, 224), 'gray')
            image_tensor = transform(dummy).unsqueeze(0).to(device)
        
        # Extract text features
        text_features = torch.tensor(extract_text_features(symptoms), dtype=torch.float32).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            logits, features_20 = model(image_tensor, text_features)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            
            # Sort predictions
            sorted_indices = np.argsort(probs)[::-1]
            
            # Top 3 predictions (sorted by probability)
            top3 = []
            for i in range(3):
                idx = sorted_indices[i]
                top3.append({
                    'disease': idx_to_disease[idx],
                    'confidence': float(probs[idx])
                })
            
            # All predictions
            all_preds = [{'disease': idx_to_disease[i], 'confidence': float(probs[i])} for i in range(5)]
            all_preds.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Features
            feature_names = ['Fever Duration', 'Cough Type', 'Breathing Difficulty', 'Chest Pain',
                           'Fatigue Level', 'Oxygen Saturation', 'Heart Rate', 'Blood Pressure',
                           'Age Risk', 'Comorbidity Index', 'Opacity Density', 'Consolidation',
                           'Nodule Size', 'Pleural Effusion', 'Lung Symmetry', 'Bronchial Thickening',
                           'Ground-Glass Opacity', 'Calcification', 'Lymph Node', 'Vascular Anomaly']
            
            features_dict = {feature_names[i]: float(features_20[0][i]) for i in range(20)}
            
            # Recommendations
            recommendations = {
                "Pneumonia": ["Complete antibiotics course", "Chest X-ray follow-up", "Monitor oxygen", "Rest"],
                "COVID-19": ["Self-isolate", "Monitor oxygen", "Rest", "Contact doctor"],
                "Tuberculosis": ["Start TB medication", "Complete 6-month treatment", "Isolate initially"],
                "Bronchitis": ["Use humidifier", "Avoid irritants", "Rest", "Increase fluids"],
                "Normal": ["Healthy lifestyle", "Regular checkups", "Stay active"]
            }
            
            result = {
                'prediction': idx_to_disease[sorted_indices[0]],
                'confidence': float(probs[sorted_indices[0]]),
                'top3_predictions': top3,
                'all_predictions': all_preds,
                'features_20': features_dict,
                'recommendations': recommendations.get(idx_to_disease[sorted_indices[0]], recommendations['Normal'])
            }
            
            return jsonify(result)
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    print("\n🚀 Server running at http://127.0.0.1:5000")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)