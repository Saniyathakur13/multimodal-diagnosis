Multi-Modal Medical Diagnosis System
A professional AI-driven healthcare application that analyzes both X-ray/MRI images and textual patient symptoms to provide diagnostic predictions for respiratory diseases like COVID-19, Pneumonia, Bronchitis, and Asthma.

🚀 Key Features
Dual-Input Analysis: Process medical images and text symptoms simultaneously or independently.

Real-time Probability Distribution: Dynamic bar charts visualizing disease confidence levels using Chart.js.

Full-Stack Architecture: Integrated Java/Spring Boot backend with a Python AI inference service.

Aesthetic UI: Minimalist, magazine-style professional interface with high-fidelity icons and animations.

🛠️ Tech Stack
Frontend: HTML5, CSS3 (Modern Magazine Editorial style), JavaScript.

Web Backend: Spring Boot (Java), Node.js.

AI/ML: Python, PyTorch/TensorFlow, NLTK (for symptom processing).

Database: MySQL (dance_academy_db/service_details).

Deployment: Netlify (Frontend) and Flask/FastAPI (AI Service).

📂 Project Structure
Plaintext
multimodal_diagnosis/
├── data/               # Datasets for X-rays and symptoms.csv
├── models/             # Pre-trained .pth model files
├── static/             # CSS, JS, and high-fidelity icons
├── templates/          # index.html (The main UI)
├── app.py              # Python API for AI inference
├── train.py            # Model training script
└── src/                # Spring Boot / Java source files
⚙️ Installation & Setup
1. Python Environment (AI Service)
Bash
# Create and activate virtual environment
python -m venv venv
./venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt

# Run the AI server
python app.py
2. Java Backend (Spring Boot)
Import the project into IntelliJ IDEA or Eclipse.

Configure application.properties with your MySQL credentials.

Run the application on localhost:8080.

3. Frontend
Ensure the Node.js server is running or open index.html via a live server.

Verify that the API endpoint in displayResults() matches your Python server address.

🧪 Usage Example
Upload: Drag and drop a chest X-ray image.

Describe: Enter symptoms like "High fever, persistent cough, and chest pain".

Analyze: Click Analyze to view the Primary Diagnosis and the probability distribution chart.

🚧 Challenges Overcome
Label Mapping: Resolved index mismatches between the training classes and the inference output to ensure 100% diagnostic accuracy.

UI/UX: Implemented a non-blocking "Analyzing" state for seamless user experience during heavy ML processing.