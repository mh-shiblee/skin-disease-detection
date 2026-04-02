# 🔬 Skin Disease Detection System

An AI-powered skin disease detection system with LLM medical advisory.

## 📊 Model Performance

- **Architecture**: EfficientNet-B4 (Transfer Learning)
- **Dataset**: Skin Disease Dataset (27,153 images, 10 classes)
- **Test Accuracy**: 85.49%
- **Test Macro F1**: 0.8056
- **Test Weighted F1**: 0.8558

## 🏗️ System Architecture

User uploads image
↓
FastAPI Backend
↓
EfficientNet-B4 → Disease prediction + Confidence
↓
Ollama (gemma3:4b) → Medical recommendations
↓
Streamlit Frontend → Results displayed

## 🦠 Detectable Conditions

| Class | Disease                                                |
| ----- | ------------------------------------------------------ |
| 1     | Eczema                                                 |
| 2     | Warts Molluscum and other Viral Infections             |
| 3     | Melanoma                                               |
| 4     | Atopic Dermatitis                                      |
| 5     | Basal Cell Carcinoma (BCC)                             |
| 6     | Melanocytic Nevi (NV)                                  |
| 7     | Benign Keratosis-like Lesions (BKL)                    |
| 8     | Psoriasis pictures Lichen Planus and related diseases  |
| 9     | Seborrheic Keratoses and other Benign Tumors           |
| 10    | Tinea Ringworm Candidiasis and other Fungal Infections |

## 🛠️ Technical Stack

- **Backend**: Python, FastAPI
- **Frontend**: Streamlit
- **ML Model**: EfficientNet-B4 (PyTorch, timm)
- **LLM**: Ollama (gemma3:4b) - runs locally
- **XAI**: LIME (Local Interpretable Model Agnostic Explanations)
- **Deployment**: Docker, Docker Compose

## 📁 Project Structure

skin-disease-detection/
├── model/
│ └── model.py # EfficientNet-B4 class definition
├── backend/
│ └── main.py # FastAPI application
├── frontend/
│ └── app.py # Streamlit UI
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md

## 🚀 Quick Start

### Option 1 - Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/yourusername/skin-disease-detection.git
cd skin-disease-detection

# Add your model weights
# Place skin_efficientnet_b4.pth in model/ folder

# Run everything
docker-compose up --build

Access:

Frontend : http://localhost:8501
Backend : http://localhost:8000
API Docs : http://localhost:8000/docs

# Clone repository
git clone https://github.com/yourusername/skin-disease-detection.git
cd skin-disease-detection

# Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install and start Ollama
# Download from ollama.com
ollama pull gemma3:4b

# Start backend (Terminal 1)
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Start frontend (Terminal 2)
streamlit run frontend/app.py

📡 API Endpoints
POST /analyze_skin
Upload a skin image and get disease prediction with LLM recommendations.

Request:

Content-Type: multipart/form-data
file: image file (jpg, jpeg, png, bmp, webp)

Response:

{
  "disease": "Eczema",
  "confidence": 0.92,
  "top3": [
    {"disease": "Eczema", "confidence": 0.92},
    {"disease": "Atopic Dermatitis", "confidence": 0.05},
    {"disease": "Psoriasis", "confidence": 0.03}
  ],
  "recommendations": "...",
  "next_steps": "...",
  "tips": "..."
}

GET /health
Check system health status.

GET /classes
Get all supported disease classes.

🧠 Training Details

Dataset Split: 70% Train, 15% Val, 15% Test (Stratified)
Augmentation: RandomFlip, RandomRotation, ColorJitter
Optimizer: Adam
Loss: CrossEntropy with class weights
Phase 1: 5 epochs frozen backbone
Phase 2: 30 epochs full fine-tuning
Early Stopping: Patience 10 on Macro F1
⚠️ Medical Disclaimer
This system is for educational purposes only.
It is not intended to replace professional medical advice.
Always consult a qualified dermatologist for proper diagnosis.
---
```
