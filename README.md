# Skin Disease Detection System

An AI-powered skin disease detection system with LLM medical advisory.

---

## Model Performance

| Model                          | Accuracy | Macro F1 | Weighted F1 |
| ------------------------------ | -------- | -------- | ----------- |
| EfficientNet-B4 (CrossEntropy) | 85.49%   | 0.8056   | 0.8558      |
| SwinV2-T (CrossEntropy)        | 89.05%   | 0.8461   | 0.8910      |
| SwinV2-T (FocalLoss)           | 89.76%   | 0.8562   | 0.8981      |

- **Final Architecture**: SwinV2-T (swinv2_tiny_window8_256)
- **Dataset**: Skin Disease Dataset (27,153 images, 10 classes)
- **Best Test Accuracy**: 89.76%
- **Best Macro F1**: 0.8562

---

## System Architecture

User uploads image
|
FastAPI Backend
|
SwinV2-T (swinv2_tiny_window8_256)
|
Disease prediction + Confidence score
|
Ollama (gemma3:4b) - Medical recommendations
|
Streamlit Frontend - Results displayed

## Detectable Conditions

| Index | Disease                                                |
| ----- | ------------------------------------------------------ |
| 0     | Eczema                                                 |
| 1     | Warts Molluscum and other Viral Infections             |
| 2     | Melanoma                                               |
| 3     | Atopic Dermatitis                                      |
| 4     | Basal Cell Carcinoma (BCC)                             |
| 5     | Melanocytic Nevi (NV)                                  |
| 6     | Benign Keratosis-like Lesions (BKL)                    |
| 7     | Psoriasis pictures Lichen Planus and related diseases  |
| 8     | Seborrheic Keratoses and other Benign Tumors           |
| 9     | Tinea Ringworm Candidiasis and other Fungal Infections |

---

## Technical Stack

| Component  | Technology                        |
| ---------- | --------------------------------- |
| Backend    | Python, FastAPI                   |
| Frontend   | Streamlit                         |
| ML Model   | SwinV2-T (PyTorch, timm)          |
| LLM        | Ollama (gemma3:4b) - runs locally |
| Deployment | Docker, Docker Compose            |

## Project Structure

skin-disease-detection/
├── model/
│ ├── model.py # SkinClassifier class definition
│ └── swinv2_skin.pth # Trained model weights
├── backend/
│ └── main.py # FastAPI application
├── frontend/
│ └── app.py # Streamlit UI
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── start.py
└── README.md

## Quick Start

### Docker

```bash
# Clone repository
git clone https://github.com/yourusername/skin-disease-detection.git
cd skin-disease-detection

# Place model weights in model/ folder
# swinv2_skin.pth

# Run everything
docker-compose up --build

Access:

Frontend : http://localhost:8501
Backend : http://localhost:8000
API Docs : http://localhost:8000/docs
Ollama : http://localhost:11434

```

Local Setup

# Clone repository

git clone https://github.com/yourusername/skin-disease-detection.git
cd skin-disease-detection

# Create virtual environment

python -m venv venv
venv\Scripts\activate # Windows
source venv/bin/activate # Linux/Mac

# Install dependencies

pip install -r requirements.txt

# Install and start Ollama

# Download from ollama.com

ollama pull gemma3:4b

# Start backend (Terminal 1)

uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Start frontend (Terminal 2)

streamlit run frontend/app.py

API Endpoints
POST /analyze_skin
Upload a skin image and get disease prediction with LLM recommendations.

Request:
Content-Type : multipart/form-data
file : image file (jpg, jpeg, png, bmp, webp)

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
Check system health and model loading status.

GET /classes
Get all supported disease classes.

Training Details
Parameter Value
Dataset Split 70% Train, 15% Val, 15% Test (Stratified)
Input Size 256x256
Batch Size 32
Optimizer Adam
Phase 1 LR 1e-3 (frozen backbone, 5 epochs)
Phase 2 LR 1e-4 (full fine-tuning, 30 epochs)
Early Stopping Patience 10 on Macro F1
Loss Function Focal Loss + Clinical Cost Matrix
Augmentation RandomResizedCrop, RandomFlip, RandomRotation, ColorJitter
Imbalance Handling Class Weights + WeightedRandomSampler
Label Smoothing 0.1
Mixup Alpha 0.2, Probability 0.5

Future Work
XAI - Explainability
Implement Attention Rollout for SwinV2-T
to visualize which image regions
the model focuses on during prediction
Add attention heatmap overlay in Streamlit UI
Compare attention maps across different disease classes

Model Improvements
Ensemble of SwinV2-T and EfficientNet-B4
for potentially higher accuracy
Two stage classifier for inflammatory
conditions (Eczema, Atopic Dermatitis, Psoriasis)
Test Time Augmentation (TTA) for
more robust predictions

System Improvements
Add patient history tracking with database
Add DICOM image support for clinical use
REST API authentication and rate limiting
Mobile friendly UI
Multi language support for recommendations
Clinical Validation
Validate model with real dermatologist feedback
Add confidence calibration
Add uncertainty estimation

Medical Disclaimer
This system is for educational purposes only.
It is not intended to replace professional medical advice, diagnosis or treatment.
Always consult a qualified dermatologist for proper diagnosis and treatment.
