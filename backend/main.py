from model.model import SkinClassifier
from dotenv import load_dotenv
import requests as http_requests
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
import io
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


load_dotenv()

# ============================================================
# Config
# ============================================================

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "model", "skin_efficientnet_b4.pth")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
# ============================================================
# Global Variables
# ============================================================

model = None
class_names = None
transform = None
gemini_model = None

# ============================================================
# Load Model
# ============================================================


def load_model():
    global model, class_names, transform

    print(f"Loading model from {MODEL_PATH}...")

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    class_names = ckpt['class_names']
    input_size = ckpt['input_size']
    mean = ckpt['mean']
    std = ckpt['std']
    num_classes = ckpt['num_classes']

    model = SkinClassifier(num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    print(f"✅ Model loaded successfully")
    print(f"   Classes    : {class_names}")
    print(f"   Input size : {input_size}")
    print(f"   Device     : {DEVICE}")


# ============================================================
# Load LLM
# ============================================================

def load_llm():
    global gemini_model

    try:
        response = http_requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            gemini_model = True  # just a flag to indicate ollama is running
            print("Ollama LLM loaded successfully")
        else:
            print("Ollama not responding")
            gemini_model = None
    except Exception as e:
        print(f"Ollama connection failed: {e}")
        gemini_model = None


# ============================================================
# Lifespan
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    load_llm()
    yield
    print("Shutting down...")


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="Skin Disease Detection API",
    description="AI powered skin disease detection with LLM recommendations",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Inference Helper
# ============================================================

def predict_image(image: Image.Image) -> dict:
    """Run model inference on image."""
    if image.mode != 'RGB':
        image = image.convert('RGB')

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)

    probs_np = probs.cpu().numpy()[0]
    predicted_idx = int(np.argmax(probs_np))
    confidence = float(probs_np[predicted_idx])
    disease = class_names[predicted_idx]

    # Top 3 predictions
    top3_indices = np.argsort(probs_np)[::-1][:3]
    top3 = [
        {
            "disease": class_names[i],
            "confidence": round(float(probs_np[i]), 4)
        }
        for i in top3_indices
    ]

    return {
        "disease": disease,
        "confidence": round(confidence, 4),
        "top3": top3,
        "predicted_idx": predicted_idx
    }


# ============================================================
# LLM Helper
# ============================================================

def get_llm_advice(disease: str, confidence: float) -> dict:

    if gemini_model is None:
        return {
            "recommendations": "LLM not available",
            "next_steps": "Please consult a dermatologist",
            "tips": "Keep the affected area clean and moisturized"
        }

    prompt = f"""You are a dermatology assistant.
Skin condition detected: {disease} (confidence: {confidence * 100:.1f}%)

Respond in this exact format only:
RECOMMENDATIONS: <2 sentence explanation and advice>
NEXT_STEPS: <2 actionable steps>
TIPS: <2 daily care tips>

Always recommend consulting a dermatologist.
"""

    try:
        response = http_requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": "gemma3:4b",
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        if response.status_code == 200:
            text = response.json().get("response", "")
            print("=== OLLAMA RAW RESPONSE ===")
            print(text)
            print("===========================")

            recommendations = ""
            next_steps = ""
            tips = ""

            for line in text.split('\n'):
                line = line.strip()
                if line.startswith('RECOMMENDATIONS:'):
                    recommendations = line.replace(
                        'RECOMMENDATIONS:', '').strip()
                elif line.startswith('NEXT_STEPS:'):
                    next_steps = line.replace('NEXT_STEPS:', '').strip()
                elif line.startswith('TIPS:'):
                    tips = line.replace('TIPS:', '').strip()

            # fallback if parsing fails
            if not recommendations:
                recommendations = text[:200]
            if not next_steps:
                next_steps = "Please consult a dermatologist"
            if not tips:
                tips = "Keep the affected area clean and moisturized"

            return {
                "recommendations": recommendations,
                "next_steps": next_steps,
                "tips": tips
            }

        else:
            raise Exception(f"Ollama returned status {response.status_code}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"LLM error: {e}")
        return {
            "recommendations": "Unable to generate recommendations",
            "next_steps": "Please consult a dermatologist",
            "tips": "Keep the affected area clean and moisturized"
        }


# ============================================================
# Routes
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Skin Disease Detection API",
        "status": "running",
        "model_loaded": model is not None,
        "llm_loaded": gemini_model is not None,
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "llm_loaded": gemini_model is not None,
        "device": str(DEVICE),
        "classes": class_names
    }


@app.post("/analyze_skin")
async def analyze_skin(file: UploadFile = File(...)):
    """
    Main endpoint.
    Upload a skin image and get disease prediction + LLM advice.
    """

    # Validate file
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image. Got: {file.content_type}"
        )

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Run inference
        prediction = predict_image(image)

        # Get LLM advice
        advice = get_llm_advice(
            disease=prediction['disease'],
            confidence=prediction['confidence']
        )

        # Build response
        return {
            "disease": prediction['disease'],
            "confidence": prediction['confidence'],
            "top3": prediction['top3'],
            "recommendations": advice['recommendations'],
            "next_steps": advice['next_steps'],
            "tips": advice['tips']
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/classes")
def get_classes():
    """Get all supported disease classes."""
    return {
        "classes": class_names,
        "total": len(class_names)
    }
