"""
FastAPI Cat vs Dog classifier (model: Firebleu/feral-ai, MobileNetV2)

Setup:
    pip install -r requirements.txt

Folder me ye 3 files honi chahiye: main.py, index.html, requirements.txt

Run:
    uvicorn main:app --reload

Use:
    Frontend  -> http://127.0.0.1:8000
    API docs  -> http://127.0.0.1:8000/docs
"""

import io
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

# ---- Settings -------------------------------------------------------------
REPO_ID = "Firebleu/feral-ai"
FILENAME = "model.pth"

# Agar results ulte aayein to ["dog", "cat"] kar do.
CLASS_NAMES = ["cat", "dog"]

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

# Standard MobileNetV2 preprocessing. Zarurat ho to source code se match karo:
# https://codeberg.org/Firebleudark/feral-ai
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
# ---------------------------------------------------------------------------

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Server start hote hi model ek baar load hoga
    print("Loading model...")
    model_path = hf_hub_download(REPO_ID, FILENAME)
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(1280, 2)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    state["model"] = model
    print("Model ready.")
    yield
    state.clear()


app = FastAPI(
    title="Cat vs Dog Classifier API",
    description="Image upload karo aur pata karo ki cat hai ya dog.",
    lifespan=lifespan,
)


# CORS: zarurat tab padti hai jab index.html seedha double-click karke khola jaye
# ya kisi aur port/server se serve ho. Production me allow_origins ko apni site tak limit karo.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

INDEX_FILE = Path(__file__).parent / "index.html"


@app.get("/", include_in_schema=False)
def home():
    if not INDEX_FILE.exists():
        raise HTTPException(status_code=404, detail="index.html, main.py ke saath wale folder me rakho.")
    return FileResponse(INDEX_FILE)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Sirf JPG, PNG, WEBP ya BMP image upload karo.",
        )

    data = file.file.read()
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Image 5 MB se badi hai.")

    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Ye valid image file nahi hai.")

    x = TRANSFORM(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(state["model"](x), dim=1)[0]

    idx = int(probs.argmax())
    return {
        "filename": file.filename,
        "prediction": CLASS_NAMES[idx],
        "confidence": round(float(probs[idx]) * 100, 2),
        "probabilities": {
            name: round(float(p) * 100, 2) for name, p in zip(CLASS_NAMES, probs)
        },
    }