
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError
from pathlib import Path
import io

# Initialize FastAPI
app = FastAPI(
    title="Blight Detection API",
    description="Plant disease detection using YOLO",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://blight-frontend.vercel.app/"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Locate backend directory
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.pt"

# Verify model exists
if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"YOLO model not found: {MODEL_PATH}"
    )

# Load trained YOLO model
model = YOLO(str(MODEL_PATH))


# Health check
@app.get("/")
def root():
    return {
        "message": "Blight Detection API is running",
        "status": "healthy"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True
    }


# Get model classes
@app.get("/classes")
def get_classes():
    return {
        "classes": model.names
    }


# Image prediction
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Validate uploaded file
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image."
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )

    try:
        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Unable to process the uploaded image."
        )

    # Run YOLO inference
    try:
        results = model(
            image,
            conf=0.25,
            verbose=False
        )[0]

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Image prediction failed."
        )

    # Extract detected objects
    boxes = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0])

        confidence = float(box.conf[0])
        class_id = int(box.cls[0])

        boxes.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "confidence": round(confidence * 100, 1),
            "label": model.names[class_id]
        })

    return {
        "boxes": boxes,
        "has_detections": len(boxes) > 0
    }
