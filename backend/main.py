
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base directories
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MODEL_PATH = BASE_DIR / "best.pt"

# Verify required files
if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"YOLO model not found: {MODEL_PATH}"
    )

if not STATIC_DIR.is_dir():
    raise FileNotFoundError(
        f"Static directory not found: {STATIC_DIR}"
    )

if not (STATIC_DIR / "index.html").is_file():
    raise FileNotFoundError(
        f"Homepage not found: {STATIC_DIR / 'index.html'}"
    )

# Load YOLO model
model = YOLO(str(MODEL_PATH))

# Serve frontend static assets
app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static"
)


# Homepage
@app.get("/")
def serve_home():
    return FileResponse(str(STATIC_DIR / "index.html"))


# Other HTML pages
@app.get("/{page}.html")
def serve_page(page: str):
    file = STATIC_DIR / f"{page}.html"

    if file.is_file():
        return FileResponse(str(file))

    return FileResponse(str(STATIC_DIR / "index.html"))


# Health check
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
            detail="Please upload a valid image file."
        )

    # Read image
    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Unable to process the uploaded image."
        )

    # Run YOLO prediction
    try:
        results = model(
            image,
            conf=0.25,
            verbose=False
        )[0]

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    # Extract detections
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

    # Return predictions
    return {
        "boxes": boxes,
        "has_detections": len(boxes) > 0
    }
