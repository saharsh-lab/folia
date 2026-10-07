"""
Flask backend for the Plant Disease Prediction web app.

Endpoints:
  GET  /           → serve the frontend
  POST /predict    → accept an image upload, return JSON prediction
  GET  /classes    → return all 38 class labels
"""

import os
import json
import io
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

# ── Optional: only import TF when model is available ────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(PROJECT_ROOT, "plant_disease_model.keras"))
CLASS_JSON = os.environ.get("CLASS_JSON", os.path.join(PROJECT_ROOT, "class_indices.json"))

app = Flask(__name__, static_folder="static", template_folder="static")
CORS(app)

# ── Load model & class indices ───────────────────────────────────────────────
print("Loading class indices …")
with open(CLASS_JSON) as f:
    raw = json.load(f)
    class_indices = {int(k): v for k, v in raw.items()}

model = None
if os.path.exists(MODEL_PATH):
    import tensorflow as tf
    print("Loading model …")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded ✓")
else:
    print(f"[WARNING] Model file not found at {MODEL_PATH}. "
          "Run `python train.py` first to train & save the model.")


# ── Disease info lookup (treatment / description) ────────────────────────────
DISEASE_INFO = {
    "healthy": {
        "status": "healthy",
        "description": "This plant appears healthy with no signs of disease.",
        "treatment": "Continue regular watering, fertilisation, and monitoring.",
        "severity": "none",
    },
    "Apple_scab": {
        "status": "diseased",
        "description": "Apple Scab is caused by the fungus Venturia inaequalis. It creates dark, scabby lesions on leaves and fruit.",
        "treatment": "Apply fungicides (captan, myclobutanil). Remove and destroy infected leaves. Ensure good air circulation.",
        "severity": "moderate",
    },
    "Black_rot": {
        "status": "diseased",
        "description": "Black Rot is a fungal disease causing brown spots with purple borders on leaves and rotting fruit.",
        "treatment": "Prune infected branches. Apply copper-based fungicides. Remove mummified fruit from trees.",
        "severity": "high",
    },
    "Cedar_apple_rust": {
        "status": "diseased",
        "description": "Cedar Apple Rust causes bright orange spots on leaves, reducing photosynthesis.",
        "treatment": "Apply fungicides at bud break. Remove nearby cedar/juniper trees if possible.",
        "severity": "moderate",
    },
    "Powdery_mildew": {
        "status": "diseased",
        "description": "Powdery Mildew appears as white powdery patches on leaf surfaces caused by fungal infection.",
        "treatment": "Apply sulphur-based fungicides. Improve air circulation. Avoid overhead irrigation.",
        "severity": "moderate",
    },
    "Early_blight": {
        "status": "diseased",
        "description": "Early Blight causes dark concentric ring spots on lower leaves, spreading upward.",
        "treatment": "Apply chlorothalonil or copper fungicides. Remove infected lower leaves. Rotate crops yearly.",
        "severity": "moderate",
    },
    "Late_blight": {
        "status": "diseased",
        "description": "Late Blight is a serious disease caused by Phytophthora infestans, causing water-soaked lesions.",
        "treatment": "Apply fungicides immediately. Destroy infected plants. Avoid overhead watering.",
        "severity": "critical",
    },
    "Bacterial_spot": {
        "status": "diseased",
        "description": "Bacterial Spot causes small, water-soaked spots that turn dark brown with yellow halos.",
        "treatment": "Apply copper-based bactericides. Avoid working with plants when wet. Use resistant varieties.",
        "severity": "high",
    },
    "Haunglongbing": {
        "status": "diseased",
        "description": "Citrus Greening (HLB) is a devastating bacterial disease spread by psyllid insects.",
        "treatment": "No cure exists. Remove and destroy infected trees. Control psyllid insect populations.",
        "severity": "critical",
    },
    "default": {
        "status": "diseased",
        "description": "A plant disease has been detected. Immediate attention is recommended.",
        "treatment": "Consult a local agricultural extension service for specific treatment recommendations.",
        "severity": "moderate",
    },
}


def get_disease_info(class_name: str) -> dict:
    """Match a class label to its info entry."""
    if "healthy" in class_name.lower():
        return DISEASE_INFO["healthy"]
    for key in DISEASE_INFO:
        if key != "default" and key.lower().replace("_", " ") in class_name.lower().replace("_", " "):
            return DISEASE_INFO[key]
    return DISEASE_INFO["default"]


# ── Preprocessing (mirrors train.py) ─────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Colour 224x224, raw 0-255 floats — the model rescales internally."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224), Image.BILINEAR)
    arr = np.array(img, dtype="float32")
    return np.expand_dims(arr, axis=0)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/classes")
def classes():
    return jsonify({"classes": list(class_indices.values())})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    image_bytes = file.read()

    if model is None:
        # Demo mode: return a fake result so the UI is testable without a saved model
        demo_class = "Potato___Early_blight"
        info = get_disease_info(demo_class)
        return jsonify({
            "demo_mode": True,
            "predicted_class": demo_class,
            "plant": "Potato",
            "disease": "Early Blight",
            "confidence": 0.87,
            **info,
        })

    try:
        arr = preprocess_image(image_bytes)
    except Exception:
        return jsonify({"error": "Could not read image"}), 400
    preds = model.predict(arr, verbose=0)
    idx = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))
    class_name = class_indices[idx]

    # Parse "Plant___Disease" format
    parts = class_name.split("___")
    plant = parts[0].replace("_", " ") if len(parts) > 0 else class_name
    disease = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"

    info = get_disease_info(class_name)

    # Top-5 predictions
    top5_idx = np.argsort(preds[0])[::-1][:5]
    top5 = [
        {"class": class_indices[int(i)], "confidence": float(preds[0][i])}
        for i in top5_idx
    ]

    return jsonify({
        "predicted_class": class_name,
        "plant": plant,
        "disease": disease,
        "confidence": confidence,
        "top5": top5,
        **info,
    })


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", port=5050)
