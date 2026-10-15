"""
Flask backend for Folia — the plant disease detection web app.

Endpoints:
  GET  /                  → serve the frontend
  POST /predict           → accept an image upload, return JSON prediction
  GET  /classes           → return all 38 class labels
  GET  /model-info        → test metrics written by train.py (metrics.json)
  GET  /samples           → list the sample leaf images
  GET  /samples/<name>    → serve one sample image
"""

import os
import json
import io
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(PROJECT_ROOT, "plant_disease_model.keras"))
CLASS_JSON = os.environ.get("CLASS_JSON", os.path.join(PROJECT_ROOT, "class_indices.json"))
METRICS_JSON = os.path.join(PROJECT_ROOT, "metrics.json")
SAMPLES_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "test_images_cnn"))
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

app = Flask(__name__, static_folder="static", template_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB uploads
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


# ── Disease info lookup (cause / description / treatment) ────────────────────
# Keys are matched against the class name: exact class names first
# (e.g. "Grape___Black_rot"), then disease-name substrings.
DISEASE_INFO = {
    "healthy": {
        "status": "healthy",
        "cause": None,
        "description": "This leaf shows no visible signs of the diseases the model was trained on.",
        "treatment": "Continue regular watering, fertilisation, and monitoring.",
        "severity": "none",
    },
    "Apple_scab": {
        "status": "diseased",
        "cause": "Fungus — Venturia inaequalis",
        "description": "Apple Scab creates olive-green to dark, scabby lesions on leaves and fruit, and can cause early leaf drop.",
        "treatment": "Apply fungicides (captan, myclobutanil) from green tip. Rake and destroy fallen leaves. Prune for good air circulation.",
        "severity": "moderate",
    },
    "Grape___Black_rot": {
        "status": "diseased",
        "cause": "Fungus — Guignardia bidwellii",
        "description": "Grape Black Rot causes tan leaf spots with dark borders and tiny black dots, and shrivels berries into hard black mummies.",
        "treatment": "Remove mummified berries and infected canes. Apply protectant fungicides (mancozeb, myclobutanil) from bud break through early fruit set. Open the canopy for airflow.",
        "severity": "high",
    },
    "Black_rot": {
        "status": "diseased",
        "cause": "Fungus — Botryosphaeria obtusa",
        "description": "Apple Black Rot causes 'frogeye' leaf spots — brown centres with purple borders — plus cankers on limbs and rotting fruit.",
        "treatment": "Prune out dead wood and cankers. Remove mummified fruit. Apply fungicides (captan) during the growing season.",
        "severity": "high",
    },
    "Cedar_apple_rust": {
        "status": "diseased",
        "cause": "Fungus — Gymnosporangium juniperi-virginianae",
        "description": "Cedar Apple Rust causes bright yellow-orange spots on leaves. The fungus alternates between apple and juniper/cedar hosts.",
        "treatment": "Apply fungicides (myclobutanil) from pink bud stage. Remove nearby cedar/juniper galls where possible. Plant resistant varieties.",
        "severity": "moderate",
    },
    "Powdery_mildew": {
        "status": "diseased",
        "cause": "Fungus — powdery mildew species",
        "description": "Powdery Mildew appears as white powdery patches on leaf surfaces and can distort and weaken new growth.",
        "treatment": "Apply sulphur or potassium bicarbonate fungicides. Improve air circulation. Avoid excess nitrogen and overhead irrigation.",
        "severity": "moderate",
    },
    "Cercospora_leaf_spot": {
        "status": "diseased",
        "cause": "Fungus — Cercospora zeae-maydis",
        "description": "Gray Leaf Spot forms long, rectangular grey-to-tan lesions bounded by leaf veins. It thrives in warm, humid weather.",
        "treatment": "Plant resistant hybrids. Rotate crops and manage infected residue. Apply a foliar fungicide if lesions reach upper leaves before grain fill.",
        "severity": "moderate",
    },
    "Common_rust": {
        "status": "diseased",
        "cause": "Fungus — Puccinia sorghi",
        "description": "Common Rust produces small, cinnamon-brown powdery pustules on both sides of the leaf.",
        "treatment": "Plant resistant hybrids. A fungicide is usually only worthwhile if infection is heavy early in the season.",
        "severity": "moderate",
    },
    "Northern_Leaf_Blight": {
        "status": "diseased",
        "cause": "Fungus — Exserohilum turcicum",
        "description": "Northern Leaf Blight causes long, cigar-shaped grey-green to tan lesions that can kill large areas of leaf.",
        "treatment": "Plant resistant hybrids. Rotate crops and bury or remove residue. Apply fungicide around tasseling if disease is present.",
        "severity": "high",
    },
    "Esca": {
        "status": "diseased",
        "cause": "Fungal trunk disease complex (Phaeomoniella, Phaeoacremonium)",
        "description": "Esca (Black Measles) gives leaves a 'tiger-stripe' pattern of yellow or red between veins, with dark spotting on berries. It lives in the vine's wood.",
        "treatment": "There is no cure. Prune in dry weather and protect pruning wounds. Remove and destroy infected wood or badly affected vines.",
        "severity": "high",
    },
    "Leaf_blight": {
        "status": "diseased",
        "cause": "Fungus — Pseudocercospora vitis (Isariopsis)",
        "description": "Isariopsis Leaf Spot forms angular dark-brown spots on grape leaves and can cause early leaf drop late in the season.",
        "treatment": "Remove fallen leaves. Improve canopy airflow. Protectant fungicides used for other grape diseases (e.g. mancozeb, copper) also help.",
        "severity": "moderate",
    },
    "Haunglongbing": {
        "status": "diseased",
        "cause": "Bacterium — Candidatus Liberibacter, spread by psyllids",
        "description": "Citrus Greening (HLB) causes blotchy yellow mottling, lopsided bitter fruit, and eventual tree decline.",
        "treatment": "There is no cure. Remove and destroy infected trees. Control Asian citrus psyllid populations and use certified clean nursery stock.",
        "severity": "critical",
    },
    "Bacterial_spot": {
        "status": "diseased",
        "cause": "Bacterium — Xanthomonas species",
        "description": "Bacterial Spot causes small, water-soaked spots that turn dark brown, often with yellow halos, on leaves and fruit.",
        "treatment": "Apply copper-based bactericides. Avoid working with plants when wet. Use disease-free seed and resistant varieties.",
        "severity": "high",
    },
    "Early_blight": {
        "status": "diseased",
        "cause": "Fungus — Alternaria species",
        "description": "Early Blight causes dark spots with concentric 'target' rings on lower leaves, spreading upward.",
        "treatment": "Apply chlorothalonil or copper fungicides. Remove infected lower leaves. Mulch, and rotate crops yearly.",
        "severity": "moderate",
    },
    "Late_blight": {
        "status": "diseased",
        "cause": "Oomycete — Phytophthora infestans",
        "description": "Late Blight causes large, dark, water-soaked lesions, often with white growth underneath. It spreads very fast in cool, wet weather.",
        "treatment": "Act immediately: apply fungicides (chlorothalonil, mancozeb). Remove and destroy infected plants. Avoid overhead watering.",
        "severity": "critical",
    },
    "Leaf_scorch": {
        "status": "diseased",
        "cause": "Fungus — Diplocarpon earlianum",
        "description": "Strawberry Leaf Scorch starts as small purple spots that merge until the leaf looks burnt.",
        "treatment": "Remove infected leaves and renovate beds after harvest. Plant resistant cultivars. Apply fungicides during flowering if needed.",
        "severity": "moderate",
    },
    "Leaf_Mold": {
        "status": "diseased",
        "cause": "Fungus — Passalora fulva",
        "description": "Leaf Mold forms pale yellow spots on top of leaves with olive-green, velvety mould underneath. Common in humid greenhouses.",
        "treatment": "Lower humidity and increase ventilation. Avoid wetting leaves. Remove infected leaves and use resistant varieties.",
        "severity": "moderate",
    },
    "Septoria_leaf_spot": {
        "status": "diseased",
        "cause": "Fungus — Septoria lycopersici",
        "description": "Septoria Leaf Spot causes many small round spots with dark borders and grey centres dotted with tiny black specks.",
        "treatment": "Remove infected lower leaves. Mulch and avoid overhead watering. Apply chlorothalonil or copper fungicides. Rotate crops.",
        "severity": "moderate",
    },
    "Spider_mites": {
        "status": "diseased",
        "cause": "Pest — two-spotted spider mite (Tetranychus urticae)",
        "description": "Spider mites cause fine yellow stippling, bronzing, and sometimes webbing on leaves. Worst in hot, dry conditions.",
        "treatment": "Hose leaves with water. Apply insecticidal soap or horticultural oil. Encourage predatory mites and avoid broad-spectrum insecticides.",
        "severity": "moderate",
    },
    "Target_Spot": {
        "status": "diseased",
        "cause": "Fungus — Corynespora cassiicola",
        "description": "Target Spot causes brown lesions with concentric rings on leaves, stems, and fruit.",
        "treatment": "Improve airflow and remove plant debris. Apply fungicides (chlorothalonil). Rotate crops.",
        "severity": "moderate",
    },
    "Yellow_Leaf_Curl_Virus": {
        "status": "diseased",
        "cause": "Virus — TYLCV, spread by whiteflies",
        "description": "Tomato Yellow Leaf Curl Virus makes leaves curl upward with yellow edges, and stunts the plant, sharply reducing yield.",
        "treatment": "There is no cure. Remove infected plants. Control whiteflies (insect netting, reflective mulch) and plant resistant varieties.",
        "severity": "critical",
    },
    "mosaic_virus": {
        "status": "diseased",
        "cause": "Virus — Tomato mosaic virus (ToMV)",
        "description": "Tomato Mosaic Virus causes light- and dark-green mottling and distorted leaves. It spreads on hands, tools, and seed.",
        "treatment": "There is no cure. Remove infected plants. Disinfect tools, wash hands, and use certified seed and resistant varieties.",
        "severity": "high",
    },
    "default": {
        "status": "diseased",
        "cause": None,
        "description": "A plant disease has been detected. Immediate attention is recommended.",
        "treatment": "Consult a local agricultural extension service for specific treatment recommendations.",
        "severity": "moderate",
    },
}


def get_disease_info(class_name: str) -> dict:
    """Match a class label to its info entry."""
    if "healthy" in class_name.lower():
        return DISEASE_INFO["healthy"]
    if class_name in DISEASE_INFO:
        return DISEASE_INFO[class_name]
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


@app.route("/model-info")
def model_info():
    metrics = None
    if os.path.exists(METRICS_JSON):
        with open(METRICS_JSON) as f:
            metrics = json.load(f)
    return jsonify({
        "model_loaded": model is not None,
        "architecture": "MobileNetV2 (ImageNet) fine-tuned",
        "input": "224×224 RGB",
        "num_classes": len(class_indices),
        "baseline": {"name": "2-layer CNN, grayscale", "val_accuracy": 0.7954},
        "metrics": metrics,
    })


@app.route("/samples")
def samples():
    if not os.path.isdir(SAMPLES_DIR):
        return jsonify({"samples": []})
    names = sorted(
        n for n in os.listdir(SAMPLES_DIR) if os.path.splitext(n)[1].lower() in ALLOWED_EXTS
    )
    return jsonify({"samples": names})


@app.route("/samples/<path:name>")
def sample_file(name):
    return send_from_directory(SAMPLES_DIR, name)


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "Image is larger than 10 MB"}), 413


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTS:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    if model is None:
        return jsonify({
            "error": "Model not loaded. Run `python train.py` to train it, then restart the app."
        }), 503

    image_bytes = file.read()
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
