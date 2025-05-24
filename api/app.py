import os
import io
import base64
import tempfile

import gdown
import numpy as np
from PIL import Image
from ultralytics import YOLO
from flask import Flask, request, jsonify, make_response

app = Flask(__name__)

# Where we’ll store the downloaded model
MODEL_PATH = os.path.join(tempfile.gettempdir(), "best.pt")

def load_model():
    if not os.path.exists(MODEL_PATH):
        url = os.getenv("MODEL_URL")
        if not url:
            raise RuntimeError("MODEL_URL not set")
        # gdown handles Drive confirmation tokens automatically
        gdown.download(url, MODEL_PATH, quiet=False)
    return YOLO(MODEL_PATH)

# cold-start load
model = load_model()

# CORS on every response
@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    return response

@app.route("/", methods=["GET", "OPTIONS"])
def index():
    if request.method == "OPTIONS":
        return make_response("", 200)
    return jsonify({"message": "Server is running!"})

@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze():
    if request.method == "OPTIONS":
        return make_response("", 200)

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img = Image.open(request.files["image"].stream).convert("RGB")
    arr = np.array(img)

    results = model(arr, conf=0.3, iou=0.4)[0]
    annotated = results.plot()
    annotated_rgb = Image.fromarray(annotated[..., ::-1])

    buf = io.BytesIO()
    annotated_rgb.save(buf, format="JPEG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    count = len(results.boxes)
    return jsonify({
        "detection":    count > 0,
        "message":      "Fractures detected." if count else "No fractures detected.",
        "image_base64": img_b64
    })
