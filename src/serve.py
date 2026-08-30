"""Flask inference API for the CIFAR-10 classifier.

Loads a trained checkpoint on startup and exposes:
  - ``GET /health``: liveness/readiness check.
  - ``POST /predict``: multipart image upload -> class prediction.

Configuration is read from environment variables so the same image can
serve different checkpoints/architectures without code changes:
  - ``MODEL_CHECKPOINT_PATH`` (default ``/app/checkpoints/classifier_v1.pt``)
  - ``MODEL_ARCHITECTURE`` (default ``resnet18``)
  - ``NUM_CLASSES`` (default ``10``)
"""

import io
import os
from typing import Optional

import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError

from dataset import CIFAR10_CLASSES, get_transforms
from model import get_model

MODEL_CHECKPOINT_PATH = os.environ.get(
    "MODEL_CHECKPOINT_PATH", "/app/checkpoints/classifier_v1.pt"
)
MODEL_ARCHITECTURE = os.environ.get("MODEL_ARCHITECTURE", "resnet18")
NUM_CLASSES = int(os.environ.get("NUM_CLASSES", "10"))

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model: Optional[torch.nn.Module] = None
inference_transform = get_transforms(train=False)


def load_model(checkpoint_path: str) -> Optional[torch.nn.Module]:
    """Build the model architecture and load weights from checkpoint_path.

    Returns None (and logs a warning) if the checkpoint cannot be
    loaded, so the app can still start and report an unhealthy status
    via ``/health`` instead of crashing.
    """
    try:
        built_model = get_model(
            architecture=MODEL_ARCHITECTURE, num_classes=NUM_CLASSES
        )
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )
        built_model.load_state_dict(state_dict)
        built_model.to(device)
        built_model.eval()
        return built_model
    except (FileNotFoundError, RuntimeError, KeyError, OSError) as exc:
        app.logger.warning("Failed to load model checkpoint '%s': %s", checkpoint_path, exc)
        return None


model = load_model(MODEL_CHECKPOINT_PATH)


@app.route("/health", methods=["GET"])
def health():
    """Report readiness based on whether the model was loaded."""
    if model is not None:
        return jsonify({"status": "ok"}), 200
    return jsonify({"status": "unavailable"}), 503


@app.route("/predict", methods=["POST"])
def predict():
    """Run inference on an uploaded image and return class predictions."""
    if model is None:
        return jsonify({"error": "model not loaded"}), 503

    if "image" not in request.files:
        return jsonify({"error": "missing 'image' file field"}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "empty 'image' file"}), 400

    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "invalid or unreadable image"}), 400

    try:
        input_tensor = inference_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = F.softmax(logits, dim=1).squeeze(0)
            predicted_class = int(torch.argmax(probabilities).item())
    except Exception as exc:  # noqa: BLE001 - surface as a clean 400/500 response
        return jsonify({"error": f"inference failed: {exc}"}), 400

    predicted_label = (
        CIFAR10_CLASSES[predicted_class]
        if predicted_class < len(CIFAR10_CLASSES)
        else str(predicted_class)
    )

    return (
        jsonify(
            {
                "predicted_class": predicted_class,
                "predicted_label": predicted_label,
                "probabilities": [round(p, 6) for p in probabilities.tolist()],
            }
        ),
        200,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
