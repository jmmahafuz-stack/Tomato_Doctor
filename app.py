import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image

try:
    import torch
    from torchvision import models, transforms
except ImportError:
    torch = None
    models = None
    transforms = None

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "tomato_disease_resnet18.pth"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# Update this list to match the classes used when training the checkpoint.
CLASS_NAMES = [
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Septoria Leaf Spot",
    "Tomato Spider Mites",
    "Tomato Target Spot",
    "Tomato Mosaic Virus",
    "Tomato Yellow Leaf Curl Virus",
    "Healthy Tomato Leaf",
]

transform = None
if transforms is not None:
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

model = None
model_error = None


def load_model():
    """Load a ResNet18 checkpoint when it is available."""
    global model, model_error

    if torch is None or models is None:
        model_error = "PyTorch is not installed. Run pip install -r requirements.txt to enable predictions."
        return

    if not MODEL_PATH.exists():
        model_error = "Model checkpoint not found. Add tomato_disease_resnet18.pth to the project root."
        return

    try:
        network = models.resnet18(weights=None)
        network.fc = torch.nn.Linear(network.fc.in_features, len(CLASS_NAMES))
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
        network.load_state_dict(state_dict)
        network.eval()
        model = network
        model_error = None
    except Exception as exc:
        model_error = f"The model could not be loaded: {exc}"


load_model()


def predict(image):
    if model is None or transform is None:
        raise RuntimeError(model_error or "The model is unavailable.")

    tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
        confidence, index = torch.max(probabilities, dim=0)

    return {
        "disease": CLASS_NAMES[index.item()],
        "confidence": round(confidence.item() * 100, 1),
    }


@app.get("/")
def index():
    return render_template("index.html", model_ready=model is not None)


@app.post("/predict")
def predict_route():
    uploaded_file = request.files.get("image")
    if uploaded_file is None or uploaded_file.filename == "":
        return jsonify({"error": "Please choose a tomato leaf image first."}), 400

    try:
        image = Image.open(uploaded_file.stream)
        result = predict(image)
        return jsonify(result)
    except (OSError, ValueError):
        return jsonify({"error": "That file is not a valid image."}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "Please upload an image smaller than 10 MB."}), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=True)
