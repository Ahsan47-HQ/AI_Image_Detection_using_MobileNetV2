from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import predict
from PIL import Image

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "API is running" 

@app.route("/predict", methods=["POST"])
def predict_api():
    file = request.files["image"]
    image = Image.open(file).convert("RGB")

    label, confidence = predict(image)

    return jsonify({
        "label": label,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(port=5000)