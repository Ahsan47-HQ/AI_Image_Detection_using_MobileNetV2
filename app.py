from flask import Flask, render_template, request
from predict import predict
from PIL import Image
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        file = request.files["image"]
        image = Image.open(file).convert("RGB")

        label, confidence = predict(image)
        result = f"{label} ({confidence:.2f})"

    return render_template("index.html", result=result)

# important step in going from local hosting to server hosting is changing:
# app.run(debug=True) to the code below 
if __name__ == "__main__":
    app.run(host="0.0.0.0",port=int(os.environ.get("PORT",10000)))

