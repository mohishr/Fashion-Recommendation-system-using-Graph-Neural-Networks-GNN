import base64
import random
from io import BytesIO

from flask import Flask, request, jsonify
from PIL import Image
import numpy as np

app = Flask(__name__)


# ===============================
# Utility Functions
# ===============================

def read_images(files):
    """
    Convert uploaded images into PIL and numpy arrays
    """
    images = []
    for file in files:
        img = Image.open(file.stream).convert("RGB")
        images.append(img)
    return images


def image_to_base64(img):
    """
    Convert PIL image to base64 string
    """
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()


# ===============================
# API 1: Outfit Compatibility
# ===============================

@app.route("/api/predict", methods=["POST"])
def predict_outfit():

    images = request.files.getlist("images")

    if len(images) < 2:
        return jsonify({"error": "Upload at least 2 images"}), 400

    pil_images = read_images(images)

    # ------------------------------
    # Dummy compatibility model
    # Replace this with your ML code
    # ------------------------------
    score = random.uniform(0.5, 0.95)

    return jsonify({
        "compatibility_score": score
    })


# ===============================
# API 2: Recommend Item
# ===============================

@app.route("/api/recommend_item", methods=["POST"])
def recommend_item():

    target_category = request.form.get("target_category")
    images = request.files.getlist("images")

    if not images:
        return jsonify({"error": "Upload at least one image"}), 400

    pil_images = read_images(images)

    # ------------------------------
    # Dummy recommendation results
    # ------------------------------
    recommendations = []

    for i in range(5):

        # Create dummy colored image
        img = Image.new(
            "RGB",
            (150, 150),
            (
                random.randint(0,255),
                random.randint(0,255),
                random.randint(0,255)
            )
        )

        base64_img = image_to_base64(img)

        recommendations.append({
            "score": random.uniform(0.6, 0.95),
            "image_base64": base64_img,
            "category": target_category
        })

    return jsonify({
        "recommendations": recommendations
    })


# ===============================
# API 3: Virtual Stylist
# ===============================

@app.route("/api/generate_outfits", methods=["POST"])
def generate_outfits():

    outfits = [
        "Casual street outfit: hoodie + jeans + sneakers",
        "Smart casual: blazer + white tee + chinos",
        "Summer look: linen shirt + shorts + loafers",
        "Winter outfit: coat + knit sweater + boots",
        "Minimalist: black tee + slim pants + white sneakers"
    ]

    return jsonify({
        "message": random.choice(outfits)
    })


# ===============================
# Health Check
# ===============================

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "running"})


# ===============================
# Run Server
# ===============================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)