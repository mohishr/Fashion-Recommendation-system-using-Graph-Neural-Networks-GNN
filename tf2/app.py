import os
import base64
import traceback
import numpy as np
from flask import Flask, request, jsonify, render_template

from interface import OutfitCompatibilityAPI, FeatureExtractor
from recommend import RecommenderEngine

app = Flask(__name__)

# Initialize components
print("Initializing Feature Extractor...")
extractor = FeatureExtractor()

print("Initializing NGNN Predictor...")
predictor = OutfitCompatibilityAPI(weights_path="./ggnn_ranker.weights.h5")

print("Initializing Recommender Engine...")
recommender = RecommenderEngine()
recommender.predictor = predictor
recommender.extractor = extractor

# --------------------------
# Routes
# --------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict_outfit():
    try:
        data = request.get_json()
        if not data or "items" not in data:
            return jsonify({"error": "Request must contain 'items' array"}), 400

        items = data["items"]
        if len(items) < 2:
            return jsonify({"error": "At least 2 items required"}), 400

        # Extract pixel arrays from uploaded items
        image_arrays = []
        for item in items:
            if "image_array" in item:
                image_arrays.append(np.array(item["image_array"], dtype=np.uint8))
            elif "image_base64" in item:
                img_bytes = base64.b64decode(item["image_base64"])
                img_array = extractor.bytes_to_array(img_bytes)
                image_arrays.append(img_array)

        score = predictor.predict_from_arrays(image_arrays)
        normalized_score = max(0, min(100, score * 10))  # scale to 0–100

        return jsonify({
            "compatibility_score": float(normalized_score),
            "message": "Score calculated."
        }), 200

    except Exception as e:
        print("Prediction Error:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/recommend_item", methods=["POST"])
def recommend_item():
    try:
        data = request.get_json()
        if not data or "partial_outfit" not in data or "target_category" not in data:
            return jsonify({"error": "Must provide partial_outfit and target_category"}), 400

        partial_outfit = []
        for item in data["partial_outfit"]:
            if "image_base64" in item:
                img_bytes = base64.b64decode(item["image_base64"])
                img_feat = extractor.extract_from_bytes(img_bytes)
                txt_feat = extractor.extract_text_features(item.get("text", ""))
                partial_outfit.append({
                    "image_embedding": img_feat.tolist(),
                    "text_embedding": txt_feat.tolist()
                })

        top_items = recommender.get_recommendations_for_outfit(
            partial_outfit=partial_outfit,
            target_category=data["target_category"],
            top_n=5
        )

        return jsonify({
            "target_category": data["target_category"],
            "recommendations": top_items
        }), 200

    except Exception as e:
        print("Recommendation Error:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)