import os
import traceback
import base64
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request

from outfit_interface_px_tfonly import OutfitCompatibilityAPI
from recommendation_engine import RecommenderEngine

app = Flask(__name__)

# ==========================================================
# Initialize Model API
# ==========================================================

print("Initializing Outfit Compatibility API...")
api = OutfitCompatibilityAPI()

print("Initializing Recommender Engine...")
recommender = RecommenderEngine(api)


# ==========================================================
# ROUTES
# ==========================================================

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


# ----------------------------------------------------------
# MODEL STATUS
# ----------------------------------------------------------

@app.route('/api/model_status', methods=['GET'])
def model_status():
    return jsonify({
        'model_loaded': True,
        'message': 'Model loaded successfully.'
    })


# ----------------------------------------------------------
# COMPATIBILITY PREDICTION
# ----------------------------------------------------------

@app.route('/api/predict', methods=['POST'])
def predict_outfit():
    """
    Expected JSON:
    {
        "items": [
            { "image_base64": "..." },
            { "image_base64": "..." }
        ]
    }
    """

    try:
        data = request.get_json()

        if not data or 'items' not in data:
            return jsonify({'error': 'Invalid request. Must contain "items" array.'}), 400

        items = data['items']

        if len(items) < 2:
            return jsonify({'error': 'Outfit must contain at least 2 items.'}), 400

        image_arrays = []

        for item in items:
            if 'image_base64' not in item:
                continue

            img_bytes = base64.b64decode(item['image_base64'])

            # Decode using TensorFlow
            img = tf.image.decode_image(img_bytes, channels=3)
            img = img.numpy()

            image_arrays.append(img)

        if len(image_arrays) < 2:
            return jsonify({'error': 'At least 2 valid images required.'}), 400

        score = api.predict_from_arrays(image_arrays)

        return jsonify({
            'compatibility_score': float(score),
            'message': 'Score successfully calculated.'
        }), 200

    except Exception as e:
        print("Prediction Error:", e)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ----------------------------------------------------------
# ITEM RECOMMENDATION
# ----------------------------------------------------------

@app.route('/api/recommend_item', methods=['POST'])
def recommend_item():
    """
    Expected JSON:
    {
        "partial_outfit": [
            { "image_base64": "..." }
        ],
        "target_category": "shoes"
    }
    """

    try:
        data = request.get_json()

        if not data or 'partial_outfit' not in data or 'target_category' not in data:
            return jsonify({'error': 'Must provide partial_outfit and target_category'}), 400

        extracted_outfit = []

        for item in data['partial_outfit']:
            if 'image_base64' not in item:
                continue

            img_bytes = base64.b64decode(item['image_base64'])

            img = tf.image.decode_image(img_bytes, channels=3)
            img = img.numpy()

            # Extract embedding using API extractor
            img_embedding = api.extractor.extract_from_array(img)

            extracted_outfit.append({
                'image_embedding': img_embedding.tolist()
            })

        recommendations = recommender.get_recommendations_for_outfit(
            partial_outfit=extracted_outfit,
            target_category=data['target_category'],
            top_n=10
        )

        return jsonify({
            'target_category': data['target_category'],
            'recommendations': recommendations
        }), 200

    except Exception as e:
        print("Recommender Error:", e)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ----------------------------------------------------------
# HEALTH CHECK
# ----------------------------------------------------------

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200


# ==========================================================
# RUN SERVER
# ==========================================================

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)