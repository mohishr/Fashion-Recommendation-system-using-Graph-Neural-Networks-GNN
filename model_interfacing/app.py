from flask import Flask, request, jsonify
import sys
import os

from feature_extractor import FeatureExtractor
from inference import OutfitPredictor
from recommendation_engine import RecommenderEngine

app = Flask(__name__)

# Initialize components
print("Initializing Feature Extractor...")
extractor = FeatureExtractor()

print("Initializing NGNN Predictor...")
# It will load out of model_weights due to the default arg
predictor = OutfitPredictor()

print("Initializing Recommender Engine...")
recommender = RecommenderEngine(predictor)

@app.route('/api/predict', methods=['POST'])
def predict_outfit():
    """
    Expects a JSON payload with a list of items:
    {
      "items": [
         {"text": "black t-shirt", "image_path": "/path/to/image1.jpg"},
         {"text": "blue denim jeans", "image_path": "/path/to/image2.jpg"}
      ]
    }
    """
    try:
        data = request.get_json()
        if not data or 'items' not in data:
            return jsonify({'error': 'Invalid request. Must contain an "items" array.'}), 400
            
        items = data['items']
        if len(items) < 2:
            return jsonify({'error': 'An outfit must contain at least 2 items.'}), 400
            
        print(f"Received request to evaluate outfit with {len(items)} items")
        
        # 1. Feature Extraction Pipeline
        images_array, texts_array, graph_array = extractor.process_outfit(items)
        
        # 2. Inference Pipeline
        score = predictor.predict(images_array, texts_array, graph_array)
        
        return jsonify({
            'compatibility_score': score,
            'message': 'Score successfully calculated.'
        }), 200

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommend_item', methods=['POST'])
def recommend_item():
    """
    Fill-In-The-Blank (FITB) Feature: Complete The Look
    Expects a partial outfit and a target category to search for.
    Payload: 
    {
       "partial_outfit": [
          {"image_base64": "...base64 string...", "text": "black t-shirt"}
       ], 
       "target_category": "Jeans"
    }
    """
    data = request.get_json()
    if not data or 'partial_outfit' not in data or 'target_category' not in data:
        return jsonify({'error': 'Must provide partial_outfit and target_category'}), 400
        
    try:
        import base64
        extracted_outfit = []
        for item in data['partial_outfit']:
            if 'image_base64' in item:
                img_bytes = base64.b64decode(item['image_base64'])
                img_feat = extractor.extract_image_features(img_bytes, is_path=False)
            else:
                img_feat = np.zeros(2048, dtype=np.float32)
                
            txt_feat = extractor.extract_text_features(item.get('text', ''))
            
            extracted_outfit.append({
                "image_embedding": img_feat.tolist(),
                "text_embedding": txt_feat.tolist()
            })
            
        recommendations = recommender.get_recommendations_for_outfit(
            partial_outfit=extracted_outfit,
            target_category=data['target_category'],
            top_n=10
        )
        
        return jsonify({
            "target_category": data['target_category'],
            "recommendations": recommendations
        }), 200
    except Exception as e:
        print(f"Recommender Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate_outfits', methods=['POST'])
def generate_outfits():
    """
    Virtual Stylist Feature: 
    Given a list of items a user owns (user_closet), generate the top N best outfits.
    Payload: {"user_closet": [...items...], "target_outfit_size": 4}
    """
    # TODO: Use combinatorial search to generate outfit subsets from user_closet,
    # evaluate each via predictor.predict(), and return the highest scoring groups.
    return jsonify({
        "top_outfits": [] # Placeholder
    }), 200

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    # Run the server. In production, use waitess or gunicorn
    app.run(host='0.0.0.0', port=5000, debug=False)
