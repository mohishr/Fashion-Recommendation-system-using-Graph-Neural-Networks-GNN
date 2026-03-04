import os
import traceback

import numpy as np
from flask import Flask, jsonify, render_template, request

from feature_extractor import FeatureExtractor
from inference import OutfitPredictor
from recommendation_engine import RecommenderEngine

app = Flask(__name__)

print('Initializing Feature Extractor...')
extractor = FeatureExtractor()

print('Initializing NGNN Predictor...')
weights_dir = os.environ.get('NGNN_WEIGHTS_DIR', './model_weights')
checkpoint_name = os.environ.get('NGNN_CHECKPOINT_NAME')
predictor = OutfitPredictor(weights_dir=weights_dir, checkpoint_name=checkpoint_name)

print('Initializing Recommender Engine...')
recommender = RecommenderEngine(predictor)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', model_loaded=predictor.is_loaded, ckpt=predictor.loaded_checkpoint)


@app.route('/api/model_status', methods=['GET'])
def model_status():
    return jsonify(
        {
            'model_loaded': predictor.is_loaded,
            'loaded_checkpoint': predictor.loaded_checkpoint,
            'weights_dir': weights_dir,
        }
    )


@app.route('/api/predict', methods=['POST'])
def predict_outfit():
    """JSON API for compatibility scoring."""
    try:
        data = request.get_json()
        if not data or 'items' not in data:
            return jsonify({'error': 'Invalid request. Must contain an "items" array.'}), 400

        items = data['items']
        if len(items) < 2:
            return jsonify({'error': 'An outfit must contain at least 2 items.'}), 400

        images_array, texts_array, graph_array = extractor.process_outfit(items)
        score = predictor.predict(images_array, texts_array, graph_array)

        return jsonify(
            {
                'compatibility_score': score,
                'model_loaded': predictor.is_loaded,
                'loaded_checkpoint': predictor.loaded_checkpoint,
                'message': 'Score successfully calculated.',
            }
        ), 200

    except Exception as e:
        print('Prediction Error: {}'.format(e))
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/predict_form', methods=['POST'])
def predict_form():
    """Simple HTML form interface for quick manual testing."""
    try:
        text_values = [request.form.get('text1', ''), request.form.get('text2', '')]
        path_values = [request.form.get('image1', ''), request.form.get('image2', '')]

        items = []
        for text_val, path_val in zip(text_values, path_values):
            entry = {'text': text_val.strip()}
            if path_val.strip():
                entry['image_path'] = path_val.strip()
            items.append(entry)

        images_array, texts_array, graph_array = extractor.process_outfit(items)
        score = predictor.predict(images_array, texts_array, graph_array)

        return render_template(
            'index.html',
            model_loaded=predictor.is_loaded,
            ckpt=predictor.loaded_checkpoint,
            compatibility_score=score,
            last_payload=items,
        )
    except Exception as e:
        return render_template(
            'index.html',
            model_loaded=predictor.is_loaded,
            ckpt=predictor.loaded_checkpoint,
            error=str(e),
        )


@app.route('/api/recommend_item', methods=['POST'])
def recommend_item():
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

            extracted_outfit.append(
                {'image_embedding': img_feat.tolist(), 'text_embedding': txt_feat.tolist()}
            )

        recommendations = recommender.get_recommendations_for_outfit(
            partial_outfit=extracted_outfit,
            target_category=data['target_category'],
            top_n=10,
        )

        return jsonify({'target_category': data['target_category'], 'recommendations': recommendations}), 200
    except Exception as e:
        print('Recommender Error: {}'.format(e))
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
