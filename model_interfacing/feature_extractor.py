import os
import numpy as np
import tensorflow as tf


class FeatureExtractor:
    def __init__(self, vocab_dir='../data'):
        self.vocab_dir = vocab_dir
        self.TEXT_FEAT_DIM = 2757
        self.preprocess_input = tf.keras.applications.inception_v3.preprocess_input

        disable_imagenet = os.environ.get('NGNN_DISABLE_IMAGENET', '0') == '1'
        model_weights = None if disable_imagenet else 'imagenet'

        try:
            self.image_model = tf.keras.applications.inception_v3.InceptionV3(
                include_top=False,
                weights=model_weights,
                pooling='avg',
            )
            self.image_model_ready = True
        except Exception as e:
            print('WARNING: Failed to initialize InceptionV3 ({}). Falling back to zero image features.'.format(e))
            self.image_model = None
            self.image_model_ready = False

    def extract_image_features(self, image_data, is_path=True):
        """Extract a 2048-dim image vector, or fallback to zeros on failure."""
        import tempfile

        if not self.image_model_ready:
            return np.zeros(2048, dtype=np.float32)

        tmp_path = None
        try:
            if is_path:
                img_path = image_data
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    tmp.write(image_data)
                    tmp_path = tmp.name
                img_path = tmp_path

            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(299, 299))
            x = tf.keras.preprocessing.image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = self.preprocess_input(x)
            features = self.image_model.predict(x, verbose=0)

            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

            return features[0]
        except Exception as e:
            print('Error extracting image features: {}'.format(e))
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
            return np.zeros(2048, dtype=np.float32)

    def extract_text_features(self, text_description):
        _ = text_description
        return np.zeros(self.TEXT_FEAT_DIM, dtype=np.float32)

    def process_outfit(self, items):
        num_items = len(items)
        max_items = 10

        images = []
        texts = []
        import base64

        for item in items:
            if 'image_base64' in item:
                img_bytes = base64.b64decode(item['image_base64'])
                img_feat = self.extract_image_features(img_bytes, is_path=False)
            else:
                img_feat = self.extract_image_features(item.get('image_path'), is_path=True)

            text_feat = self.extract_text_features(item.get('text', ''))
            images.append(img_feat)
            texts.append(text_feat)

        while len(images) < max_items:
            images.append(np.zeros(2048, dtype=np.float32))
            texts.append(np.zeros(self.TEXT_FEAT_DIM, dtype=np.float32))

        graph = np.zeros((max_items, max_items), dtype=np.float32)
        for i in range(num_items):
            for j in range(num_items):
                graph[i, j] = 1.0

        return np.array([images]), np.array([texts]), np.array([graph])
