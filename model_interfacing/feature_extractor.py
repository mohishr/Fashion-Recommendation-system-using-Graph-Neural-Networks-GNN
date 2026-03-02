import os
import json
import numpy as np
import tensorflow as tf

class FeatureExtractor:
    def __init__(self, vocab_dir='../data'):
        # We will use tf.keras.applications.inception_v3.InceptionV3 for images 
        # as seen in `data/use_inception_for_vec.py`
        self.image_model = tf.keras.applications.inception_v3.InceptionV3(
            include_top=False, 
            weights='imagenet', 
            pooling='avg'
        )
        self.preprocess_input = tf.keras.applications.inception_v3.preprocess_input
        self.vocab_dir = vocab_dir
        
        # In a full production system, you would load the word2vec model
        # or the pre-computed text vectors from the dataset.
        # For this API template, we provide a placeholder that returns the expected shape (2757 zeros)
        # based on the `main_multi_modal.py` placeholder `text_pos = tf.placeholder(tf.float32, [batch_size, num_category, 2757])`
        self.TEXT_FEAT_DIM = 2757
        
    def extract_image_features(self, image_data, is_path=True):
        """
        Extracts 2048-dimensional feature vector from an image using InceptionV3.
        """
        import tempfile
        tmp_path = None
        try:
            if is_path:
                img_path = image_data
            else:
                # Write bytes to a temporary file safely
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    tmp.write(image_data)
                    tmp_path = tmp.name
                img_path = tmp_path
                
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(299, 299))
            x = tf.keras.preprocessing.image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = self.preprocess_input(x)
            
            # Predict returns a [1, 2048] array
            features = self.image_model.predict(x)
            
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
                
            return features[0]
        except Exception as e:
            print(f"Error extracting image features: {e}")
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
            # return zero vector on failure to prevent crashes during demo
            return np.zeros(2048, dtype=np.float32)

    def extract_text_features(self, text_description):
        """
        Extracts a 2757-dimensional feature vector for the given text.
        In the original repo, this was done via pre-computed word2vec/onehot embeddings.
        """
        # TODO: Implement real text parsing matching `.data/onehot_embedding.py`
        # For now, return a zeroed vector to ensure the API can run
        features = np.zeros(self.TEXT_FEAT_DIM, dtype=np.float32)
        return features

    def process_outfit(self, items):
        """
        Given a list of items (each a dict with 'image_base64' or 'image_path' and 'text'), 
        extracts features and builds the graph representation.
        """
        num_items = len(items)
        MAX_ITEMS = 10 
        
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
            
        # Pad the rest with zeros if needed by your specific tf placeholder size
        while len(images) < MAX_ITEMS:
            images.append(np.zeros(2048, dtype=np.float32))
            texts.append(np.zeros(self.TEXT_FEAT_DIM, dtype=np.float32))
            
        # Create a simple fully connected graph for the items present
        # shape [MAX_ITEMS, MAX_ITEMS]
        graph = np.zeros((MAX_ITEMS, MAX_ITEMS), dtype=np.float32)
        for i in range(num_items):
            for j in range(num_items):
                graph[i, j] = 1.0
                
        # Return with batch dimension prepended (batch_size = 1)
        return (
            np.array([images]), 
            np.array([texts]), 
            np.array([graph])
        )
