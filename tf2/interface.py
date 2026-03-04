import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


# =========================
# GGNN MODEL
# =========================

class MessagePassing(layers.Layer):
    def __init__(self, hidden_size):
        super().__init__()
        self.w_out = layers.Dense(hidden_size, use_bias=False)
        self.w_in = layers.Dense(hidden_size, use_bias=False)

    def call(self, h, adj):
        h_out = self.w_out(h)
        agg = tf.matmul(adj, h_out)
        return self.w_in(agg)


class GGNN(Model):
    def __init__(self, hidden_size=64, steps=3):
        super().__init__()
        self.steps = steps
        self.input_proj = layers.Dense(hidden_size, activation="tanh")
        self.message = MessagePassing(hidden_size)
        self.gru = layers.GRUCell(hidden_size)

    def call(self, images, adj):
        h = self.input_proj(images)

        for _ in range(self.steps):
            msg = self.message(h, adj)

            B = tf.shape(msg)[0]
            N = tf.shape(msg)[1]
            H = tf.shape(msg)[2]

            h_flat = tf.reshape(h, [-1, H])
            msg_flat = tf.reshape(msg, [-1, H])

            new_h, _ = self.gru(msg_flat, [h_flat])
            h = tf.reshape(new_h, [B, N, H])

        return h


class RankModel(Model):
    def __init__(self, hidden_size=64, steps=3):
        super().__init__()
        self.gnn = GGNN(hidden_size, steps)
        self.w_conf = layers.Dense(1, activation="sigmoid")
        self.w_score = layers.Dense(1)

    def call(self, images, adj):
        h = self.gnn(images, adj)
        conf = self.w_conf(h)
        score = tf.nn.leaky_relu(self.w_score(h))
        node_score = conf * score
        return tf.reduce_sum(node_score, axis=1)


# =========================
# FEATURE EXTRACTOR (TF ONLY)
# =========================

class FeatureExtractor:
    def __init__(self):
        self.preprocess = tf.keras.applications.inception_v3.preprocess_input
        self.model = tf.keras.applications.InceptionV3(
            include_top=False,
            weights="imagenet",
            pooling="avg"
        )

    def extract_from_array(self, image_array):
        """
        image_array: numpy array (H, W, 3)
        dtype: uint8 or float32
        RGB format
        """

        # Convert to tensor
        img = tf.convert_to_tensor(image_array)

        # Ensure float32
        img = tf.cast(img, tf.float32)

        # Resize using TF
        img = tf.image.resize(img, (299, 299))

        # Add batch dimension
        img = tf.expand_dims(img, axis=0)

        # Preprocess for InceptionV3
        img = self.preprocess(img)

        features = self.model(img, training=False)

        return features[0].numpy()  # 2048-dim


# =========================
# COMPATIBILITY API
# =========================

class OutfitCompatibilityAPI:
    def __init__(self, weights_path="./ggnn_ranker.weights.h5"):
        self.max_items = 10
        self.hidden_size = 64
        self.steps = 3

        self.model = RankModel(self.hidden_size, self.steps)

        # Build model
        dummy_images = np.zeros((1, self.max_items, 2048), dtype=np.float32)
        dummy_graph = np.zeros((1, self.max_items, self.max_items), dtype=np.float32)
        self.model(dummy_images, dummy_graph)

        self.model.load_weights(weights_path)
        print("Model loaded successfully.")

        self.extractor = FeatureExtractor()

    def build_graph(self, num_items):
        graph = np.zeros((self.max_items, self.max_items), dtype=np.float32)
        graph[:num_items, :num_items] = 1.0
        return graph

    def predict_from_arrays(self, image_arrays):
        """
        image_arrays: list of numpy arrays [(H,W,3), ...]
        """

        features = []

        for img in image_arrays:
            feat = self.extractor.extract_from_array(img)
            features.append(feat)

        # Pad to max_items
        while len(features) < self.max_items:
            features.append(np.zeros(2048, dtype=np.float32))

        images = np.array([features])
        graph = np.array([self.build_graph(len(image_arrays))])

        score = self.model(images, graph)

        return float(score.numpy()[0][0])


# =========================
# TEST
# =========================

if __name__ == "__main__":
    api = OutfitCompatibilityAPI()

    # Generate 4 random pixel arrays
    random_images = [
        np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        for _ in range(10)
    ]

    score = api.predict_from_arrays(random_images)

    print("Compatibility score:", score)