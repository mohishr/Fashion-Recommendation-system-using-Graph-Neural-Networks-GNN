import os
import tensorflow as tf
import numpy as np
import sys

# Append parent dir so we can import the original model definitions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model_multimodal import weights, biases, message_pass, GNN

class OutfitPredictor:
    def __init__(self, weights_dir='./model_weights'):
        self.weights_dir = weights_dir
        self.image_hidden_size = 12
        self.text_hidden_size = 12
        self.n_steps = 3
        
        # You'll need to know the max `num_category` used during training.
        # Often this is passed dynamically or hardcoded based on the dataset logic.
        self.num_category = 10  # Ensure this matches training max items
        self.beta = 0.2

        self.graph = tf.Graph()
        with self.graph.as_default():
            self._build_model()
            
            # Start session and attempt to load weights
            self.sess = tf.Session()
            self.saver = tf.train.Saver()
            
            # The authors provided multiple single-file checkpoints. 
            # We explicitly load the main multimodal GNN model checkpoint:
            ckpt_path = os.path.join(self.weights_dir, "model.ckpt-34865")
            
            if os.path.exists(ckpt_path) or os.path.exists(ckpt_path + ".meta") or os.path.exists(ckpt_path + ".index"):
                print(f"Loading weights from {ckpt_path}...")
                # Suppress the warning if it's not a V2 checkpoint
                self.saver.restore(self.sess, ckpt_path)
            else:
                print(f"WARNING: Checkpoint {ckpt_path} not found. Predictions will be random.")
                self.sess.run(tf.global_variables_initializer())

    def _build_model(self):
        """
        Reconstructs the precise tf.placeholder and forward pass logic 
        from main_multi_modal.py (cm_ggnn function).
        """
        self.batch_size = 1 # We are inferring one outfit at a time

        # Placeholders
        self.image_pos = tf.placeholder(tf.float32, [self.batch_size, self.num_category, 2048])
        self.text_pos = tf.placeholder(tf.float32, [self.batch_size, self.num_category, 2757])
        self.graph_pos = tf.placeholder(tf.float32, [self.batch_size, self.num_category, self.num_category])

        # Variables
        hidden_stdv = np.sqrt(1. / (self.image_hidden_size))
        with tf.variable_scope("cm_ggnn", reuse=None):
            self.w_conf_image = tf.get_variable(name='gnn/w/conf_image', shape=[self.image_hidden_size, 1], initializer=tf.random_normal_initializer(hidden_stdv))
            self.w_score_image = tf.get_variable(name='gnn/w/score_image', shape=[self.image_hidden_size, 1], initializer=tf.random_normal_initializer(hidden_stdv))
            self.w_conf_text = tf.get_variable(name='gnn/w/conf_text', shape=[self.text_hidden_size, 1], initializer=tf.random_normal_initializer(hidden_stdv))
            self.w_score_text = tf.get_variable(name='gnn/w/score_text', shape=[self.text_hidden_size, 1], initializer=tf.random_normal_initializer(hidden_stdv))

        # GNN Passes
        with tf.variable_scope("gnn_image", reuse=None):
            image_state_pos, _ = GNN('image', self.image_pos, self.batch_size, self.image_hidden_size, self.n_steps, self.num_category, self.graph_pos)
        
        with tf.variable_scope("gnn_text", reuse=None):
            text_state_pos, _ = GNN('text', self.text_pos, self.batch_size, self.text_hidden_size, self.n_steps, self.num_category, self.graph_pos)

        # Output Score Calculation
        # For batch size = 1, we pull out the first element
        image_conf_pos = tf.nn.sigmoid(tf.reshape(tf.matmul(image_state_pos[0], self.w_conf_image), [1, self.num_category]))
        image_score_pos = tf.reshape(tf.matmul(image_state_pos[0], self.w_score_image), [self.num_category, 1])
        image_score_pos = tf.maximum(0.01 * image_score_pos, image_score_pos)
        image_score_pos = tf.reshape(tf.matmul(image_conf_pos, image_score_pos), [1])
        
        text_conf_pos = tf.nn.sigmoid(tf.reshape(tf.matmul(text_state_pos[0], self.w_conf_text), [1, self.num_category]))
        text_score_pos = tf.reshape(tf.matmul(text_state_pos[0], self.w_score_text), [self.num_category, 1])
        text_score_pos = tf.maximum(0.01 * text_score_pos, text_score_pos)
        text_score_pos = tf.reshape(tf.matmul(text_conf_pos, text_score_pos), [1])
        
        self.score_pos = self.beta * image_score_pos + (1 - self.beta) * text_score_pos

    def predict(self, images_array, texts_array, graph_array):
        """
        Runs the forward pass through the loaded graph.
        """
        feed_dict = {
            self.image_pos: images_array,
            self.text_pos: texts_array,
            self.graph_pos: graph_array
        }
        score = self.sess.run(self.score_pos, feed_dict=feed_dict)
        # Score is returned as a 1D array with 1 float, grab exactly that value
        return float(score[0])
