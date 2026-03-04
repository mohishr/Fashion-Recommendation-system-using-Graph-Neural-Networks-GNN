import os
import sys
import tensorflow as tf
import numpy as np

# Append parent dir so we can import the original model definitions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model_multimodal import GNN


class OutfitPredictor:
    def __init__(self, weights_dir='./model_weights', checkpoint_name=None):
        self.weights_dir = weights_dir
        self.image_hidden_size = 12
        self.text_hidden_size = 12
        self.n_steps = 3
        self.num_category = 10  # must match trained model
        self.beta = 0.2
        self.loaded_checkpoint = None

        self.graph = tf.Graph()
        with self.graph.as_default():
            self._build_model()
            self.sess = tf.Session()
            self.saver = tf.train.Saver()

            ckpt_path = self._resolve_checkpoint_path(checkpoint_name)
            if ckpt_path:
                print("Loading weights from {}...".format(ckpt_path))
                self.saver.restore(self.sess, ckpt_path)
                self.loaded_checkpoint = ckpt_path
            else:
                print("WARNING: No checkpoint found. Initializing random weights.")
                self.sess.run(tf.global_variables_initializer())

    def _resolve_checkpoint_path(self, checkpoint_name=None):
        """Find a TensorFlow checkpoint prefix path.

        Priority:
        1) explicit checkpoint_name argument
        2) env NGNN_CKPT_PATH
        3) TensorFlow checkpoint state file under weights_dir
        4) First '*.index' checkpoint found under weights_dir
        """
        candidates = []

        if checkpoint_name:
            candidates.append(os.path.join(self.weights_dir, checkpoint_name))

        env_ckpt = os.environ.get('NGNN_CKPT_PATH')
        if env_ckpt:
            candidates.append(env_ckpt)

        candidates.append(os.path.join(self.weights_dir, 'model.ckpt-34865'))
        candidates.append(os.path.join(self.weights_dir, 'cm_ggnn.ckpt'))

        for prefix in candidates:
            if self._checkpoint_prefix_exists(prefix):
                return prefix

        ckpt_state = tf.train.get_checkpoint_state(self.weights_dir)
        if ckpt_state and ckpt_state.model_checkpoint_path:
            ckpt_prefix = ckpt_state.model_checkpoint_path
            if not os.path.isabs(ckpt_prefix):
                ckpt_prefix = os.path.join(self.weights_dir, ckpt_prefix)
            if self._checkpoint_prefix_exists(ckpt_prefix):
                return ckpt_prefix

        if os.path.isdir(self.weights_dir):
            for name in sorted(os.listdir(self.weights_dir)):
                if name.endswith('.index'):
                    return os.path.join(self.weights_dir, name[:-6])

        return None

    @staticmethod
    def _checkpoint_prefix_exists(prefix):
        return (
            os.path.exists(prefix)
            or os.path.exists(prefix + '.index')
            or os.path.exists(prefix + '.meta')
        )

    @property
    def is_loaded(self):
        return self.loaded_checkpoint is not None

    def _build_model(self):
        self.batch_size = 1

        self.image_pos = tf.placeholder(tf.float32, [self.batch_size, self.num_category, 2048])
        self.text_pos = tf.placeholder(tf.float32, [self.batch_size, self.num_category, 2757])
        self.graph_pos = tf.placeholder(tf.float32, [self.batch_size, self.num_category, self.num_category])

        hidden_stdv = np.sqrt(1.0 / self.image_hidden_size)
        with tf.variable_scope('cm_ggnn', reuse=None):
            self.w_conf_image = tf.get_variable(
                name='gnn/w/conf_image',
                shape=[self.image_hidden_size, 1],
                initializer=tf.random_normal_initializer(hidden_stdv),
            )
            self.w_score_image = tf.get_variable(
                name='gnn/w/score_image',
                shape=[self.image_hidden_size, 1],
                initializer=tf.random_normal_initializer(hidden_stdv),
            )
            self.w_conf_text = tf.get_variable(
                name='gnn/w/conf_text',
                shape=[self.text_hidden_size, 1],
                initializer=tf.random_normal_initializer(hidden_stdv),
            )
            self.w_score_text = tf.get_variable(
                name='gnn/w/score_text',
                shape=[self.text_hidden_size, 1],
                initializer=tf.random_normal_initializer(hidden_stdv),
            )

        with tf.variable_scope('gnn_image', reuse=None):
            image_state_pos, _ = GNN(
                'image',
                self.image_pos,
                self.batch_size,
                self.image_hidden_size,
                self.n_steps,
                self.num_category,
                self.graph_pos,
            )

        with tf.variable_scope('gnn_text', reuse=None):
            text_state_pos, _ = GNN(
                'text',
                self.text_pos,
                self.batch_size,
                self.text_hidden_size,
                self.n_steps,
                self.num_category,
                self.graph_pos,
            )

        image_conf_pos = tf.nn.sigmoid(
            tf.reshape(tf.matmul(image_state_pos[0], self.w_conf_image), [1, self.num_category])
        )
        image_score_pos = tf.reshape(tf.matmul(image_state_pos[0], self.w_score_image), [self.num_category, 1])
        image_score_pos = tf.maximum(0.01 * image_score_pos, image_score_pos)
        image_score_pos = tf.reshape(tf.matmul(image_conf_pos, image_score_pos), [1])

        text_conf_pos = tf.nn.sigmoid(
            tf.reshape(tf.matmul(text_state_pos[0], self.w_conf_text), [1, self.num_category])
        )
        text_score_pos = tf.reshape(tf.matmul(text_state_pos[0], self.w_score_text), [self.num_category, 1])
        text_score_pos = tf.maximum(0.01 * text_score_pos, text_score_pos)
        text_score_pos = tf.reshape(tf.matmul(text_conf_pos, text_score_pos), [1])

        self.score_pos = self.beta * image_score_pos + (1 - self.beta) * text_score_pos

    def predict(self, images_array, texts_array, graph_array):
        feed_dict = {
            self.image_pos: images_array,
            self.text_pos: texts_array,
            self.graph_pos: graph_array,
        }
        score = self.sess.run(self.score_pos, feed_dict=feed_dict)
        return float(score[0])
