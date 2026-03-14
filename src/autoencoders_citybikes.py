import datetime

import tensorflow as tf
from tqdm import tqdm


class AutoencodersCityBikes(tf.Module):
    def __init__(
            self,
            input_dim: int,
            latent_dim: int,
            hidden_dim: int = 32,
            learning_rate: float = 0.001,
            dropout_rate: float = 0.1,
            l2_lambda: float = 0.0001,
            seed: int = 42,
            logdir: str = 'logs/autoencoders_citybikes',

    ):
        super().__init__()

        tf.random.set_seed(seed)
        np.random.seed(seed)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda

        # Encoder

        self.w1 = tf.Variable(
            tf.random.normal([input_dim, hidden_dim]),
            name="w1",
        )

        self.b1 = tf.Variable(
            tf.zeros([hidden_dim]),
        )

        self.w2 = tf.Variable(
            tf.random.normal([hidden_dim, latent_dim]),
            name="w2",
        )

        self.b2 = tf.Variable(
            tf.zeros([latent_dim]),
        )

        # decoder

        self.w3 = tf.Variable(
            tf.random.normal([latent_dim, hidden_dim]),
            name="w3",
        )
        self.b3 = tf.Variable(
            tf.zeros([hidden_dim]),
        )
        self.w4 = tf.Variable(
            tf.random.normal([hidden_dim, input_dim]),
        )
        self.b4 = tf.Variable(
            tf.zeros([input_dim]),
        )

        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

        self.writer = tf.summary.create_file_writer(
            os.path.join(
                logdir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            )
        )

    @tf.function
    def normalize(self, x):
        mean, variance = tf.nn.moments(x, axes=[0])
        return (x - mean) / tf.sqrt(variance + 1e-8)


