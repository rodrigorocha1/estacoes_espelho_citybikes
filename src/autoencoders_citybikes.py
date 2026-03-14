from __future__ import annotations

import os
from datetime import datetime
from typing import Tuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm


class AutoEncodersCitybikes(tf.Module):

    def __init__(
            self,
            input_dim: int,
            latent_dim: int,
            hidden_dim: int = 32,
            learning_rate: float = 0.001,
            dropout_rate: float = 0.2,
            l2_lambda: float = 0.0001,
            seed: int = 42,
            logdir: str = "logs/autoencoder"
    ) -> None:

        super().__init__()

        tf.random.set_seed(seed)
        np.random.seed(seed)

        self.input_dim: int = input_dim
        self.latent_dim: int = latent_dim
        self.hidden_dim: int = hidden_dim
        self.dropout_rate: float = dropout_rate
        self.l2_lambda: float = l2_lambda

        self.w1: tf.Variable = tf.Variable(
            tf.random.normal([input_dim, hidden_dim]),
            name="w1"
        )
        self.b1: tf.Variable = tf.Variable(
            tf.zeros([hidden_dim])
        )

        self.w2: tf.Variable = tf.Variable(
            tf.random.normal([hidden_dim, latent_dim]),
            name="w2"
        )
        self.b2: tf.Variable = tf.Variable(
            tf.zeros([latent_dim])
        )

        self.w3: tf.Variable = tf.Variable(
            tf.random.normal([latent_dim, hidden_dim]),
            name="w3"
        )
        self.b3: tf.Variable = tf.Variable(
            tf.zeros([hidden_dim])
        )

        self.w4: tf.Variable = tf.Variable(
            tf.random.normal([hidden_dim, input_dim]),
            name="w4"
        )
        self.b4: tf.Variable = tf.Variable(
            tf.zeros([input_dim])
        )

        self.optimizer: tf.optimizers.Adam = tf.optimizers.Adam(
            learning_rate
        )

        self.writer: tf.summary.SummaryWriter = tf.summary.create_file_writer(
            os.path.join(
                logdir,
                datetime.now().strftime("%Y%m%d-%H%M%S")
            )
        )

    @tf.function
    def normalize(
            self,
            x: tf.Tensor
    ) -> tf.Tensor:

        mean, variance = tf.nn.moments(
            x,
            axes=[0]
        )

        return (x - mean) / tf.sqrt(variance + 1e-8)

    @tf.function
    def encode(
            self,
            x: tf.Tensor,
            training: bool = False
    ) -> tf.Tensor:

        x = self.normalize(x)

        h1 = tf.nn.relu(
            tf.matmul(x, self.w1) + self.b1
        )

        if training:
            h1 = tf.nn.dropout(
                h1,
                rate=self.dropout_rate
            )

        z = tf.nn.relu(
            tf.matmul(h1, self.w2) + self.b2
        )

        return z

    @tf.function
    def decode(
            self,
            z: tf.Tensor,
            training: bool = False
    ) -> tf.Tensor:

        h2 = tf.nn.relu(
            tf.matmul(z, self.w3) + self.b3
        )

        if training:
            h2 = tf.nn.dropout(
                h2,
                rate=self.dropout_rate
            )

        x_hat = tf.matmul(
            h2,
            self.w4
        ) + self.b4

        return x_hat

    @tf.function
    def forward(
            self,
            x: tf.Tensor,
            training: bool = False
    ) -> tf.Tensor:

        z = self.encode(
            x,
            training
        )

        x_hat = self.decode(
            z,
            training
        )

        return x_hat

    @tf.function
    def l2_regularization(self) -> tf.Tensor:

        return self.l2_lambda * (
                tf.reduce_sum(tf.square(self.w1)) +
                tf.reduce_sum(tf.square(self.w2)) +
                tf.reduce_sum(tf.square(self.w3)) +
                tf.reduce_sum(tf.square(self.w4))
        )

    @tf.function
    def compute_loss(
            self,
            x: tf.Tensor
    ) -> tf.Tensor:

        x_hat = self.forward(
            x,
            training=True
        )

        reconstruction_loss = tf.reduce_mean(
            tf.square(x - x_hat)
        )

        reg_loss = self.l2_regularization()

        return reconstruction_loss + reg_loss

    def train_step(
            self,
            x: tf.Tensor
    ) -> tf.Tensor:

        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)

        variables = [
            self.w1, self.b1,
            self.w2, self.b2,
            self.w3, self.b3,
            self.w4, self.b4
        ]

        grads = tape.gradient(
            loss,
            variables
        )

        self.optimizer.apply_gradients(
            zip(grads, variables)
        )

        return loss

    def fit(
            self,
            dataset: tf.data.Dataset,
            epochs: int = 100,
            patience: int = 10
    ) -> None:

        best_loss: float = np.inf
        wait: int = 0

        for epoch in range(epochs):

            losses: list[float] = []

            progress = tqdm(
                dataset,
                desc=f"Epoch {epoch + 1}/{epochs}"
            )

            for batch in progress:
                loss = self.train_step(batch)

                loss_value: float = float(
                    loss.numpy()
                )

                losses.append(loss_value)

                progress.set_postfix(
                    batch_loss=f"{loss_value:.6f}"
                )

            epoch_loss: float = float(
                np.mean(losses)
            )

            print(
                f"Epoch {epoch + 1}: mean_loss={epoch_loss:.6f}"
            )

            with self.writer.as_default():

                tf.summary.scalar(
                    "loss",
                    epoch_loss,
                    step=epoch
                )

            if epoch_loss < best_loss:

                best_loss = epoch_loss
                wait = 0

            else:

                wait += 1

            if wait >= patience:
                print(
                    f"Early stopping na época {epoch + 1}"
                )
                break

    def latent_space(
            self,
            X: np.ndarray
    ) -> np.ndarray:

        return self.encode(
            tf.constant(X),
            training=False
        ).numpy()

    def reconstruction_error(
            self,
            X: np.ndarray
    ) -> np.ndarray:

        X_tf = tf.constant(X)

        X_hat = self.forward(
            X_tf,
            training=False
        )

        error = tf.reduce_mean(
            tf.square(X_tf - X_hat),
            axis=1
        )

        return error.numpy()

    def detect_anomalies(
            self,
            X: np.ndarray,
            percentile: int = 95
    ) -> Tuple[np.ndarray, float, np.ndarray]:

        errors = self.reconstruction_error(X)

        threshold = float(
            np.percentile(errors, percentile)
        )

        anomalies = errors > threshold

        return errors, threshold, anomalies
