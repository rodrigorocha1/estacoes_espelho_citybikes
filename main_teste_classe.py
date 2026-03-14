import numpy as np
import tensorflow as tf
from src.autoencoders_citybikes import AutoEncodersCitybikes
np.random.seed(42)

X = np.random.rand(1000, 10).astype(np.float32)

dataset = tf.data.Dataset.from_tensor_slices(X).batch(32)

modelo = AutoEncodersCitybikes(
    input_dim=10,
    hidden_dim=16,
    latent_dim=3,
    learning_rate=0.001,

)

modelo.fit(dataset, epochs=20, patience=5)

latent = modelo.latent_space(X)
print(f'latent: {latent} {latent.shape}')

erros = modelo.reconstruction_error(X)

print(f'\nPrimeiros erros: {erros[:10]}')

errors, threshold, anomalies = modelo.detect_anomalies(X, percentile=95)
print(threshold)
print(anomalies.sum())