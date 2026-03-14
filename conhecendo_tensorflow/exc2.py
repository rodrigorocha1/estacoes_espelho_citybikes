import tensorflow as tf
from datetime import datetime

# diretório de logs
logdir = "logs/grad_graph/" + datetime.now().strftime("%Y%m%d-%H%M%S")

writer = tf.summary.create_file_writer(logdir)

# função com gradiente
@tf.function
def compute_gradient(x, y):

    with tf.GradientTape() as tape:
        z = x**2 + y**3

    grads = tape.gradient(z, [x, y])
    return grads


x = tf.Variable(2.0)
y = tf.Variable(3.0)

# ativa tracing do grafo
tf.summary.trace_on(graph=True, profiler=False)

# executa função
compute_gradient(x, y)

# exporta para TensorBoard
with writer.as_default():
    tf.summary.trace_export(
        name="gradient_graph",
        step=0
    )