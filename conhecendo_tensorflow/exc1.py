from datetime import datetime
import tensorflow as tf

log_dir = "logs/exercicios_tensorboard/" + datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tf.summary.create_file_writer(log_dir)

# Ativa rastreamento do grafo
tf.summary.trace_on(graph=True, profiler=False)

@tf.function
def operacoes():
    a = tf.constant(
        [
            [1, 2],
            [3, 4],
        ],
        dtype=tf.float32,
    )

    b = tf.ones((2, 2))
    c = tf.random.uniform((2, 2))

    sum_ab = tf.add(a, b, name="sum_ab")
    mean_c = tf.reduce_mean(c, name="mean_ab")

    return sum_ab, mean_c


sum_ab, mean_c = operacoes()

with writer.as_default():
    tf.summary.scalar("mean_random_tensor", mean_c, step=0)
    tf.summary.histogram("sum_tensor", sum_ab, step=0)

    # Exporta o grafo
    tf.summary.trace_export(
        name="grafo_operacoes",
        step=0,
        profiler_outdir=log_dir
    )