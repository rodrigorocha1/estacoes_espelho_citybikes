import tensorflow as tf
from datetime import datetime

print("TensorFlow:", tf.__version__)

# pasta de logs
logdir = "logs/graph/" + datetime.now().strftime("%Y%m%d-%H%M%S")

writer = tf.summary.create_file_writer(logdir)


@tf.function
def calcular():

    with tf.name_scope("1_Entradas"):
        x = tf.constant(5.0, name="x")
        y = tf.constant(3.0, name="y")

    with tf.name_scope("2_Processamento"):
        soma = tf.add(x, y, name="soma")
        multiplicacao = tf.multiply(soma, 2.0, name="multiplicacao")
        ajuste = tf.subtract(multiplicacao, 1.0, name="ajuste")

    with tf.name_scope("3_Saida_Final"):

        resultado = tf.identity(ajuste, name="resultado_final")

    return resultado


# ativa trace
tf.summary.trace_on(graph=True, profiler=False)

# executa
resultado = calcular()

# exporta trace
with writer.as_default():
    tf.summary.trace_export(
        name="grafo_profissional",
        step=0,
        profiler_outdir=logdir
    )

print("Resultado:", resultado.numpy())
print("Logs:", logdir)