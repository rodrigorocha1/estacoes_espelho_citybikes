import tensorflow as tf
from datetime import datetime

print("TensorFlow:", tf.__version__)

# diretório de logs
logdir = "logs/graph/" + datetime.now().strftime("%Y%m%d-%H%M%S")

# writer
writer = tf.summary.create_file_writer(logdir)

# função com grafo organizado
@tf.function
def calcular():

    with tf.name_scope("Entradas"):
        x = tf.constant(2.0, name="x")
        y = tf.constant(3.0, name="y")

    with tf.name_scope("Operacoes_Basicas"):
        soma = tf.add(x, y, name="soma")
        produto = tf.multiply(x, y, name="produto")

    with tf.name_scope("Operacoes_Intermediarias"):
        quadrado = tf.square(soma, name="quadrado_soma")
        dobro = tf.multiply(produto, 2.0, name="dobro_produto")

    with tf.name_scope("Resultado_Final"):
        resultado = tf.add(quadrado, dobro, name="resultado_final")

    return resultado

# ativa trace
tf.summary.trace_on(graph=True, profiler=False)

# executa função
resultado = calcular()

# grava grafo
with writer.as_default():
    tf.summary.trace_export(
        name="grafo_organizado",
        step=0,
        profiler_outdir=logdir
    )

print("Resultado:", resultado.numpy())
print("Logs salvos em:", logdir)