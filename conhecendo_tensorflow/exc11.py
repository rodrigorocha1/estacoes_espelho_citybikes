import tensorflow as tf
from datetime import datetime

print("TensorFlow:", tf.__version__)

# diretório logs
logdir = "logs/graph/" + datetime.now().strftime("%Y%m%d-%H%M%S")

writer = tf.summary.create_file_writer(logdir)

@tf.function
def calcular():

    with tf.name_scope("1_Entradas"):
        x = tf.constant(2.0, name="x")
        y = tf.constant(3.0, name="y")

    with tf.name_scope("2_Operacoes_Basicas"):
        mult = tf.multiply(x, y, name="multiplicacao")
        soma = tf.add(x, y, name="soma")

    with tf.name_scope("3_Combinacao"):
        resultado = tf.add(mult, soma, name="resultado_final")

    return resultado

# gera concrete graph
concrete = calcular.get_concrete_function()

# grava grafo
with writer.as_default():
    tf.summary.graph(concrete.graph)

writer.flush()

# executa
resultado = calcular()

print("Resultado:", resultado.numpy())
print("Logs:", logdir)