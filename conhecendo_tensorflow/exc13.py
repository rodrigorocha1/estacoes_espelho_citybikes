import tensorflow as tf
from datetime import datetime

print("TensorFlow:", tf.__version__)

logdir = "logs/exc13/" + datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tf.summary.create_file_writer(logdir)

@tf.function
def calcular():

    with tf.name_scope("Constantes"):
        k = tf.constant(5, name="constante")

    with tf.name_scope("Operacoes"):
        a = tf.add(2, 2, name='add')
        b = tf.multiply(a, 3, name='mult1')
        c = tf.multiply(b, a, name='mult2')
        d = tf.add(c, k, name='resultado_final')

    return d

# gerar grafo concreto
concrete = calcular.get_concrete_function()

# salvar no TensorBoard
with writer.as_default():
    tf.summary.graph(concrete.graph)

writer.flush()

# executar
resultado = calcular()

print("Resultado:", resultado.numpy())
print("Logs em:", logdir)