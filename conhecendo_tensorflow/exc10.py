import tensorflow as tf
from datetime import datetime
import os

# 1. Variáveis Globais
with tf.name_scope("variables"):
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
    total_output = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="total_output")

# 2. Configuração do Log
logdir = os.path.join("logs", "model_v2", datetime.now().strftime("%Y%m%d-%H%M%S"))
writer = tf.summary.create_file_writer(logdir)


# 3. Função com Grafo Estático
@tf.function
def executar_modelo(input_tensor):
    # Declarar variáveis como globais para permitir o assign_add dentro da tf.function
    with tf.name_scope("transformation"):
        with tf.name_scope("input"):
            a = input_tensor
        with tf.name_scope("intermediate_layer"):
            b = tf.reduce_prod(a, name="product_b")
            c = tf.reduce_sum(a, name="sum_c")
        with tf.name_scope("output"):
            output = tf.add(b, c, name="output_final")

    with tf.name_scope("update"):
        # Operações de atualização
        update_total = total_output.assign_add(output)
        increment_step = global_step.assign_add(1)

    with tf.name_scope("summaries"):
        avg = tf.divide(update_total, tf.cast(increment_step, tf.float32), name="average")
        tf.summary.scalar('Output', output, step=tf.cast(increment_step, tf.int64))
        tf.summary.scalar('Sum_of_outputs', update_total, step=tf.cast(increment_step, tf.int64))
        tf.summary.scalar('Average_of_outputs', avg, step=tf.cast(increment_step, tf.int64))

    return output


# --- EXECUÇÃO E CAPTURA DO GRAFO ---

input_data_1 = tf.constant([2.0, 3.0], dtype=tf.float32)
input_data_2 = tf.constant([1.0, 5.0, 2.0], dtype=tf.float32)

# IMPORTANTE: Para gerar o grafo no TensorBoard 2.x de forma limpa:
# Criamos a concrete function especificando o "shape" esperado (None permite tamanhos variados)
concrete = executar_modelo.get_concrete_function(tf.TensorSpec(shape=[None], dtype=tf.float32))

with writer.as_default():
    # Escreve o grafo no TensorBoard
    tf.summary.graph(concrete.graph)

    # Execuções normais
    res1 = executar_modelo(input_data_1)
    print(f"Execução 1 - Resultado: {res1.numpy()} | Total: {total_output.numpy()}")

    res2 = executar_modelo(input_data_2)
    print(f"Execução 2 - Resultado: {res2.numpy()} | Total: {total_output.numpy()}")

writer.close()

print(f"\nLogs gravados em: {logdir}")
print("Comando: tensorboard --logdir logs")