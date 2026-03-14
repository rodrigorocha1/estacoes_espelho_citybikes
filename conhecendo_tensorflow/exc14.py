import tensorflow as tf
from datetime import datetime

print("TensorFlow:", tf.__version__)

# --- Diretório de logs ---
logdir = "logs/exc13/" + datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tf.summary.create_file_writer(logdir)

# --- Dados do AND ---
X = tf.constant([[0, 0],
                 [0, 1],
                 [1, 0],
                 [1, 1]], dtype=tf.float32)
Y = tf.constant([[0],
                 [0],
                 [0],
                 [1]], dtype=tf.float32)

# --- Pesos e bias ---
with tf.name_scope("Pesos_e_Bias"):
    W = tf.Variable(tf.random.normal([2, 1]), name="W")
    b = tf.Variable(tf.random.normal([1]), name="b")

# --- Função da rede AND ---
@tf.function
def calcular(x):
    with tf.name_scope("Forward"):
        with tf.name_scope("Linear"):
            z = tf.matmul(x, W) + b
        with tf.name_scope("Ativacao"):
            y_pred = tf.sigmoid(z, name="SigmoidOutput")
    return y_pred

# --- Função de perda ---
@tf.function
def loss_fn(y_true, y_pred):
    with tf.name_scope("Loss"):
        return tf.reduce_mean(-y_true * tf.math.log(y_pred + 1e-8) -
                              (1 - y_true) * tf.math.log(1 - y_pred + 1e-8))

# --- Otimizador ---
optimizer = tf.optimizers.SGD(learning_rate=0.1)

# --- Treinamento ---
for step in range(1000):
    with tf.GradientTape() as tape:
        y_pred = calcular(X)
        loss = loss_fn(Y, y_pred)
    grads = tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(grads, [W, b]))
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.numpy():.4f}")

# --- Teste da rede ---
y_test = calcular(X)
print("\nEntradas:")
print(X.numpy())
print("Saídas previstas (arredondadas):")
print(tf.round(y_test).numpy())

# --- FORÇAR criação do grafo e salvar no TensorBoard ---
_ = calcular(X)  # chamada prévia para gerar o grafo
concrete = calcular.get_concrete_function(tf.TensorSpec(shape=[None, 2], dtype=tf.float32))

with writer.as_default():
    tf.summary.graph(concrete.graph)  # salva grafo
writer.flush()

print(f"\nGrafo salvo! Abra no TensorBoard com:\n\ntensorboard --logdir={logdir}")