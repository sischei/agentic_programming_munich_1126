"""Deterministic Brock-Mirman DEQN, TensorFlow/Keras: the legacy input of Step 1.

This is the code of the course notebook 02_01_Brock_Mirman_1972_DEQN.ipynb
(Deep Learning for Solving and Estimating Dynamic Economic Models, RWTH Aachen,
October 2026) with the markdown and the plotting removed. It is what the agent is
asked to port to PyTorch in Part III of the lecture. It is kept as it was, including
its weaknesses: the only accuracy measure it prints is the training loss.

Model: max sum beta^t ln C_t  s.t.  K_{t+1} + C_t = K_t^alpha + (1 - delta) K_t.
With delta = 1 the solution is known: K_{t+1} = alpha beta K_t^alpha, i.e. a constant
savings rate s* = alpha beta = 0.3564 and a stationary point K* = (alpha beta)^(1/(1-alpha)).

Not run in the lecture (TensorFlow is not a requirement of this repository).
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras

RUN_MODE = "teaching"                      # "smoke" | "teaching" | "production"
SEED = 0
NUM_EPISODES = {"smoke": 201, "teaching": 2001, "production": 10001}[RUN_MODE]
np.random.seed(SEED)
tf.random.set_seed(SEED)

alpha = 0.36   # capital share
beta = 0.99    # discount factor
delta = 1.0    # depreciation; with delta = 1 the closed form holds


def k_compute_infty(alpha, beta):
    """Stationary point for full depreciation."""
    return (1 / (beta * alpha)) ** (1 / (alpha - 1))


def Kplus_compute_analytic(K, alpha, beta):
    """Next-period capital under the closed form (delta = 1)."""
    return alpha * beta * K ** alpha


# The network maps K_t to the savings rate s_t in (0, 1); K_{t+1} = (1 - delta) K_t + s_t Y_t.
num_input, num_hidden1, num_hidden2, num_output = 1, 50, 50, 1
nn = keras.Sequential([
    keras.layers.Input(shape=(num_input,)),
    keras.layers.Dense(num_hidden1, activation="relu"),
    keras.layers.Dense(num_hidden2, activation="relu"),
    keras.layers.Dense(num_output, activation="sigmoid"),
])


@tf.function
def compute_cost(X, nn):
    """Mean squared relative Euler error on a batch of states X (n_data x 1)."""
    K_t = X
    Y_t = K_t ** alpha
    s_t = nn(X)
    K_tplus1 = (1. - delta) * K_t + Y_t * s_t
    C_t = Y_t - Y_t * s_t
    Y_tplus1 = K_tplus1 ** alpha
    s_tplus1 = nn(K_tplus1)
    C_tplus1 = Y_tplus1 - s_tplus1 * Y_tplus1
    R_tplus1 = alpha * K_tplus1 ** (alpha - 1.)
    errREE = 1 - C_tplus1 / (beta * C_t * (R_tplus1 + 1. - delta))
    cost = tf.reduce_mean(errREE ** 2)
    return cost, errREE, C_t, C_tplus1, K_tplus1, R_tplus1


def grad(X, nn):
    with tf.GradientTape() as tape:
        loss_value = compute_cost(X, nn)[0]
    return loss_value, tape.gradient(loss_value, nn.trainable_variables)


def get_training_data(k_lb, k_ub, n_data):
    """States are drawn from a fixed interval, not from the ergodic set."""
    return tf.random.uniform(shape=[n_data, 1], minval=k_lb, maxval=k_ub, dtype=tf.float32)


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
n_data_per_epi, k_lb, k_ub = 64, 0.10, 1.0
train_loss = []
for ep in range(NUM_EPISODES):
    X = get_training_data(k_lb, k_ub, n_data_per_epi)
    loss, grads = grad(X, nn)
    optimizer.apply_gradients(zip(grads, nn.trainable_variables))
    train_loss.append(loss.numpy())
    if ep % 100 == 0:
        print("episode = {}, loss [log10] = {:.4f}".format(ep, np.log10(loss.numpy())))
# The notebook ends here: it plots the policy against alpha beta K^alpha but never
# prints the error against the closed form.
