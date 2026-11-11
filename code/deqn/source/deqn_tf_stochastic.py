"""Stochastic Brock-Mirman DEQN, TensorFlow/Keras: the legacy input of template 5.

The code of the course notebook 02_02_Brock_Mirman_Uncertainty_DEQN.ipynb (RWTH
Aachen, October 2026) with the markdown, the plots and the second training stage on
simulated states removed. Productivity follows log z' = rho log z + sigma eps; the
conditional expectation in the Euler equation is a 5-node Gauss-Hermite sum.

Kept as it was, including a real bug that no number ever revealed: in compute_cost
the accumulator `expectation` is created with shape (n_data, dim_state) = (n, 2)
instead of (n, 1). Broadcasting then makes errREE an (n, 2) array with two identical
columns. The mean squared error is unchanged, so the training loss looks right, and
the code is wrong. A shape assertion on the residual catches it in one line.

Not run in the lecture (TensorFlow is not a requirement of this repository).
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras

SEED = 0
np.random.seed(SEED)
tf.random.set_seed(SEED)

n_int = 5                                                   # Gauss-Hermite nodes
x_int_norm, w_int = np.polynomial.hermite.hermgauss(n_int)
w_int = tf.constant(w_int / np.sqrt(np.pi), dtype=tf.float32)
x_int_norm = tf.constant(x_int_norm * np.sqrt(2), dtype=tf.float32)

alpha, beta, delta = 0.36, 0.99, 0.1
sigma_tfp, rho_tfp = 0.04, 0.9
x_int = x_int_norm * sigma_tfp                              # innovations at the nodes
NUM_EPISODES_UNIFORM = 5001

num_input, num_hidden1, num_hidden2, num_output = 2, 50, 50, 1   # state = (z, K) in levels
nn = keras.Sequential([
    keras.layers.Input(shape=(num_input,)),
    keras.layers.Dense(num_hidden1, activation="relu"),
    keras.layers.Dense(num_hidden2, activation="relu"),
    keras.layers.Dense(num_output, activation="sigmoid"),
])


@tf.function
def get_singleinside(X_tplus1, nn):
    """(1 / C') (1 - delta + r') at a next-period state."""
    Z_tplus1 = X_tplus1[:, 0:1]
    K_tplus1 = X_tplus1[:, 1:2]
    Y_tplus1 = Z_tplus1 * K_tplus1 ** alpha
    r_tplus1 = alpha * Z_tplus1 * K_tplus1 ** (alpha - 1.)
    s_tplus1 = nn(X_tplus1)
    C_tplus1 = Y_tplus1 - Y_tplus1 * s_tplus1
    return (1. / C_tplus1) * (1. - delta + r_tplus1)


@tf.function
def compute_cost(X, nn):
    """Mean squared relative Euler error on a batch of states X (n_data x 2)."""
    n_data = X.shape[0]
    dim_state = X.shape[1]
    Z_t = X[:, 0:1]
    K_t = X[:, 1:2]
    Y_t = Z_t * K_t ** alpha
    s_t = nn(X)
    K_tplus1 = (1. - delta) * K_t + Y_t * s_t
    C_t = Y_t - Y_t * s_t
    expectation = tf.zeros((n_data, dim_state))      # BUG: shape (n, 2); should be (n, 1)
    for i in range(n_int):
        Z_tplus1 = tf.exp(rho_tfp * tf.math.log(Z_t) + x_int[i])
        X_tplus1 = tf.concat([Z_tplus1, K_tplus1], axis=1)
        expectation = expectation + w_int[i] * get_singleinside(X_tplus1, nn)
    errREE = 1. - 1. / (C_t * beta * expectation)     # (n, 2) with identical columns
    cost = tf.reduce_mean(errREE ** 2)                # ... which leaves the mean unchanged
    return cost, errREE, C_t, K_tplus1


def grad(X, nn):
    with tf.GradientTape() as tape:
        loss_value = compute_cost(X, nn)[0]
    return loss_value, tape.gradient(loss_value, nn.trainable_variables)


def get_training_data(z_lb, z_ub, k_lb, k_ub, n_data):
    """Uniform draws over a box in (z, K); the ergodic set is not used here."""
    Z = tf.random.uniform(shape=[n_data, 1], minval=z_lb, maxval=z_ub, dtype=tf.float32)
    K = tf.random.uniform(shape=[n_data, 1], minval=k_lb, maxval=k_ub, dtype=tf.float32)
    return tf.concat([Z, K], axis=1)


optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
n_data_per_epi, z_lb, z_ub, k_lb, k_ub = 128, 0.7, 1.3, 0.9, 12.0
train_loss = []
for ep in range(NUM_EPISODES_UNIFORM):
    X = get_training_data(z_lb, z_ub, k_lb, k_ub, n_data_per_epi)
    loss, grads = grad(X, nn)
    optimizer.apply_gradients(zip(grads, nn.trainable_variables))
    train_loss.append(loss.numpy())
    if ep % 250 == 0:
        print("episode = {}, loss [log10] = {:.4f}".format(ep, np.log10(loss.numpy())))
