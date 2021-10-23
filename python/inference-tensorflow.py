import tensorflow as tf
import tensorflow_probability.distributions as tfd
import numpy as np
import math

tf = tf.compat.v1
tf.disable_v2_behavior()


# Taken from https://github.com/tensorflow/tensorflow/issues/9162
def _entry_stop_gradients(target, mask):
    mask_h = tf.cast(tf.logical_not(mask), dtype=target.dtype)
    mask = tf.cast(mask, dtype=target.dtype)

    tf.add(tf.stop_gradient(tf.multiply(mask_h, target)), tf.multiply(mask, target))

def _colmeans(mat):
    '''replicates R's colMeans'''
    return np.mean(mat, axis=0)

def _scale(mat):
    '''This replicates R's scale. It notably uses std with ddof 1
    '''
    return (mat - np.mean(mat)) / np.std(mat, ddof=1)

def inference_tensorflow(Y, rho, s, X, G, C, N, P, shrinkage, B=10, verbose=False, n_batches=1, 
                         rel_tol_adam = 1e-4,
                         rel_tol_em = 1e-4,
                         max_iter_adam = 1e5,
                         max_iter_em = 20,
                         learning_rate = 1e-4,
                         random_seed = None,
                         min_delta = 2,
                         dirichlet_concentration = 1e-2,
                         threads = 0):


    tf.reset_default_graph()

    # Data placeholders
    Y_ = tf.placeholder(tf.float64, shape=(None, G), name="Y_")
    X_ = tf.placeholder(tf.float64, shape=(None, P), name="X_")
    s_ = tf.placeholder(tf.float64, shape=(None), name="s_")
    rho_ = tf.placeholder(tf.float64, shape=(G, C), name="rho_")

    sample_idx = tf.placeholder(tf.int32, shape = (None), name = "sample_idx")

    # basis_means_fixed gets min of array Y and max of array Y and interpolates a list of len B between them
    basis_means_fixed = np.linspace(np.amin(Y),np.amax(Y),B)
    basis_means = tf.constant(basis_means_fixed, dtype=tf.float64)

    b_init = 2 * ((basis_means_fixed[1] - basis_means_fixed[0])**2)

    LOWER_BOUND = 1e-10

    # Variables

    ## Shrinkage prior on delta
    if shrinkage:
        delta_log_mean = tf.Variable(0, dtype=tf.float64)
        delta_log_variance = tf.Variable(1, dtype=tf.float64)
    
    ## Regular variables
    delta_log = tf.Variable(
        tf.random_uniform(
         (G,C), minval=-2, maxval=2, seed=random_seed, dtype=tf.float64
        ),
        dtype=tf.float64,
        constraint=lambda x: tf.clip_by_value(x, tf.constant(math.log(min_delta), dtype=tf.float64), tf.constant(math.inf, dtype=tf.float64))
    )


    # add these colmeans of Y (which is shape (row N, col G) as a col onto a matrix of zeros
    # so basically
    # [beta_init_col(transposed to col) empty_mat] where rows are G and cols are P-1
    beta_0_init = _scale(_colmeans(Y))
    zeroed_arr = np.zeros((G, P-1))
    beta_init = np.column_stack((beta_0_init, zeroed_arr))
    beta = tf.Variable(tf.constant(beta_init, dtype=tf.float64), dtype=tf.float64)

    theta_logit = tf.Variable(tf.random_normal([C], mean=0, stddev=1, seed=random_seed, dtype=tf.float64), dtype=tf.float64)

    ## Spline variables
    a = tf.exp(tf.Variable(tf.zeros([B], dtype=tf.float64)))
    negative_log_init = -math.log(b_init)
    b = tf.exp(tf.constant(np.full(B, negative_log_init), dtype=tf.float64))

    # Stop gradient for irrelevant entries of delta_log
    delta_log = _entry_stop_gradients(delta_log, tf.cast(rho_, tf.bool))

    # Transformed variables
    delta = tf.exp(delta_log)
    theta_log = tf.nn.log_softmax(theta_logit)

    # Model likelihood
    base_mean = tf.transpose(tf.einsum('np,gp->gn'), X_, beta) + tf.log(s_))
    base_mean_list = [ base_mean for x in range(C) ]

    mu_ngc = tf.add(tf.stack(base_mean_list, axis=2), tf.multiply(delta, rho_), name="adding_base_mean_to_delta_rho")

    # reshape
    mu_cng = tf.transpose(mu_ngc, perm=(2,0,1))

    mu_cngb = tf.tile(tf.expand_dims(mu_cng, axis=3), (1,1,1,B))
    
    phi_cng = tf.reduce_sum(a * tf.exp(-b * tf.square(mu_cngb - basis_means)), 3) + LOWER_BOUND

    phi = tf.transpose(phi_cng, perm=(1,2,0))

    mu_ngc = tf.exp(tf.transpose(mu_cng, perm=(1,2,0)))

    p = mu_ngc / (mu_ngc + phi)

    nb_pdf = tfd.NegativeBinomial(probs=p, total_count = phi)