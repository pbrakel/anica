import tensorflow as tf
import numpy as np
from itertools import permutations


def plot_signals(pl, x, y, n=5):
    for i in range(n):
        pl.subplot(n, 2, i * 2 + 1)
        pl.plot(x[i], color='black')
        ymin = np.min(x[i]) - .5
        ymax = np.max(x[i]) + .5
        axes = pl.gca()
        axes.set_ylim([ymin, ymax])
        axes.locator_params(nbins=5)
        pl.subplot(n, 2, i * 2 + 2)
        pl.plot(y[i], color='black')
        axes = pl.gca()
        axes.locator_params(nbins=4, tight=True)
    return pl


def plot_signals_single(pl, x, rescale=True):
    n = x.shape[0]
    for i in range(n):
        pl.subplot(n, 1, i + 1)
        pl.plot(x[i], color='black')
        if rescale:
            ymin = np.min(x[i]) - .5
            ymax = np.max(x[i]) + .5
            axes = pl.gca()
            axes.set_ylim([ymin, ymax])
            axes.locator_params(nbins=5)
    return pl


def lognormdens(x, mu, logsigma):
    numerator = tf.reduce_sum(-.5 * (x - mu)**2 / tf.exp(logsigma * 2), 1)
    denominator = tf.reduce_sum(.5 * tf.log(2 * np.pi) + logsigma, 1)
    return numerator - denominator


def get_random_batch(x, n):
    indices = np.random.randint(x.shape[0], size=(n,))
    return x[indices]


def weight_variable(shape, stddev=0.01, name='weight'):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)


def bias_variable(shape, value=0.0, name='bias'):
    initial = tf.constant(value, shape=shape)
    return tf.Variable(initial, name=name)


def sample_integers(n, shape):
    sample = tf.random_uniform(shape, minval=0, maxval=tf.cast(n, 'float32'))
    sample = tf.cast(sample, 'int32')
    return sample


def resample_rows_per_column(x):
    """Permute all rows for each column independently."""
    n_batch = tf.shape(x)[0]
    n_dim = tf.shape(x)[1]
    row_indices = sample_integers(n_batch, (n_batch * n_dim,))
    col_indices = tf.tile(tf.range(n_dim), [n_batch])
    indices = tf.transpose(tf.stack([row_indices, col_indices]))
    x_perm = tf.gather_nd(x, indices)
    x_perm = tf.reshape(x_perm, (n_batch, n_dim))
    return x_perm


def get_corr_xy(x, y):
    """Find the pairing of signals with the largest correlation."""
    x_centered = x - tf.reduce_mean(x, 0, keep_dims=True)
    y_centered = y - tf.reduce_mean(y, 0, keep_dims=True)
    cov_xy = tf.matmul(x_centered, y_centered, transpose_a=True)
    cov_xy /= tf.cast(tf.shape(x)[0], 'float32')
    sd_x = tf.sqrt(tf.reduce_mean(x_centered**2, 0, keep_dims=True))
    sd_y = tf.sqrt(tf.reduce_mean(y_centered**2, 0, keep_dims=True))
    corr_xy = cov_xy / tf.matmul(sd_x, sd_y, transpose_a=True)
    return corr_xy


def get_max_corr(x, y):
    corr_xy = get_corr_xy(x, y)
    return tf.reduce_mean(tf.reduce_max(tf.abs(corr_xy), 0))


def get_max_corr_np(x, y):
    """This function is only for quick approximate evaluation purposes.

    Use 'get_max_corr_perm' for the actual evaluation.

    """
    corr_xy = get_corr_xy_np(x, y)
    return np.mean(np.max(np.abs(corr_xy), 0))


def get_corr_xy_np(x, y):
    x_centered = x - np.mean(x, 0, keepdims=True)
    y_centered = y - np.mean(y, 0, keepdims=True)
    cov_xy = np.dot(x_centered.T, y_centered) / np.float32(x.shape[0])
    sd_x = np.sqrt(np.mean(x_centered**2, 0, keepdims=True))
    sd_y = np.sqrt(np.mean(y_centered**2, 0, keepdims=True))
    corr_xy = cov_xy / np.dot(sd_x.T, sd_y)
    return corr_xy


def get_max_corr_perm(x, y):
    x_centered = x - np.mean(x, 0, keepdims=True)
    y_centered = y - np.mean(y, 0, keepdims=True)
    sd_x = np.sqrt(np.mean(x_centered**2, 0, keepdims=True))
    x_norm = x_centered / sd_x
    sd_y = np.sqrt(np.mean(y_centered**2, 0, keepdims=True))
    corrs = []
    for x_perm in permutations(x_norm.T):
        x_perm = np.stack(x_perm)
        cov_diag = np.mean(x_perm.T * y_centered, 0) / sd_y
        corrs.append(np.mean(np.abs(cov_diag)))
    return np.max(corrs)
