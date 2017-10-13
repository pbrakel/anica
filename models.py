import tensorflow as tf
import numpy as np
from utils import lognormdens, weight_variable, bias_variable


class MLP(object):

    def __init__(self, dims, activations, stddev=1., bias_value=0.0):
        self.dims = dims
        self.activations = activations
        self.layers = []
        previous_dim = dims[0]
        for i, dim, activation in zip(xrange(len(activations)),
                                      dims[1:], activations):
            with tf.variable_scope('layer' + str(i)):
                weights = weight_variable((previous_dim, dim),
                                          stddev / np.sqrt(previous_dim))
                if i < len(activations) - 1:
                    biases = bias_variable((dim,), value=bias_value)
                else:
                    biases = bias_variable((dim,), value=0.0)

            self.layers.append((weights, biases, activation))
            previous_dim = dim

    def __call__(self, x, add_bias=True, return_activations=False):
        h = x
        hidden = []
        for weights, biases, activation in self.layers:
            h = tf.matmul(h, weights)
            if add_bias:
                h += biases
            if activation:
                h = activation(h)
            hidden.append(h)
        self.hidden = hidden
        if return_activations:
            return hidden
        else:
            return h

    def get_forward_derivative(self, x, fprimes):
        h = x
        for layer, fprime in zip(self.layers, fprimes):
            weights, biases, activation = layer
            h = tf.matmul(h, weights)
            h *= fprime
        return h


class MLPBlock(object):
    """Applies a separate MLP to each dimension of the input.

    The output dimensionality is assumed to be identical to the input.
    """

    def __init__(self, input_dim, hidden_dim, n_layers=1,
            stddev=1., bias_value=0.0):
        # bias value will only be applied to the hidden layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = []
        with tf.variable_scope('block_mlp'):
            self.w_in_var = weight_variable((input_dim, input_dim * hidden_dim),
                                   stddev / np.sqrt(hidden_dim),
                                   name='w_in')
            self.w_out_var = weight_variable((input_dim * hidden_dim, input_dim),
                                    stddev / np.sqrt(hidden_dim),
                                    name='w_out')
            mask = np.zeros((input_dim, input_dim * hidden_dim),
                            dtype='float32')
            hid_to_hid_mask = np.zeros((input_dim * hidden_dim,
                                        input_dim * hidden_dim),
                                       dtype='float32')
            self.bias_hid = bias_variable((hidden_dim * input_dim,),
                                          value=bias_value,
                                          name='bias_first_hid')
            self.bias_out = bias_variable((input_dim,),
                                          name='bias_out')
            for i, row in enumerate(mask):
                row[i * hidden_dim:(i + 1) * hidden_dim] = 1.0

            for i in range(0, input_dim * hidden_dim, hidden_dim):
                hid_to_hid_mask[i:i + hidden_dim, i:i + hidden_dim] = 1.0

            self.hid_to_hid_mask = tf.convert_to_tensor(hid_to_hid_mask)
            self.in_out_mask = tf.convert_to_tensor(mask)
            self.w_in = self.w_in_var * self.in_out_mask
            self.w_out = self.w_out_var * tf.transpose(self.in_out_mask)
            for i in range(n_layers - 1):
                with tf.variable_scope('layer_' + str(i)):
                    w_hid = weight_variable((input_dim * hidden_dim,
                                             input_dim * hidden_dim),
                                             stddev / np.sqrt(hidden_dim))
                    b_hid = bias_variable((hidden_dim * input_dim,),
                                          value=bias_value)
                    self.hidden_layers.append((w_hid * self.hid_to_hid_mask,
                                               b_hid))

    def __call__(self, y, **kwargs):
        return self.forward(y, **kwargs)

    def forward(self, y, activation=None):
        h = tf.matmul(y, self.w_in) + self.bias_hid
        if activation is not None:
            h = activation(h)
        for w_hid, b_hid in self.hidden_layers:
            h = tf.matmul(h, w_hid) + b_hid
            if activation is not None:
                h = activation(h)
        x = tf.matmul(h, self.w_out) + self.bias_out
        return x


class LinearMISEP(object):
    """Implements the linear version of the MISEP model.

    MISEP, as described in [almeida2003]_,
    is an infomax-based ICA model. A normalized neural network replaces
    the usual sigmoid non-linearity of the infomax model.

    .. [almeida2003]
    Almeida, L. B. (2003).
    *MISEP--Linear and Nonlinear ICA Based on Mutual Information.*
    Journal of Machine Learning Research, 4(Dec), 1297-1318.

    """

    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        with tf.variable_scope('misep'):
            self.F = MLP([input_dim, input_dim], [None])
            # Psi should be contstrained to be non-decreasing.  In the paper
            # this is done by using sigmoid functions and normalizing the norm
            # of the 'output' weights to be 1 / sqrt(h) and ensuring that the
            # weights are positive.  According to the paper, initializing the
            # weights with positive values ensures that they remain positive due
            # to the nature of the objective.  The input matrix should ensure
            # independent Psi functions. An n-diagonal martrix should work.  In
            # practice, tanh is used instead of the logistic sigmoid.
            w_in = tf.random_uniform(shape=(input_dim, input_dim * hidden_dim),
                                     minval=-.1, maxval=.1)
            w_out = tf.random_uniform(shape=(input_dim * hidden_dim, input_dim),
                                      minval=0, maxval=.2)
            self.Psi_weights_in = tf.Variable(w_in, name='Psi_weights_in')
            self.Psi_weights_out = tf.Variable(w_out, name='Psi_weights_out')
            mask = np.zeros((input_dim, input_dim * hidden_dim),
                            dtype='float32')
            self.bias = bias_variable((hidden_dim * input_dim,))
        for i, row in enumerate(mask):
            row[i * hidden_dim:(i + 1) * hidden_dim] = 1.0

        self.Psi_mask = tf.convert_to_tensor(mask)
        self.w_in = self.Psi_weights_in * self.Psi_mask
        self.w_out = self.Psi_weights_out * tf.transpose(self.Psi_mask)

    def forward(self, o):
        y = self.F(o)
        h = tf.matmul(y, self.w_in) + self.bias
        h = tf.tanh(h)
        w_out = self.w_out / tf.sqrt(tf.reduce_sum(self.w_out**2, 0,
                                                   keep_dims=True))
        w_out = w_out / tf.sqrt(tf.cast(self.hidden_dim, 'float32'))
        x = tf.matmul(h, w_out)
        return x, h, y

    def compute_jacobian(self, o):
        x, h, _ = self.forward(o) # h is (batch, input_dim*hidden_dim)
        y_prime = self.F(tf.eye(self.input_dim), add_bias=False)
        h_prime = tf.matmul(y_prime, self.w_in)
        # h_prime is now (input_dim, input_dim*hidden_dim)
        h_prime = tf.expand_dims(h_prime, 0)
        h_prime = h_prime * tf.expand_dims(1 - h**2, 1)
        # h_prime should now be (batch, input_dim, input_dim*hidden_dim)
        w_out = self.w_out / tf.sqrt(tf.reduce_sum(self.w_out**2, 0,
                                                   keep_dims=True))
        w_out = w_out / tf.sqrt(tf.cast(self.hidden_dim, 'float32'))
        # J should be (batch, input_dim, input_dim)
        # FIXME: einsum only seems to work when the batch dimension is known.
        J = tf.einsum('aij,jk->aik', h_prime, w_out) 
        return J

    def get_log_det_jacobian(self, o):
        J = self.compute_jacobian(o)
        def step(J_i):
            return tf.log(tf.abs(tf.matrix_determinant(J_i)))
        return tf.map_fn(step, J)

    def get_log_det_jacobian2(self, o):
        # requires tensorflow >= 1.0 but is much faster
        J = self.compute_jacobian(o)
        operator = tf.contrib.linalg.LinearOperatorFullMatrix(J)
        return operator.log_abs_determinant()


class PNLMISEP(object):
    """Implements a PNL version of the MISEP model.

    MISEP, as described in [almeida2003]_,
    is an infomax-based ICA model. A normalized neural network replaces
    the usual sigmoid non-linearity of the infomax model.

    This version uses an architecture which correponds to a learnable
    non-linearity followed by a linear transfmation. Subsequently, there is
    another constrained learnable non-linearity like in the linear MISEP model.
    This architecture is based on [zheng2007]_.

    .. [almeida2003]
    Almeida, L. B. (2003).
    *MISEP--Linear and Nonlinear ICA Based on Mutual Information.*
    Journal of Machine Learning Research, 4(Dec), 1297-1318.

    .. [zheng2007]
    Zheng, C. H., Huang, D. S., Li, K., Irwin, G., & Sun, Z. L. (2007).
    *MISEP method for postnonlinear blind source separation.*
    Neural computation, 19(9), 2557-2578.


    """

    def __init__(self, input_dim, hidden_dim, scaling=1.0, stddev=1.0):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        with tf.variable_scope('misep'):
            # TODO: should just make matrix for this
            self.F = MLP([input_dim, input_dim], [None], stddev=stddev)
            # Psi should be contstrained to be non-decreasing.  In the paper
            # this is done by using sigmoid functions and normalizing the norm
            # of the 'output' weights to be 1 / sqrt(h) and ensuring that the
            # weights are positive.  According to the paper, initializing the
            # weights with positive values ensures that they remain positive due
            # to the nature of the objective.  The input matrix should ensure
            # independent Psi functions. An n-diagonal martrix should work.  In
            # practice, tanh is used instead of the logistic sigmoid.
            entrance_w_in = tf.random_uniform(shape=(input_dim,
                                                     input_dim * hidden_dim),
                                              minval=-.1, maxval=.1) * scaling
            entrance_w_out = tf.random_uniform(shape=(input_dim * hidden_dim,
                                                      input_dim),
                                               minval=0, maxval=.2) * scaling
            exit_w_in = tf.random_uniform(shape=(input_dim,
                input_dim * hidden_dim), minval=-.1, maxval=.1) * scaling
            exit_w_out = tf.random_uniform(shape=(input_dim * hidden_dim,
                                                  input_dim),
                                           minval=0, maxval=.2) * scaling
            self.entrance_weights_in = tf.Variable(entrance_w_in,
                                                   name='entrance_weights_in')
            self.entrance_weights_out = tf.Variable(entrance_w_out,
                                                    name='entrance_weights_out')
            self.exit_weights_in = tf.Variable(exit_w_in,
                                               name='exit_weights_in')
            self.exit_weights_out = tf.Variable(exit_w_out,
                                                name='exit_weights_out')
            mask = np.zeros((input_dim, input_dim * hidden_dim),
                    dtype='float32')
            self.entrance_bias_h = bias_variable((hidden_dim * input_dim,))
            self.entrance_bias_o = bias_variable((input_dim,))
            self.exit_bias_h = bias_variable((hidden_dim * input_dim,))

        for i, row in enumerate(mask):
            row[i * hidden_dim:(i + 1) * hidden_dim] = 1.0

        self.mask = tf.convert_to_tensor(mask)
        self.entrance_w_in = self.entrance_weights_in * self.mask
        self.entrance_w_out = self.entrance_weights_out * tf.transpose(self.mask)
        self.exit_w_in = self.exit_weights_in * self.mask
        self.exit_w_out = self.exit_weights_out * tf.transpose(self.mask)

    def forward(self, o):
        entrance_h = tf.matmul(o, self.entrance_w_in) + self.entrance_bias_h
        entrance_h = tf.tanh(entrance_h)
        entrance_o = (tf.matmul(entrance_h, self.entrance_w_out) +
                      self.entrance_bias_o)
        y = self.F(entrance_o)  # prediction of the model
        exit_h = tf.matmul(y, self.exit_w_in) + self.exit_bias_h
        exit_h = tf.tanh(exit_h)
        w_out = self.exit_w_out / tf.sqrt(tf.reduce_sum(self.exit_w_out**2, 0,
                                                        keep_dims=True))
        w_out = w_out / tf.sqrt(tf.cast(self.hidden_dim, 'float32'))
        x = tf.matmul(exit_h, w_out)
        return x, entrance_h, exit_h, y

    def compute_jacobian(self, o):
        # 90s style manual gradient computations
        x, entrance_h, exit_h, y = self.forward(o)
        entrance_h_prime = tf.matmul(tf.eye(self.input_dim),
                                     self.entrance_w_in)
        # entrance_h_prime is now (input_dim, input_dim*hidden_dim)
        entrance_h_prime = tf.expand_dims(entrance_h_prime, 0)
        entrance_h_prime *= tf.expand_dims(1 - entrance_h**2, 1)
        # entrance_h_prime should be (batch, input_dim, input_dim*hidden_dim)
        entrance_o_prime = tf.einsum('aij,jk->aik', entrance_h_prime,
                self.entrance_w_out) 
        # we're at (batch, input_dim, input_dim)
        y_prime = tf.einsum('aij,jk->aik',
                            entrance_o_prime,
                            self.F.layers[0][0])
        # still at (batch, input_dim, input_dim)
        exit_h_prime = tf.einsum('aij,jk->aik',
                                 y_prime,
                                 self.exit_w_in) 
        # (batch, input_dim, input_dim*hidden_dim)
        # exit_h should be (batch, input_dim*hidden_dim)
        exit_h_prime = exit_h_prime * tf.expand_dims(1 - exit_h**2, 1)
        w_out = self.exit_w_out / tf.sqrt(tf.reduce_sum(self.exit_w_out**2, 0,
                                                   keep_dims=True))
        w_out = w_out / tf.sqrt(tf.cast(self.hidden_dim, 'float32'))
        # J should be (batch, input_dim, input_dim)
        J = tf.einsum('aij,jk->aik', exit_h_prime, w_out) 
        return J

    def get_log_det_jacobian(self, o):
        J = self.compute_jacobian(o)

        def step(J_i):
            return tf.log(tf.abs(tf.matrix_determinant(J_i)))

        return tf.map_fn(step, J)

    def get_log_det_jacobian2(self, o):
        # requires tensorflow >= 1.0 but is much faster
        J = self.compute_jacobian(o)
        operator = tf.contrib.linalg.LinearOperatorFullMatrix(J)
        return operator.log_abs_determinant()
