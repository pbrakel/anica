#!/usr/bin/env python
import tensorflow as tf
import io
import os
import numpy as np

from models import PNLMISEP
from utils import get_random_batch, get_max_corr_perm
from datasets import synthetic, audio
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='Infomax/MISEP experiment')
    parser.add_argument('--learning_rate', default=.001, type=float)
    parser.add_argument('--data', default='synthetic',
            help='Data to use (synthetic, audio)')
    parser.add_argument('--n_validation', default=500, type=int)
    parser.add_argument('--folder', default='./', help='results folder')
    parser.add_argument('--hidden_dim', default=32, type=int)
    parser.add_argument('--block_weight_scaling', default=1.0, type=float)
    parser.add_argument('--weight_stddev', default=1.0, type=float)
    parser.add_argument('--results_file', default=None, help='results file')
    return parser


def main(args):
    # hyperparameters/settings
    EXP = False
    n_hidden = args.hidden_dim
    batch_size = 256
    source_dim = 6 if args.data == 'synthetic' else 3
    input_dim = source_dim

    with tf.variable_scope('separator'):
        separator = PNLMISEP(input_dim, n_hidden,
                args.block_weight_scaling, args.weight_stddev)

    sep_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                 scope='separator')

    y = tf.placeholder(tf.float32, shape=[None, input_dim])

    prediction = separator.forward(y)[-1]

    prediction_norm = prediction - tf.reduce_mean(prediction, 0, keep_dims=True)
    cov_mat = (tf.matmul(tf.transpose(prediction_norm), prediction_norm) /
               tf.cast(tf.shape(prediction)[0], prediction.dtype))

    tot_cost = -tf.reduce_mean(separator.get_log_det_jacobian2(y))
    optimizer = tf.train.RMSPropOptimizer
    train_step_sep = optimizer(args.learning_rate).minimize(tot_cost,
                                                              var_list=sep_vars)


    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    plot_size = 500
    if args.data == 'synthetic':
        all_x, all_y, A = synthetic.get_data(seed=101,
                                                 task_type='pnl',
                                                 mix_dim=input_dim)
        val_x = all_x[:, :args.n_validation]
        val_y = all_y[:, :args.n_validation]
        train_x = all_x[:, args.n_validation:]
        train_y = all_y[:, args.n_validation:]
    elif args.data == 'audio': 
        linear_mix, pnl_mix, A, sources = audio.get_data()
        all_y = pnl_mix
        all_x = sources
        val_x = all_x[:, :args.n_validation]
        val_y = all_y[:, :args.n_validation]
        train_x = all_x[:, args.n_validation:]
        train_y = all_y[:, args.n_validation:]
        plot_size = None
    else:
        raise ValueError('No data set specified')

    prediction_np = sess.run(prediction,
                             feed_dict={y: train_y.T})

    for i in range(500000):
        if i % 1000 == 0:
            prediction_np = sess.run(prediction,
                                     feed_dict={y: train_y.T})
            if np.isnan(prediction_np[0, 0]):
                    raise ValueError('NAN!')
        train_y_batch = get_random_batch(train_y.T, batch_size)
        train_step_sep.run(feed_dict={y: train_y_batch}, session=sess) 

    tot_cost_np = sess.run(tot_cost,
                           feed_dict={y: val_y.T})
    max_corr_np = max_corr_np = get_max_corr_perm(prediction_np, train_x.T)

    if args.results_file is not None:
        with open(args.results_file, 'w') as fout:
            fout.write(str(tot_cost_np))
            fout.write(' ' + str(max_corr_np) + '\n')
            for arg, value in args.__dict__.items():
                fout.write('{} : {}\n'.format(arg, value))


if __name__ == '__main__':
    parser = get_parser()
    main(parser.parse_args())
