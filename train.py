#!/usr/bin/env python
import matplotlib
import warnings
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np
import argparse
import ConfigParser
import visdom

from models import MLP, MLPBlock
from utils import (resample_rows_per_column,
                   get_max_corr,
                   get_max_corr_perm,
                   plot_signals,
                   plot_signals_single,
                   get_random_batch)
from datasets import synthetic, audio
from collections import OrderedDict


def get_parser():
    # Creates a parser which is initialized with the settings in a
    # configuration file if specified.
    conf_parser = argparse.ArgumentParser(
        description=__doc__, # printed with -h/--help
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False
        )
    conf_parser.add_argument("-c", "--conf_file",
                        help="Specify config file", metavar="FILE")
    args, remaining_argv = conf_parser.parse_known_args()

    defaults = {}

    if args.conf_file:
        config = ConfigParser.SafeConfigParser()
        config.read([args.conf_file])
        defaults.update(dict(config.items("Training")))

    # Parse rest of arguments
    # Don't suppress add_help here so it will handle -h
    parser = argparse.ArgumentParser(
        # Inherit options from config_parser
        parents=[conf_parser]
        )

    parser = argparse.ArgumentParser(description='Train a model for source\
            separation', parents=[conf_parser])

    parser.add_argument('--gan_type', default='default',
                        help='Type of GAN objective\
                                (default, wgan-gp, gan-gp, kl[not recommended])')
    parser.add_argument('--prior', default='anica',
                        help='Type of prior\
                            (anica, gaussian, uniform, trainable)')
    parser.add_argument('--no-visdom', dest='visdom', action='store_false')
    parser.set_defaults(visdom=True)
    parser.add_argument('--normalize', dest='normalize', action='store_true')
    parser.add_argument('--normalize_rec', dest='normalize_rec',
            action='store_true')
    parser.set_defaults(normalize_rec=False)
    parser.add_argument('--no-normalize',
                        dest='normalize',
                        action='store_false')
    parser.set_defaults(normalize=True)
    parser.add_argument('--backprop', dest='backprop', action='store_true')
    parser.add_argument('--no-backprop', dest='backprop', action='store_false')
    parser.set_defaults(backprop=True)
    parser.add_argument('--blind', dest='blind', action='store_true',
        help="Don't use the groundtruth for evaluation.")
    parser.set_defaults(blind=False)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--iterations', default=500000, type=int)
    parser.add_argument('--mix_dim', default=6, type=int)
    parser.add_argument('--n_validation', default=500, type=int)
    parser.add_argument('--source_dim', default=6, type=int,
                        help='default 6. Should be 3 for audio data.')
    parser.add_argument('--plot_dim', default=None, type=int,
                        help='default is same as source signals.')
    parser.add_argument('--plot_truncate', default=None, type=int,
                        help='Only plot the first n timesteps.')
    parser.add_argument('--hidden_dim', default=32, type=int,
                        help='nr of hidden units for seperator and mixer.\
                                Default is 32')
    parser.add_argument('--task', default='linear',
                        help='Task type (linear, pnl, mlp)')
    parser.add_argument('--data', default='synthetic',
                        help='Data to use (synthetic, audio)')
    parser.add_argument('--learning_rate', default=.001, type=float)
    parser.add_argument('--gp_scaling', default=.01, type=float)
    parser.add_argument('--sep_stddev', default=.1, type=float)
    parser.add_argument('--disc_stddev', default=.1, type=float)
    parser.add_argument('--prior_stddev', default=.1, type=float)
    parser.add_argument('--mix_stddev', default=.1, type=float)
    parser.add_argument('--rec_scaling', default=1.0, type=float,
                        help='Scaling of the reconstruction cost')
    parser.add_argument('--mixer_type', default='linear',
                        help='Type of mixer/decoder (linear, pnl, mlp')
    parser.add_argument('--vd_server', default='localhost',
                        help='visdom server location')
    parser.add_argument('--vd_port', default=8097, type=int,
                        help='visdom port (default 8097)')
    parser.add_argument('--vd_env', default='main',
                        help='visdom environment name (default "main")')
    parser.add_argument('--separator_type', default='linear',
                        help='Type of separator/encoder (linear, pnl, mlp')
    parser.add_argument('--ind_scaling', default=.1, type=float,
                        help='Scaling of the independence cost')
    parser.add_argument('--disc_updates', default=4, type=int,
                        help='Number of discriminator updates for each\
                                separator update. Default is 4')
    parser.add_argument('--folder', default=None, help='predictions folder')
    parser.add_argument('--results_file', default=None, help='results file')

    parser.set_defaults(**defaults)
    return parser


def main(args, silent_mode=False):
    if not silent_mode:
        print 'Using the following settings:'
        for arg, value in args.__dict__.items():
            print arg, ':', value

    # hyperparameters/settings
    optimizer = tf.train.RMSPropOptimizer
    #optimizer = tf.train.AdamOptimizer
    BMLP_ACTIVATION = tf.nn.relu
    EPS = 1e-3
    hidden_dim = args.hidden_dim
    source_dim = args.source_dim
    input_dim = args.mix_dim

    if args.visdom:
        vis = visdom.Visdom(server=args.vd_server,
                            port=args.vd_port,
                            env='main')

    if not args.backprop and not silent_mode:
        print 'Not backpropagaging through product distribution'

    plot_size = args.plot_truncate

    #################### get the data ####################
    if args.data == 'synthetic':
        all_x, all_y, A = synthetic.get_data(seed=101,
                                                 task_type=args.task,
                                                 mix_dim=input_dim)
        val_x = all_x[:, :args.n_validation]
        val_y = all_y[:, :args.n_validation]
        train_x = all_x[:, args.n_validation:]
        train_y = all_y[:, args.n_validation:]
        plot_size = 500
    elif args.data == 'audio':
        linear_mix, pnl_mix, A, sources = audio.get_data()
        if args.task == 'linear':
            all_y = linear_mix
        elif args.task == 'pnl':
            all_y = pnl_mix
        else:
            raise ValueError('task not supported for data set')
        all_x = sources
        val_x = all_x[:, :args.n_validation]
        val_y = all_y[:, :args.n_validation]
        train_x = all_x[:, args.n_validation:]
        train_y = all_y[:, args.n_validation:]

    if args.blind:
        train_x = train_y.copy()
    ######################################################

    # construct the parts or the graph which contain trainable parameters
    with tf.variable_scope('separator'):
        if args.separator_type == 'linear':
            separator = MLP([input_dim, source_dim], [None],
                    stddev=args.sep_stddev)
        elif args.separator_type == 'pnl':
            linear_separator = MLP([input_dim, source_dim], [None],
                    stddev=args.sep_stddev)
            in_block = MLPBlock(input_dim, 32, n_layers=2,
                    stddev=args.sep_stddev)
            def separator(x):
                return linear_separator(in_block(x, activation=BMLP_ACTIVATION))
        elif args.separator_type == 'mlp':
            separator = MLP([input_dim, hidden_dim, hidden_dim, source_dim],
                            [tf.nn.relu, tf.nn.relu, None],
                            stddev=args.sep_stddev)
                
        if args.mixer_type == 'linear':
            mixer = MLP([source_dim, input_dim], [None], stddev=args.mix_stddev)
        elif args.mixer_type == 'pnl':
            linear_mixer = MLP([source_dim, input_dim], [None],
                    stddev=args.mix_stddev)
            out_block = MLPBlock(input_dim, 16, n_layers=2, bias_value=0.0,
                    stddev=args.mix_stddev)
            def mixer(x):
                return out_block(linear_mixer(x), activation=BMLP_ACTIVATION)
        else:
            mixer = MLP([source_dim, hidden_dim, hidden_dim, input_dim],
                        [tf.nn.relu, tf.nn.relu, None],
                        stddev=args.mix_stddev)

        if args.prior == 'trainable':
            prior_bmlp = MLPBlock(source_dim, 32, n_layers=2,
                    stddev=args.prior_stddev)

        if args.normalize:
            initial_gamma = tf.constant(.1, shape=(source_dim,))
            gamma = tf.Variable(initial_gamma, name='gamma')
            initial_beta = tf.constant(0.0, shape=(source_dim,))
            beta = tf.Variable(initial_beta, name='beta')

    with tf.variable_scope('discriminator'):
        if args.task == 'mlp':
            discriminator = MLP([source_dim, hidden_dim, hidden_dim, 1],
                                [tf.nn.relu, tf.nn.relu, None],
                                stddev=args.disc_stddev)
        else:
            discriminator = MLP([source_dim, 64, 1],
                                [tf.nn.relu, None],
                                stddev=args.disc_stddev)

    sep_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                 scope='separator')
    disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                  scope='discriminator')

    y = tf.placeholder(tf.float32, shape=[None, input_dim])
    if not args.blind:
        x = tf.placeholder(tf.float32, shape=[None, source_dim])

    prediction = separator(y)

    prediction_processed = prediction

    if args.normalize:
        prediction_mean = tf.reduce_mean(prediction, 0)
        # note that we don't want to use -= here.
        prediction_processed = prediction_processed - prediction_mean
        prediction_sd = tf.sqrt(tf.reduce_mean(prediction_processed**2, 0) + EPS)
        prediction_processed /= prediction_sd

    if args.prior == 'anica':
        prediction_perm = resample_rows_per_column(prediction_processed)
    elif args.prior == 'gaussian':
        prediction_perm = tf.random_normal(tf.shape(prediction))
    elif args.prior == 'uniform':
        prediction_perm = tf.random_uniform(tf.shape(prediction))
    elif args.prior == 'trainable':
        prior_samp = tf.random_normal(tf.shape(prediction))
        prediction_perm = prior_bmlp(prior_samp, activation=BMLP_ACTIVATION)
        if args.normalize:
            prediction_perm_mean = tf.reduce_mean(prediction_perm, 0)
            prediction_perm_norm = prediction_perm - prediction_perm_mean
            prediction_perm_sd = tf.sqrt(tf.reduce_mean(prediction_perm_norm**2, 0) + EPS)
            prediction_perm_norm /= prediction_perm_sd
            prediction_perm = prediction_perm_norm

    else:
        raise ValueError("Unknown 'prior'")

    joint_logit = discriminator(prediction_processed)
    marg_logit = discriminator(prediction_perm)

    if args.gan_type == 'default':
        disc_cost = (tf.reduce_mean(tf.nn.softplus(-marg_logit)) +
                     tf.reduce_mean(tf.nn.softplus(joint_logit)))
        if args.backprop:
            gen_cost = -disc_cost
        else:
            gen_cost = -tf.reduce_mean(tf.nn.softplus(joint_logit))
    elif args.gan_type == 'kl':
        disc_cost = (tf.reduce_mean(tf.nn.softplus(-marg_logit)) +
                     tf.reduce_mean(tf.nn.softplus(joint_logit)))
        # there is no grad wrt the marginals by definition for this loss
        gen_cost = -tf.reduce_mean(joint_logit)
    elif args.gan_type == 'bgan':
        disc_cost = (tf.reduce_mean(tf.nn.softplus(-marg_logit)) +
                     tf.reduce_mean(tf.nn.softplus(joint_logit)))
        if args.backprop:
            gen_cost = (tf.reduce_mean(marg_logit**2) +
                        tf.reduce_mean(joint_logit**2))
        else:
            gen_cost = tf.reduce_mean(joint_logit**2)
    elif args.gan_type == 'wgan-gp':
        joint_term = tf.reduce_mean(joint_logit)
        marg_term = tf.reduce_mean(marg_logit)
        disc_cost_mon = joint_term - marg_term
        if args.backprop:
            gen_cost = -disc_cost_mon
        else:
            gen_cost = -joint_term
        # compute gradient penalty
        alpha = tf.random_uniform(shape=(tf.shape(prediction)[0], 1))
        interpolates = alpha * (prediction_perm - prediction_processed)
        interpolates += prediction_processed
        gradients = tf.gradients(discriminator(interpolates),
                                 [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients),
                         reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        disc_cost = disc_cost_mon + args.gp_scaling * gradient_penalty
    elif args.gan_type == 'gan-gp':
        # same cost as default but with gradient penalty
        disc_cost_mon = (tf.reduce_mean(tf.nn.softplus(-marg_logit)) +
                         tf.reduce_mean(tf.nn.softplus(joint_logit)))
        if args.backprop:
            gen_cost = -disc_cost_mon
        else:
            gen_cost = -tf.reduce_mean(tf.nn.softplus(joint_logit))
        gradients_joint = tf.gradients(joint_logit, [prediction_processed])[0]
        gradients_marg = tf.gradients(marg_logit, [prediction_perm])[0]
        ss_joint = tf.reduce_sum(tf.square(gradients_joint),
                                 reduction_indices=[1])
        ss_marg = tf.reduce_sum(tf.square(gradients_marg),
                                reduction_indices=[1])
        gp_marg = tf.reduce_mean(ss_marg * (1 - tf.nn.sigmoid(marg_logit))**2)
        gp_joint = tf.reduce_mean(ss_joint * tf.nn.sigmoid(joint_logit)**2)
        disc_cost = disc_cost_mon + args.gp_scaling * (gp_joint + gp_marg)
    else:
        raise ValueError('Unknown GAN type')

    prediction_norm = prediction - tf.reduce_mean(prediction, 0,
                                                  keep_dims=True)
    cov_mat = (tf.matmul(tf.transpose(prediction_norm), prediction_norm) /
               tf.cast(tf.shape(prediction)[0], prediction.dtype))

    # This computes the average absolute value of the correlation matrix.
    # It can be an interesting value to monitor to see if the model is at least
    # able to remove the linear dependencies.
    diag = tf.diag_part(cov_mat)
    cor_mat = cov_mat / tf.sqrt(diag[:, None] * diag[None, :])
    total_corr = ((tf.reduce_sum(tf.abs(cor_mat)) -
                 tf.cast(source_dim, 'float32')) /
                 (source_dim * (source_dim - 1)))
    
    if args.normalize_rec:
        reconstruction = mixer(prediction_processed * gamma + beta)
    else:
        reconstruction = mixer(prediction)

    rec_cost = tf.abs(reconstruction - y)
    rec_cost = tf.reduce_mean(rec_cost)

    tot_cost = args.rec_scaling * rec_cost + args.ind_scaling * gen_cost

    train_step_sep = optimizer(args.learning_rate).minimize(tot_cost,
            var_list=sep_vars)
    train_step_disc = optimizer(args.learning_rate).minimize(disc_cost,
            var_list=disc_vars)
    max_corr = get_max_corr(x, prediction)

    summary_vars = OrderedDict({'total_corr': total_corr,
                                 'total_cost': tot_cost,
                                 'gen_cost': gen_cost,
                                 'disc_cost': disc_cost,
                                 'rec_cost': rec_cost})
    if not args.blind:
        summary_vars['max_corr'] = max_corr

    if args.gan_type == 'wgan-gp':
        summary_vars['disc_cost_mon'] = disc_cost_mon
                    

    # intialize session
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)


    def fig2rgb_array(fig, expand=True):
        fig.canvas.draw()
        buf = fig.canvas.tostring_rgb()
        ncols, nrows = fig.canvas.get_width_height()
        shape = (nrows, ncols, 3) if not expand else (1, nrows, ncols, 3)
        return np.fromstring(buf, dtype=np.uint8).reshape(shape)

    prediction_np = sess.run(prediction, feed_dict={y: train_y.T})

    if args.plot_dim is None:
        num_signals = source_dim
    else:
        num_signals = args.plot_dim

    if args.blind:
        plot_fig = plot_signals(plt, prediction_np.T[:, :plot_size],
                                prediction_np.T[:, :plot_size],
                                n=num_signals)
    else:
        plot_fig = plot_signals(plt, train_x[:, :plot_size],
                                prediction_np.T[:, :plot_size],
                                n=num_signals)

    if args.folder is not None and not silent_mode:
        if not os.path.isdir(args.folder):
            os.makedirs(args.folder)
        print 'Saving logs to:', args.folder

    summary_lists = OrderedDict((key, []) for key in summary_vars)

    iteration_indices = []

    def fig2rgb_array(fig, expand=True):
        fig.canvas.draw()
        buf = fig.canvas.tostring_rgb()
        ncols, nrows = fig.canvas.get_width_height()
        shape = (nrows, ncols, 3) if not expand else (1, nrows, ncols, 3)
        return np.fromstring(buf, dtype=np.uint8).reshape(shape)

    if args.source_dim > 7 and not args.blind:
        warnings.warn('Sourcedim > 7. Using approximate corr evaluation.')

    for i in range(args.iterations):
        if i % 1000 == 0:
            feed_dict = {y: val_y.T}
            if not args.blind:
                feed_dict[x] = val_x.T
            summary = sess.run(summary_vars.values(),
                               feed_dict=feed_dict)

            prediction_np = sess.run(prediction,
                                     feed_dict={y: train_y.T})

            if np.isnan(prediction_np[0, 0]):
                if silent_mode:
                    return np.nan
                else:
                    raise ValueError('NAN!')

            plt.gcf().clear()
            if args.blind:
                plot_fig = plot_signals(plt, prediction_np.T[:, :plot_size],
                                        prediction_np.T[:, :plot_size],
                                        n=num_signals)
            else:
                plot_fig = plot_signals(plt, train_x[:, :plot_size],
                                        prediction_np.T[:, :plot_size],
                                        n=num_signals)

            fig_rgb = fig2rgb_array(plot_fig.gcf(), expand=False)
            if args.visdom and not silent_mode:
                vis.image(fig_rgb.transpose(2, 0, 1),
                          win='predictions',
                          env=args.vd_env)

            if args.folder is not None:
                np.save(os.path.join(args.folder, 'output' + str(i) + '.npy'),
                        prediction_np)

            iteration_indices.append(i)
            for summ_val, summ_name in zip(summary, summary_vars.keys()):
                if summ_name == 'max_corr' and not args.blind:
                    if args.source_dim < 8:
                        max_corr_np = get_max_corr_perm(prediction_np,
                                                        train_x.T)
                    else:
                        max_corr_np = summ_val
                    summary_lists['max_corr'].append(max_corr_np)
                else:
                    if summ_name == 'total_cost':
                        total_cost_np = summ_val
                    summary_lists[summ_name].append(summ_val)
                if args.visdom and not silent_mode:
                    vis.line(Y=np.asarray(summary_lists[summ_name]),
                             X=np.asarray(iteration_indices),
                             win=summ_name,
                             env=args.vd_env,
                             opts=dict(title=summ_name))

            if not args.blind and not silent_mode:
                print i, 'Current max corr:',  max_corr_np

        train_y_batch = get_random_batch(train_y.T, args.batch_size)
        train_step_sep.run(feed_dict={y: train_y_batch},
                           session=sess)

        for j in range(args.disc_updates):
            train_y_batch = get_random_batch(train_y.T, args.batch_size)
            train_step_disc.run(feed_dict={y: train_y_batch}, session=sess)

    # store final result somewhere in home folder together with the config
    if args.results_file is not None and not silent_mode:
        with open(args.results_file, 'w') as fout:
            fout.write(str(total_cost_np))
            if not args.blind:
                fout.write(' ' + str(max_corr_np) + '\n')
            else:
                fout.write('\n')
            for arg, value in args.__dict__.items():
                fout.write('{} : {}\n'.format(arg, value))

    return total_cost_np


if __name__ == '__main__':
    parser = get_parser()
    main(parser.parse_args())
