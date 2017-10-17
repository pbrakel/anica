# Adversarial Non-linear Independent Component Analysis

This is an implementation of the work descriped in the ICML 2017 workshop on implicit models titled
"Maximizing Independence with GANs for Non-linear ICA"

ArXiv version (with a slightly different title): [https://arxiv.org/abs/1710.05050]

### Dependencies

- numpy, tensorflow, visdom

### Optional dependencies:
- matplotlib for plotting results
- scipy for reading the audio data
- the audio data itself

### Installation

Clone the repository and you should be good to go.

### Training a model

First, start a visdom server using `python -m visdom.server` for visualizing the results.

Next, run `python train.py -c ./examples/gan_mlp_example.conf --vd_server=http://127.0.0.1`
to train a model with the settings from one of the example configurations.
The settings in the configuration files can be overridden using the command line.

`python train.py -h` will print the available command line and configuration options.

The folder `./examples/best` contains the hyper-parameters found using a random search.

### License

MIT
