import numpy as np


def get_data(seed=101, mix_dim=6, task_type='linear', samples=4000):
    # 10kHz, 4000 samples by default
    # This version of the task adds laplacian noise as a source and uses a
    # non-linear partially non-invertible, possibly overdetermined,
    # transformation.
    np.random.seed(seed)
    t = np.linspace(0, samples * 1e-4, samples)
    two_pi = 2 * np.pi
    s0 = np.sign(np.cos(two_pi * 155 * t))
    s1 = np.sin(two_pi * 800 * t)
    s2 = np.sin(two_pi * 300 * t + 6 * np.cos(two_pi * 60 * t))
    s3 = np.sin(two_pi * 90 * t)
    s4 = np.random.uniform(-1, 1, (samples,))
    s5 = np.random.laplace(0, 1, (samples,))
    x = np.stack([s0, s1, s2, s3, s4, s5])
    mix_mat = np.random.uniform(-.5, .5, (mix_dim, 6))
    y = np.dot(mix_mat, x)
    if task_type in ['mlp', 'pnl']:
        y = np.tanh(y)
        if task_type == 'mlp':
            mix_mat2 = np.random.uniform(-.5, .5, (mix_dim, mix_dim))
            y = np.tanh(np.dot(mix_mat2, y))
    return x, y, mix_mat
