import numpy as np
import pylab as pl
from scipy.io import wavfile


def get_data():
    """The data files come from the TSP dataset 16k set"""
    _, man = wavfile.read('./wav_data/MA02_04.wav')
    rate, woman = wavfile.read('./wav_data/FA01_03.wav')
    man = man.astype('float32')
    woman = woman.astype('float32')
    man_max = np.max(man)
    woman_max = np.max(woman)
    man /= man_max
    woman /= woman_max
    shortest = min(len(man), len(woman))
    woman = woman[:shortest]
    man = man[:shortest]
    np.random.seed(101)
    noise = np.random.uniform(-1, 1, len(man))
    sources = np.stack((woman, man, noise))
    A = np.random.uniform(-1, 1, (3, 3))
    linear_mix = np.dot(A, sources)
    pnl_mix = linear_mix.copy()
    pnl_mix[0] = np.tanh(pnl_mix[0])
    pnl_mix[1] = (pnl_mix[1] + pnl_mix[1]**3) / 2
    pnl_mix[2] = np.exp(pnl_mix[2])
    return linear_mix, pnl_mix, A, sources


if __name__ == '__main__':
    linear_mix, pnl_mix, A, sources = get_data()
    wavfile.write('./mixtape1.wav', rate=rate, data=linear_mix[0])
    wavfile.write('./mixtape2.wav', rate=rate, data=linear_mix[1])
    wavfile.write('./mixtape3.wav', rate=rate, data=linear_mix[2])
    wavfile.write('./pnlmixtape1.wav', rate=rate, data=pnl_mix[0])
    wavfile.write('./pnlmixtape2.wav', rate=rate, data=pnl_mix[1])
    wavfile.write('./pnlmixtape3.wav', rate=rate, data=pnl_mix[2])
    pl.subplot(311)
    pl.plot(man)
    pl.subplot(312)
    pl.plot(woman)
    pl.subplot(313)
    pl.plot(noise)
    pl.show()

