import torch
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

matplotlib.use("QtAgg")

from TTS.api import TTS
from network import *

import librosa

import torch.nn.functional as F

import numpy as np
import hyperparams as hp

import librosa.feature
import librosa.display

def align_waveforms(reference, input):
    import pytsmod as tsm
    x_length = reference.shape[-1]  # length of the audio sequence x.

    s_fixed = 1.3  # stretch the audio signal 1.3x times.
    x_s_fixed = tsm.wsola(input, s_fixed)
    librosa.display.waveshow(x_s_fixed, sr=fs)

if __name__ == '__main__':

    x_1, fs = librosa.load('C:/Users/nicol/Downloads/DNS/train-clean/LibriSpeech/train-clean-100/1034/121119/1034-121119-0024.flac')
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=False)

    x_2 = tts.tts("AND YET HIS MANNERS SHOWED THE EFFECTS OF A SORT OF EDUCATION IF NOT A COMPLETE ONE", speaker_wav="C:/Users/nicol/Downloads/DNS/train-clean/LibriSpeech/train-clean-100/1034/121119/1034-121119-0024.flac",
                      language="en")
    x_2=np.asarray(x_2)


    wavs_clean=torch.from_numpy(x_1)
    wavs_tts=torch.from_numpy(x_2)

    len_orig = wavs_clean.size(0)
    len_tts = wavs_tts.size(0)
    if len_tts > len_orig:
        wavs_clean = F.pad(wavs_clean, (
            len_tts - len_orig, 0))  # zero pad to shape all inputs to one output
    if len_tts < len_orig:
        wavs_tts = F.pad(wavs_tts, (
            len_orig - len_tts, 0))  # zero pad to shape all inputs to one output
    x_1=wavs_clean.numpy()
    x_2=wavs_tts.numpy()

    n_fft = hp.n_fft
    hop_size = hp.hop_length

    X = librosa.feature.chroma_cens(y=x_1, sr=hp.sr)
    Y=librosa.feature.chroma_cens(y=x_2, sr=hp.sr)
    D, wp = librosa.sequence.dtw(X, Y, subseq=True, backtrack=True)
    fig1 = plt.figure(figsize=(10, 10))
    ax = fig1.add_subplot(111)
    librosa.display.specshow(D, x_axis='time', y_axis='time',
                             cmap='gray_r', hop_length=hop_size)
    imax = ax.imshow(D, cmap=plt.get_cmap('gray_r'),
                     origin='lower', interpolation='nearest', aspect='auto')
    ax.plot(wp[:, 1], wp[:, 0], marker='o', color='r')
    plt.title('Warping Path on Acc. Cost Matrix $D$')
    plt.colorbar()
    plt.savefig('alignment.svg')

    fig, axs = plt.subplots(2)
    axs[0].plot(np.array(x_1), 'r')
    axs[1].plot(np.array(x_2), 'r')
    plt.savefig("wavs2.svg")

    plt.tight_layout()

    trans_figure = fig.transFigure.inverted()
    lines = []
    arrows = 30
    points_idx = np.int16(np.round(np.linspace(0, wp.shape[0] - 1, arrows)))

    # for tp1, tp2 in zip((wp[points_idx, 0]) * hop_size, (wp[points_idx, 1]) * hop_size):
    for tp1, tp2 in wp[points_idx] * hop_size / fs:
        # get position on axis for a given index-pair
        coord1 = trans_figure.transform(ax1.transData.transform([tp1, 0]))
        coord2 = trans_figure.transform(ax2.transData.transform([tp2, 0]))

        # draw a line
        line = matplotlib.lines.Line2D((coord1[0], coord2[0]),
                                       (coord1[1], coord2[1]),
                                       transform=fig.transFigure,
                                       color='r')
        lines.append(line)

    fig.lines = lines
    plt.savefig('wavs.svg')

    align_waveforms(x_1,x_2)
