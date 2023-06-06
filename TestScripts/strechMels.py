import torchaudio
import matplotlib
from matplotlib import pyplot as plt

import hyperparams as hp
import numpy as np
import torch
from scipy.io.wavfile import write
import librosa
from TTS.api import TTS
import torchaudio.transforms as T


def align_waveforms(reference, input):
    import numpy as np
    import pytsmod as tsm
    import soundfile as sf  # you can use other audio load packages.
    x_length = reference.shape[0]  # length of the audio sequence x.

    s_fixed = x_length / input.shape[0]  # stretch the audio signal
    x_s_fixed = strech_signal(reference, input).numpy()  # librosa.effects.time_stretch(y=input, rate=s_fixed)
    torchaudio.save(filepath="./samples/stretched.wav", src=torch.from_numpy(x_s_fixed).unsqueeze(0),
                    sample_rate=hp.sr)
    # write("./samples/stretched"+".wav", hp.sr, torch.from_numpy(x_s_fixed).numpy())
    write("./samples/not_stretched" + ".wav", hp.sr, torch.from_numpy(reference).numpy())
    fig, axs = plt.subplots(3)
    axs[0].plot(np.array(reference), 'r')
    axs[1].plot(np.array(input), 'r')
    axs[2].plot(np.array(x_s_fixed), 'r')
    plt.savefig("wavs2.svg")


def strech_signal(reference, input):
    ref_length = reference.shape[0]
    input_length = input.shape[0]
    factor = input_length / ref_length
    output = []
    out = librosa.effects.time_stretch(y=input, rate=factor)
    for i in range(input_length):
        appnd = input[i]
        for j in range(int(factor)):
            if (len(output) < ref_length):
                output.append(appnd)
    output = torch.from_numpy(np.asarray(output))
    return out

def plot_mel(mel, mel_pred, mel_target):
    fig, axs = plt.subplots(3)
    axs[0].set_title('Mel_TTS')
    axs[0].set_ylabel('mel freq')
    axs[0].set_xlabel('frame')
    im = axs[0].imshow(librosa.power_to_db(mel[0,:, :]), origin='lower',
                       aspect='auto')
    fig.colorbar(im, ax=axs[0])
    axs[1].set_title('Mel_Noise')
    axs[1].set_ylabel('mel freq')
    axs[1].set_xlabel('frame')
    im = axs[1].imshow(librosa.power_to_db(mel_pred[0,:, :].squeeze(0)), origin='lower',
                       aspect='auto')
    fig.colorbar(im, ax=axs[1])
    axs[2].set_title('Mel_pred')
    axs[2].set_ylabel('mel freq')
    axs[2].set_xlabel('frame')
    im = axs[2].imshow(librosa.power_to_db(mel_target[0,:, :].squeeze(0)), origin='lower',
                       aspect='auto')
    plt.savefig('melStrech.svg')
    matplotlib.pyplot.close()


if __name__ == '__main__':
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=hp.sr,
        n_fft=hp.n_fft,
        win_length=hp.win_length,
        hop_length=hp.hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        onesided=True,
        n_mels=hp.num_mels,
        mel_scale="htk",
    ).to('cpu')

    inv_mel_spectrogram = T.InverseMelScale(sample_rate=hp.sr, n_stft=hp.n_fft // 2 + 1, n_mels=hp.num_mels).to('cpu')

    griffin_lim = T.GriffinLim(
        n_fft=hp.n_fft,
    ).to('cpu')

    x_1, fs = librosa.load(
        'C:/Users/nicol/Downloads/DNS/train-clean/LibriSpeech/train-clean-100/1034/121119/1034-121119-0024.flac')
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=False)

    x_2 = tts.tts("AND YET HIS MANNERS SHOWED THE EFFECTS OF A SORT OF EDUCATION IF NOT A COMPLETE ONE",
                  speaker_wav="C:/Users/nicol/Downloads/DNS/train-clean/LibriSpeech/train-clean-100/1034/121119/1034-121119-0024.flac",
                  language="en")
    x_2 = np.asarray(x_2).astype(float)
    x_1=x_1.astype(float)
    wavs_clean = torch.from_numpy(x_1.astype(float))
    wavs_tts = torch.from_numpy(x_2.astype(float))
    out = strech_signal(x_1, x_2).astype(float)

    melspec_out=mel_spectrogram(torch.from_numpy(out).unsqueeze(0).float())
    melspec_in=mel_spectrogram(torch.from_numpy(x_2).unsqueeze(0).float())
    melspec_ref=mel_spectrogram(torch.from_numpy(x_1).unsqueeze(0).float())
    # inv_spec = inv_mel_spectrogram((melspec).cpu())
    # out = griffin_lim(inv_spec)

    out = torch.from_numpy(out).unsqueeze(0).float()

    #torchaudio.save(file_path="./samples/stretched" + ".wav", src=out, sample_rate=hp.sr)

    #write("./samples/not_stretched" + ".wav", hp.sr, x_1)
    #write("./samples/stretched" + ".wav", hp.sr, out)
    fig, axs = plt.subplots(3)
    axs[0].plot(np.array(x_1), 'r')
    axs[1].plot(np.array(x_2), 'r')
    axs[2].plot(np.array(out), 'r')
    plt.savefig("wavs2.svg")

    plot_mel(melspec_ref, melspec_in, melspec_out)
    print(melspec_out.size())
    print(melspec_ref.size())
