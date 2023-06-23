import matplotlib
from matplotlib import pyplot as plt
from dnsmos import DNSMOS
import random
from TTS.api import TTS
import torch as t
import torchaudio
import torchaudio.transforms as T

import librosa

from DataLoader.data import get_batch_loader
from DataLoader.data_loaders import TextLoader
from DataLoader.padder import get_padders
from DataLoader.pipelines import get_pipelines
from DataLoader.tokenizer import CharTokenizer
from typing import Union
import torchaudio.functional
import mixCNNCh1
import numpy as np
import hyperparams as hp
from torchaudio.utils import download_asset
from scipy.io.wavfile import write
import shiftMel

from DataLoader.args import (
    get_args,
    get_aud_args,
    get_data_args,
)

import os
import torch


def get_tokenizer(args):
    tokenizer = CharTokenizer()
    tokenizer_path = args.tokenizer_path
    if args.tokenizer_path is not None:
        tokenizer.load_tokenizer(tokenizer_path)
        return tokenizer
    data = TextLoader(args.test_path).load().split('\n')
    data = list(map(lambda x: (x.split(args.sep))[2], data))

    tokenizer.add_pad_token().add_eos_token()
    tokenizer.set_tokenizer(data)
    tokenizer_path = os.path.join(args.checkpoint_dir, 'tokenizer.json')
    tokenizer.save_tokenizer(tokenizer_path)
    print(f'tokenizer saved to {tokenizer_path}')
    return tokenizer


def main():
    """
    This function performs mixing of the samples from the predict_path text file. It adds noise and performs TTS and
    mixes the resulting mel specs with the pretrained MixNet model. It stores the resulting audios in the samples folder
    """
    args = get_args()

    tokenizer = get_tokenizer(args)
    aud_args = get_aud_args(args)
    data_args = get_data_args(args)

    text_padder, aud_padder = get_padders(0, tokenizer.special_tokens.pad_id)
    audio_pipeline, text_pipeline = get_pipelines(tokenizer, aud_args)

    data_loader = get_batch_loader(
        TextLoader(args.predict_path),
        audio_pipeline,
        text_pipeline,
        aud_padder,
        text_padder,
        **data_args
    )

    mel_spectrogram = T.MelSpectrogram(
        sample_rate=hp.sr,
        n_fft=hp.n_fft,
        win_length=hp.win_length,
        hop_length=hp.hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        n_mels=hp.num_mels,
        mel_scale="htk",
    ).to(hp.device)

    inv_mel_spectrogram = T.InverseMelScale(sample_rate=hp.sr, n_stft=hp.n_fft // 2 + 1, n_mels=hp.num_mels).to(
        hp.device)

    griffin_lim = T.GriffinLim(
        n_fft=hp.n_fft,
    ).to(hp.device)

    # dictionary for similar sounding words
    dictionary = {'DRESS': ['MESS', 'REST', 'GUESS'], 'MIND': ['KIND', 'WIND'], 'STAIRS': ['CARES', 'SQUARE'],
                  'STICK': ['KICK', 'SICK']}
    # dictionary for random words
    dictionary_rand = ['DRESS', 'MESS', 'REST', 'GUESS''MIND', 'KIND', 'WIND' 'STAIRS', 'CARES', 'SQUARE']

    # evaluate DNSMOS
    dnsmos = DNSMOS()

    num_frames = hp.num_frames
    chunk_size = hp.chunk_size
    num_chunks = int(2 * num_frames / chunk_size)
    model = mixCNNCh1.MixCNNCh1(hidden_size=hp.hidden_size_DNN, num_layers=hp.layers_DNN,
                                input_len=2 * hp.num_frames * hp.num_mels,
                                output_len=int(num_frames / chunk_size) * num_chunks,
                                num_chunks_in=int(2 * num_frames / chunk_size),
                                num_chunks_out=int(num_frames / chunk_size)).to(hp.device)

    state_dict = t.load('./models/checkpoint_%s_%d.pth.tar' % ("MixNetLoss1", 100))
    model.load_state_dict(state_dict['model'])
    model = model.to(hp.device)
    model.eval()
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=False)

    for i, data in enumerate(data_loader):

        embeds_path, file_path, text = data
        file_path = " ".join([x for x in file_path])
        embeds_path = " ".join([x for x in embeds_path])
        text = " ".join([x for x in text])
        wav, in_sr = librosa.load(file_path)
        wav = torch.from_numpy(wav).float()
        wav = torchaudio.functional.resample(wav, in_sr, hp.sr, lowpass_filter_width=6)
        write("./samples/clean" + str(i) + ".wav", hp.sr, wav.numpy())

        SAMPLE_NOISE = download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo-8000hz.wav")
        noise, _ = torchaudio.load(SAMPLE_NOISE)
        noise = torch.cat((
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise), 1)
        noise_add = noise[:, : wav.shape[0]]
        snr_dbs = torch.tensor([0])

        waveform_noise = torchaudio.functional.add_noise(waveform=wav.unsqueeze(0), noise=noise_add, snr=snr_dbs)
        torchaudio.save(filepath="./samples/noisy_audio" + str(i) + ".wav", src=waveform_noise,
                        sample_rate=hp.sr)
        words = text.split()
        random.randint(0, len(dictionary) - 1)
        for j in range(len(words)):
            if random.randint(0, 4) == 4:
                replacement = dictionary_rand[random.randrange(len(dictionary_rand))]
                words[j] = replacement
        text = ' '.join(words)

        """
        words = text.split()
        for j in range(len(words)):
            word = words[j]
            replacement = dictionary.get(word)
            if replacement:
                words[j] = random.choice(replacement)
        text = ' '.join(words)
        """

        wav_tts = tts.tts(text, speaker_wav=embeds_path,
                          language="en")
        wav_tts = torch.FloatTensor(wav_tts)
        write("./samples/tts" + str(i) + ".wav", hp.sr, wav_tts.numpy())

        ##################TTS end##################

        # ALIGN THE WAVEFORMS
        shape_model = waveform_noise.squeeze(0).numpy()
        wav_tts = torch.from_numpy(strech_signal(shape_model, wav_tts.numpy().astype(float)).astype(float)).float()

        melspec_target = mel_spectrogram(wav.unsqueeze(0).to(hp.device))
        melspec_noise = mel_spectrogram(waveform_noise.to(hp.device))
        melspec_tts = mel_spectrogram(wav_tts.to(hp.device)).unsqueeze(0)

        # pad to obtain required DNN input size
        pad_len_noise = melspec_noise.size(2)
        pad_len_tts = melspec_tts.size(2)

        mel_appnd = hp.pad_value * torch.ones(melspec_noise.size(0), melspec_noise.size(1), num_frames - pad_len_tts)
        melspec_tts = (torch.cat((mel_appnd, melspec_tts), dim=2))
        mel_appnd = hp.pad_value * torch.ones(melspec_noise.size(0), melspec_noise.size(1), num_frames - pad_len_noise)
        melspec_noise = (torch.cat((mel_appnd, melspec_noise), dim=2))

        pad_len_clean = melspec_target.size(2)
        mel_appnd = hp.pad_value * torch.ones(melspec_noise.size(0), melspec_noise.size(1), num_frames - pad_len_clean)
        melspec_target = (torch.cat((mel_appnd, melspec_target), dim=2))

        melspec_tts = shiftMel.shiftMel(melspec_tts.unsqueeze(1), 2, 20).squeeze(1)

        # concatenate tts and noisy melspec along the time domain
        mel = torch.cat((melspec_tts, melspec_noise), 2)
        mel = torch.permute(mel, (0, 2, 1))  # [1,T,F]
        mel_target = mel_target.to(hp.device)

        mel = mel.to(hp.device)
        melspec_target = melspec_target.to(hp.device)

        with t.no_grad():
            mel_pred = model(mel)  # forward pass
            mel_pred = mel_pred.permute(0, 2, 1) #[1,F,T]
            mel = mel.permute(0, 2, 1) #[1,F,T]
            mel_target = mel_target.permute(0, 2, 1)

            # zero columns are automatically mapped to zero
            non_empty_mask = mel[:, :, num_frames:].abs().sum(dim=1).bool()
            mel_pred = torch.permute(mel_pred, (0, 2, 1))
            mel_pred = mel_pred[non_empty_mask, :].unsqueeze(0)
            mel_pred = torch.permute(mel_pred, (0, 2, 1)) #[1,F,T]
            plot_mel(mel.cpu(), mel_pred.cpu(), melspec_target.cpu())

        mel_pred = mel_pred.squeeze(0)

        melspec_tts = torch.permute(melspec_tts, (0, 2, 1))
        melspec_tts = melspec_tts[non_empty_mask, :].unsqueeze(0)
        melspec_tts = torch.permute(melspec_tts, (0, 2, 1))

        # griffin lim vocoder
        inv_spec = inv_mel_spectrogram((mel_pred).cpu())
        wav_model = griffin_lim(inv_spec)

        inv_spec = inv_mel_spectrogram((melspec_target.squeeze(0)).cpu())
        wav_griffin = griffin_lim(inv_spec)

        inv_spec = inv_mel_spectrogram((melspec_tts.squeeze(0)).cpu())
        wav_tts_griffin = griffin_lim(inv_spec)

        write("./samples/test" + str(i) + ".wav", hp.sr, wav_model.numpy())
        torchaudio.save(filepath="./samples/griffin_clean" + str(i) + ".wav", src=wav_griffin.unsqueeze(0),
                        sample_rate=hp.sr)

        torchaudio.save(filepath="./samples/griffin_tts" + str(i) + ".wav", src=wav_tts_griffin.unsqueeze(0),
                        sample_rate=hp.sr)

        SNR_before = si_snr(wav_griffin.squeeze(0), wav_tts_griffin.squeeze(0))
        SNR_after = si_snr(wav_griffin.squeeze(0), wav_model)
        print("SNR_after: " + str(SNR_after) + " vs SNR_before: " + str(SNR_before))
        quality_before = dnsmos(wav_tts.numpy())
        quality_lstm = dnsmos(wav_model.numpy())
        print("MOS_after: " + str(quality_lstm) + " vs MOS_before: " + str(quality_before))


def si_snr(target: Union[torch.tensor, np.ndarray],
           estimate: Union[torch.tensor, np.ndarray]) -> torch.tensor:
    """Calculates SI-SNR estiamte from target audio and estiamte audio. The
    audio sequene is expected to be a tensor/array of dimension more than 1.
    The last dimension is interpreted as time.
    The implementation is based on the example here:
    https://www.tutorialexample.com/wp-content/uploads/2021/12/SI-SNR-definition.png
    Parameters
    ----------
    target : Union[torch.tensor, np.ndarray]
        Target audio waveform.
    estimate : Union[torch.tensor, np.ndarray]
        Estimate audio waveform.
    Returns
    -------
    torch.tensor
        SI-SNR of each target and estimate pair.
    """
    EPS = 1e-8

    if not torch.is_tensor(target):
        target: torch.tensor = torch.tensor(target)
    if not torch.is_tensor(estimate):
        estimate: torch.tensor = torch.tensor(estimate)

    # zero mean to ensure scale invariance
    s_target = target - torch.mean(target, dim=-1, keepdim=True)
    s_estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)

    # <s, s'> / ||s||**2 * s
    pair_wise_dot = torch.sum(s_target * s_estimate, dim=-1, keepdim=True)
    s_target_norm = torch.sum(s_target ** 2, dim=-1, keepdim=True)
    pair_wise_proj = pair_wise_dot * s_target / s_target_norm

    e_noise = s_estimate - pair_wise_proj

    pair_wise_sdr = torch.sum(pair_wise_proj ** 2,
                              dim=-1) / (torch.sum(e_noise ** 2,
                                                   dim=-1) + EPS)
    return 10 * torch.log10(pair_wise_sdr + EPS)


def plot_mel(mel, mel_pred, mel_target):
    fig, axs = plt.subplots(3)
    axs[0].set_title('Mel_TTS')
    axs[0].set_ylabel('mel freq')
    axs[0].set_xlabel('frame')
    im = axs[0].imshow(librosa.power_to_db(mel[0, :, :].squeeze(0)), origin='lower',
                       aspect='auto')
    fig.colorbar(im, ax=axs[0])
    axs[1].set_title('Mel_pred')
    axs[1].set_ylabel('mel freq')
    axs[1].set_xlabel('frame')
    im = axs[1].imshow(librosa.power_to_db(mel_pred.detach()[0, :, :].squeeze(0)), origin='lower',
                       aspect='auto')
    fig.colorbar(im, ax=axs[1])
    axs[2].set_title('Mel_clean')
    axs[2].set_ylabel('mel freq')
    axs[2].set_xlabel('frame')
    im = axs[2].imshow(librosa.power_to_db(mel_target.detach()[0, :, :].squeeze(0)), origin='lower',
                       aspect='auto')
    fig.colorbar(im, ax=axs[2])
    fig.tight_layout(pad=0.5)
    fig.set_size_inches(18.5, 10.5, forward=True)
    plt.savefig('melEvalDNN.svg')
    matplotlib.pyplot.close()


def strech_signal(reference, input):
    """
    This function stretches the waveform input to the length of the reference waveform
    :param reference: numpy array of shape [N,]
    :param input: numpy array of shape [M,]
    :return: numpy array of shape [N,]
    """
    ref_length = reference.shape[0]
    input_length = input.shape[0]
    factor = input_length / ref_length
    out = librosa.effects.time_stretch(y=input, rate=factor)
    return out


if __name__ == '__main__':
    main()