import create_chunks, mixCNNCh1
import shiftMel
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
import numpy as np
import hyperparams as hp
from torchaudio.utils import download_asset

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
    This function evaluates the samples in the predict_path according to the IntelNDS metrics SI-SNR and DNSMOS as well as
    MSE loss of the mel specs
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

    # for mel spec to waveform conversion
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

    # dnsmos as evaluation metric
    dnsmos = DNSMOS()

    num_frames = hp.num_frames
    chunk_size = hp.chunk_size
    num_chunks = int(2 * num_frames / chunk_size)
    model = mixCNNCh1.MixCNNCh1(hidden_size=hp.hidden_size_DNN, num_layers=hp.layers_DNN,
                                input_len=2 * hp.num_frames * hp.num_mels,
                                output_len=int(num_frames / chunk_size) * num_chunks,
                                num_chunks_in=int(2 * num_frames / chunk_size),
                                num_chunks_out=int(num_frames / chunk_size)).to(hp.device)

    # load model
    state_dict = t.load('./models/checkpoint_%s_%d.pth.tar' % ("MixCNNCh2Loss2", 100))
    model.load_state_dict(state_dict['model'])
    model = model.to(hp.device)
    model.eval()
    # TTS
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=False)

    # store the metrics for later averaging
    SNRs_joint_griffin_to_target = []
    SNRs_joint_griffin_to_target_griffin = []
    SNRs_tts_griffin_to_target = []
    SNRs_tts_griffin_to_target_griffin = []
    SNR_tts_no_griffin_to_target = []
    SNR_noisy_griffin_to_target_griffin = []
    SNR_noisy_griffin_to_target = []
    SNR_noisy_to_target = []

    mel_tts_loss = []
    mel_model_loss = []
    mel_noise_loss = []

    mel_tts_sisnr = []
    mel_model_sisnr = []
    mel_noise_sisnr = []

    DNS_joint_griffin = []
    DNS_tts_griffin = []
    DNS_tts = []
    DNS_noisy = []
    DNS_noisy_griffin = []
    DNS_clean = []
    DNS_clean_griffin = []

    # MSE loss as evaluation metric
    loss = torch.nn.MSELoss()

    for i, data in enumerate(data_loader):

        embeds_path, file_path, text = data
        file_path = " ".join([x for x in file_path])
        embeds_path = " ".join([x for x in embeds_path])
        text = " ".join([x for x in text])
        wav, in_sr = librosa.load(file_path)
        wav = torch.from_numpy(wav)
        wav = torchaudio.functional.resample(wav, in_sr, hp.sr, lowpass_filter_width=6)

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
        ######### TTS begin
        words = text.split()
        random.randint(0, len(dictionary) - 1)
        for j in range(len(words)):
            if random.randint(0, 9) == 9:
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

        mel_appnd = hp.pad_value*torch.ones(melspec_noise.size(0), melspec_noise.size(1), num_frames - pad_len_tts).to(hp.device)
        melspec_tts = (torch.cat((mel_appnd, melspec_tts), dim=2))
        mel_appnd = hp.pad_value*torch.ones(melspec_noise.size(0), melspec_noise.size(1), num_frames - pad_len_noise).to(hp.device)
        melspec_noise = (torch.cat((mel_appnd, melspec_noise), dim=2))

        pad_len_clean = melspec_target.size(2)
        mel_appnd = hp.pad_value*torch.ones(melspec_noise.size(0), melspec_noise.size(1), num_frames - pad_len_clean).to(hp.device)
        melspec_target = (torch.cat((mel_appnd, melspec_target), dim=2))

        melspec_tts = shiftMel.shiftMel(melspec_tts.unsqueeze(1), 2, 20).squeeze(1)

        # concatenate tts and noisy melspec along the time domain
        mel = torch.cat((melspec_tts, melspec_noise), 2)
        mel = torch.permute(mel, (0, 2, 1))  # [1,T,F]

        mel = mel.to(hp.device)
        melspec_target = melspec_target.to(hp.device)

        with t.no_grad():
            mask = model(mel)  # forward pass
            # normalize the mask: ensure that the chunk weights sum to one for each frame
            norm_factor = torch.sum(mask, dim=1).unsqueeze(2).permute(0, 2, 1)
            mask = mask / norm_factor
            mask[mask != mask] = 0
            mel = torch.permute(mel, (0, 2, 1))  # [1,F,T]

            if chunk_size == 1:  # num_chunks=num_frames: each frame can be mixed with each other frame->simple matmul
                mel_pred = torch.matmul(mel, mask)
            else:  # the mask maps chunks of the mel->special function necessary
                mel_pred = create_chunks.join_chunks(mel, chunk_size, mask, int(num_frames / chunk_size))

            # zero columns are automatically mapped to zero
            non_empty_mask = mel[:, :, num_frames:].abs().sum(dim=1).bool()
            mel_pred = torch.permute(mel_pred, (0, 2, 1))
            mel_pred = mel_pred[non_empty_mask, :].unsqueeze(0)
            mel_pred = torch.permute(mel_pred, (0, 2, 1))

        # cut off all padded areas
        melspec_tts = torch.permute(melspec_tts, (0, 2, 1))
        melspec_tts = melspec_tts[non_empty_mask, :].unsqueeze(0)
        melspec_tts = torch.permute(melspec_tts, (0, 2, 1))

        melspec_noise = torch.permute(melspec_noise, (0, 2, 1))
        melspec_noise = melspec_noise[non_empty_mask, :].unsqueeze(0)
        melspec_noise = torch.permute(melspec_noise, (0, 2, 1))  # [B,F,T]

        # append the loss values
        mel_tts_loss.append(loss(melspec_target, melspec_tts).unsqueeze(0))
        mel_model_loss.append(loss(melspec_target, mel_pred).unsqueeze(0))
        mel_noise_loss.append((loss(melspec_target, melspec_noise)).unsqueeze(0))

        mel_tts_sisnr.append(mel_si_snr(melspec_target.squeeze(0), melspec_tts.squeeze(0)).unsqueeze(0))
        mel_model_sisnr.append(mel_si_snr(melspec_target.squeeze(0), mel_pred.squeeze(0)).unsqueeze(0))
        mel_noise_sisnr.append((mel_si_snr(melspec_target.squeeze(0), melspec_noise.squeeze(0))).unsqueeze(0))

        # reshape for vocoder
        mel_pred = mel_pred.squeeze(0)

        # griffin lim vocoder: transform mel spec to waveform
        inv_spec = inv_mel_spectrogram((mel_pred).cpu())
        wav_model = griffin_lim(inv_spec)

        inv_spec = inv_mel_spectrogram((melspec_target.squeeze(0)).cpu())
        wav_griffin = griffin_lim(inv_spec)

        inv_spec = inv_mel_spectrogram((melspec_tts.squeeze(0)).cpu())
        wav_tts_griffin = griffin_lim(inv_spec)

        inv_spec = inv_mel_spectrogram((melspec_noise.squeeze(0)).cpu())
        wav_noisy_griffin = griffin_lim(inv_spec)

        SNRs_joint_griffin_to_target.append(si_snr(wav, torch.from_numpy(strech_signal(wav, wav_model.numpy()))))
        SNRs_joint_griffin_to_target_griffin.append(si_snr(wav_griffin.squeeze(0), wav_model))
        SNRs_tts_griffin_to_target.append(
            si_snr(wav, torch.from_numpy(strech_signal(wav, wav_tts_griffin.squeeze(0).numpy()))))
        SNRs_tts_griffin_to_target_griffin.append(si_snr(wav_griffin.squeeze(0), wav_tts_griffin.squeeze(0)))
        SNR_tts_no_griffin_to_target.append(si_snr(wav, wav_tts))
        SNR_noisy_griffin_to_target_griffin.append(si_snr(wav_griffin.squeeze(0), wav_noisy_griffin.squeeze(0)))
        SNR_noisy_griffin_to_target.append(
            si_snr(wav, torch.from_numpy(strech_signal(wav, wav_noisy_griffin.squeeze(0).numpy()))))
        SNR_noisy_to_target.append(si_snr(wav, waveform_noise))

        DNS_joint_griffin.append(torch.from_numpy(dnsmos(wav_model.numpy())).unsqueeze(1))
        DNS_tts.append(torch.from_numpy(dnsmos(wav_tts.numpy())).unsqueeze(1))
        DNS_tts_griffin.append(torch.from_numpy(dnsmos(wav_tts_griffin.squeeze(0).numpy())).unsqueeze(1))
        DNS_noisy_griffin.append(torch.from_numpy(dnsmos(wav_noisy_griffin.squeeze(0).numpy())).unsqueeze(1))
        DNS_noisy.append(torch.from_numpy(dnsmos(waveform_noise.squeeze(0).numpy())).unsqueeze(1))
        DNS_clean.append(torch.from_numpy(dnsmos(wav.numpy())).unsqueeze(1))
        DNS_clean_griffin.append(torch.from_numpy(dnsmos(wav_griffin.squeeze(0).numpy())).unsqueeze(1))

        SNR_before = si_snr(wav_griffin.squeeze(0), wav_tts_griffin.squeeze(0))
        SNR_after = si_snr(wav_griffin.squeeze(0), wav_model)
        print("SNR_after: " + str(SNR_after) + " vs SNR_before: " + str(SNR_before))
        quality_before = dnsmos(wav_tts.numpy())
        quality_lstm = dnsmos(wav_model.numpy())
        print("MOS_after: " + str(quality_lstm) + " vs MOS_before: " + str(quality_before))

    #### Print results

    print("SI-SNR joint griffin to target: " + str(torch.mean(torch.stack(SNRs_joint_griffin_to_target))))
    print(
        "SI-SNR joint griffin to target griffin: " + str(torch.mean(torch.stack(SNRs_joint_griffin_to_target_griffin))))
    print("SI-SNR tts griffin to target: " + str(torch.mean(torch.stack(SNRs_tts_griffin_to_target))))
    print("SI-SNR tts griffin to target griffin: " + str(torch.mean(torch.stack(SNRs_tts_griffin_to_target_griffin))))
    print("SI-SNR tts to target: " + str(torch.mean(torch.stack(SNR_tts_no_griffin_to_target))))
    print(
        "SI-SNR noisy griffin to target griffin: " + str(torch.mean(torch.stack(SNR_noisy_griffin_to_target_griffin))))
    print("SI-SNR noisy griffin to target: " + str(torch.mean(torch.stack(SNR_noisy_griffin_to_target))))
    print("SI-SNR noisy to target: " + str(torch.mean(torch.stack(SNR_noisy_to_target))))

    print("MOS joint griffin: " + str(torch.mean(torch.cat(DNS_joint_griffin, dim=1), dim=1)))
    print("MOS tts: " + str(torch.mean(torch.cat(DNS_tts, dim=1), dim=1)))
    print("MOS tts griffin: " + str(torch.mean(torch.cat(DNS_tts_griffin, dim=1), dim=1)))
    print("MOS noisy griffin: " + str(torch.mean(torch.cat(DNS_noisy_griffin, dim=1), dim=1)))
    print("MOS noisy: " + str(torch.mean(torch.cat(DNS_noisy, dim=1), dim=1)))
    print("MOS clean griffin: " + str(torch.mean(torch.cat(DNS_clean_griffin, dim=1), dim=1)))
    print("MOS clean: " + str(torch.mean(torch.cat(DNS_clean, dim=1), dim=1)))

    print("Mel Model loss: " + str(torch.mean(torch.cat(mel_model_loss, dim=0))))
    print("Mel Noise loss: " + str(torch.mean(torch.cat(mel_noise_loss, dim=0))))
    print("Mel TTS loss: " + str(torch.mean(torch.cat(mel_tts_loss, dim=0))))

    print("Mel Model sisnr: " + str(torch.mean(torch.cat(mel_model_sisnr, dim=0))))
    print("Mel Noise sisnr: " + str(torch.mean(torch.cat(mel_noise_sisnr, dim=0))))
    print("Mel TTS sisnr: " + str(torch.mean(torch.cat(mel_tts_sisnr, dim=0))))


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


def mel_si_snr(target: Union[torch.tensor, np.ndarray],
               estimate: Union[torch.tensor, np.ndarray]) -> torch.tensor:
    """Calculates SI-SNR estiamte from target audio and estiamte audio. The
    audio sequene is expected to be a tensor/array of dimension more than 1.
    The last dimension is interpreted as time.
    The implementation is based on the example here:
    https://www.tutorialexample.com/wp-content/uploads/2021/12/SI-SNR-definition.png
    Parameters
    ----------
    target : Union[torch.tensor, np.ndarray]
        Target audio melspec.
    estimate : Union[torch.tensor, np.ndarray]
        Estimate audio melspec.
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
    s_target = target - torch.mean(target, dim=0, keepdim=True)
    s_estimate = estimate - torch.mean(estimate, dim=0, keepdim=True)
    s_target = torch.flatten(s_target)
    s_estimate = torch.flatten(s_estimate)

    # <s, s'> / ||s||**2 * s
    pair_wise_dot = torch.sum(s_target * s_estimate, dim=-1, keepdim=True)
    s_target_norm = torch.sum(s_target ** 2, dim=-1, keepdim=True)
    pair_wise_proj = pair_wise_dot * s_target / s_target_norm

    e_noise = s_estimate - pair_wise_proj

    pair_wise_sdr = torch.sum(pair_wise_proj ** 2,
                              dim=-1) / (torch.sum(e_noise ** 2,
                                                   dim=-1) + EPS)
    return 10 * torch.log10(pair_wise_sdr + EPS)


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
