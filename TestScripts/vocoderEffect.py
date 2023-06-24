import librosa
import torch
import torchaudio
from TTS.api import TTS
from torchaudio.utils import download_asset
from typing import Union
import numpy as np
from dnsmos import DNSMOS
import MixNet.hyperparams as hp
import torchaudio.transforms as T

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

# load the audio
wav, in_sr = librosa.load("C:/Users/nicol/Downloads/DNS/train-clean/LibriSpeech/train-clean-100/1502/122619/1502-122619-0024.flac")
wav = torch.from_numpy(wav)
# resample the audio
wav = torchaudio.functional.resample(wav, in_sr, hp.sr, lowpass_filter_width=6)



#add the noise
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
# add the noise
waveform_noise = torchaudio.functional.add_noise(waveform=wav.unsqueeze(0), noise=noise_add, snr=snr_dbs)

#tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=False)
# = tts.tts("Very welcome to my seminar presentation. My name is Jason.", speaker_wav="C:/Users/nicol/Downloads/DNS/mySound.wav", language="en")
#wav_tts = tts.tts_to_file("Very welcome to my seminar presentation. My name is Jason.", speaker_wav="C:/Users/nicol/Downloads/DNS/mySound.wav", language="en",file_path="C:/Users/nicol/Downloads/DNS/welcome.wav")

mel_spectrogram = T.MelSpectrogram(
    sample_rate=hp.sr,
    n_fft=hp.n_fft,
    win_length=hp.win_length,
    hop_length=hp.hop_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
    norm="slaney",
    n_mels=64,
    mel_scale="htk",
).to(hp.device)

mel_clean=mel_spectrogram(wav)
mel_noise=mel_spectrogram(waveform_noise)


# for mel spec to waveform conversion
inv_mel_spectrogram = T.InverseMelScale(sample_rate=hp.sr, n_stft=hp.n_fft // 2 + 1, n_mels=64).to(
    hp.device)

griffin_lim = T.GriffinLim(
    n_fft=hp.n_fft,
).to(hp.device)

inv_spec = inv_mel_spectrogram((mel_noise))
wav_noisy_griffin = griffin_lim(inv_spec)

inv_spec = inv_mel_spectrogram((mel_clean))
wav_clean_griffin = griffin_lim(inv_spec)

print(si_snr(wav_clean_griffin, wav_noisy_griffin))


