import torch
import torchaudio.transforms as T
import MixDNN.hyperparams as hp
from dnsmos import DNSMOS
import MixDNN
import numpy as np
from typing import Union
#a=torch.rand(2,3,5)
#print(a)
#print(torch.sum(a,dim=1).unsqueeze(2).permute(0,2,1))
#print(a/torch.sum(a,dim=1).unsqueeze(2))
"""
torch.manual_seed (1414)
t = torch.randn (8, 4)
a = t.argmax (1)
m = torch.zeros (t.shape).scatter (1, a.unsqueeze (1), 1.0)
print ('\n', t, '\n\n', a, '\n\n', m)
print(m.T)"""

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
    s_target=torch.flatten(s_target)
    s_estimate=torch.flatten(s_estimate)

    # <s, s'> / ||s||**2 * s
    pair_wise_dot = torch.sum(s_target * s_estimate, dim=-1, keepdim=True)
    s_target_norm = torch.sum(s_target ** 2, dim=-1, keepdim=True)
    pair_wise_proj = pair_wise_dot * s_target / s_target_norm

    e_noise = s_estimate - pair_wise_proj

    pair_wise_sdr = torch.sum(pair_wise_proj ** 2,
                              dim=-1) / (torch.sum(e_noise ** 2,
                                                   dim=-1) + EPS)
    return 10 * torch.log10(pair_wise_sdr + EPS)

a=torch.rand(800)
griffin_lim = T.GriffinLim(
    n_fft=hp.n_fft,
).to(hp.device)

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

melspec=mel_spectrogram(a)
wav=griffin_lim(inv_mel_spectrogram(melspec))
#print(wav.size())
dnsmos=DNSMOS()
c=dnsmos(a.numpy())
b=torch.from_numpy(c).unsqueeze(1)

d=torch.rand(64,800)
e=torch.rand(64,800)
f=mel_si_snr(d,e)

loss=torch.nn.MSELoss()
f=loss(d,e)
d=1

e=torch.flatten(e)
f=torch.count_nonzero(e)
g=1


