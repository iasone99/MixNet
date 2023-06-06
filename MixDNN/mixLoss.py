import torch
import torch.nn as nn
import hyperparams as hp
import torchaudio.transforms as T
import numpy as np
from typing import Union


class MixLoss(nn.Module):
    """
    This object is the Loss which is optimized by MixNet
    """

    def __init__(self):
        super(MixLoss, self).__init__()

    def forward(self, output, target, mel_noise, mel_tts):
        criterion = nn.MSELoss()
        loss = criterion(output, target)

        reg = criterion(mel_tts, mel_noise) * criterion(output, mel_noise)
        reg2 = 0
        for i in range(output.size(0)):
            reg2 = reg2 + mel_si_snr(target[i, :, :], output[i, :, :])
        # reg = criterion(output, mel_tts)  # penalize taking the noisy input and therefore high noise
        reg3=(100 - reg2/output.size(0))
        return loss + 10*reg + 0.01*reg3


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
