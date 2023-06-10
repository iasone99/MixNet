import librosa
import torch
import torchaudio
from scipy.io.wavfile import write
from torchaudio.utils import download_asset

file_path='C:/Users/nicol/Downloads/DNS/train-clean/LibriSpeech/train-clean-100/103/1240/103-1240-0015.flac'

wav, in_sr = librosa.load(file_path)
write("./Original.wav", 22050, wav)
wav = torch.from_numpy(wav).float()
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
snr_dbs = torch.tensor([0])
resampler=torchaudio.transforms.Resample(int(in_sr),16000,dtype=wav.dtype)
#wav=resampler(wav)
wav=torchaudio.functional.resample(wav,int(in_sr),16000)
write("./downsampled.wav", 16000, wav.numpy())
noise_add = noise[:, : wav.shape[0]]

waveform_noise = torchaudio.functional.add_noise(waveform=wav.unsqueeze(0), noise=noise_add, snr=snr_dbs)
wav=waveform_noise.squeeze(0).numpy()
write("./noisy.wav", 22050, wav)
