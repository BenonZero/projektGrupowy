import torch
import torchaudio as torchaudio
from module import DeepTemplateMatchingModule
from torchaudio import transforms
# torchaudio.set_audio_backend("soundfile")
# print(torchaudio.get_audio_backend())
# waveform, sample_rate = torchaudio.load("test.wav")
# transform = transforms.MelSpectrogram(sample_rate)
# mel_specgram = transform(waveform)
# print(sample_rate)
# print(mel_specgram)

temp = torch.rand(100, 128)
print(temp)
print(temp.shape)

mod = DeepTemplateMatchingModule()
out = mod(temp, temp)
print(out)
