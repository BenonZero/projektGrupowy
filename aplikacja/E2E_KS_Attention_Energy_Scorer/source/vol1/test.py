from python_speech_features import mfcc, logfbank
from torch import tensor, nn, ones, onnx, functional as F, transpose, randn
import scipy.io.wavfile as wav
from SED import SpeechEncDec
from QED import QueryEncDec
# from Att import AttMech
# from Energy import EneSc

sed = SpeechEncDec()
qed = QueryEncDec()

sed.train(True)
qed.train(True)

# (rate,sig) = wav.read("E2E_KS_Attention_Energy_Scorer\source\PropellerEngine.wav")
(rate,sig) = wav.read("C:\\Users\\kubak\\INF_SEM5\\PG\\repo\\projektGrupowy\\E2E_KS_Attention_Energy_Scorer\\source\\PropellerEngine.wav")
mfcc_feat = mfcc(sig,rate) # -> array[n][13]
mfcc_feat = [[float(i) for i in row] for row in mfcc_feat]
# fbank_feat = logfbank(sig, rate)
mfcc_feat = transpose(tensor(mfcc_feat), 0, 1)

mfcc_feat = randn(13, 99)

S = sed(mfcc_feat, training = True)
Q = qed("birdie", training = True)