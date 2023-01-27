from python_speech_features import mfcc, logfbank
from torch import tensor, nn, ones, onnx, functional as F, transpose
import scipy.io.wavfile as wav
from SED import SpeechEncDec
from QED import QueryEncDec


sed = SpeechEncDec()
qed = QueryEncDec()
# onnx.export(sed, \
#     tensor([[float(i+j) for i in range(5)] for j in range(13)]),\
#         "module.onnx")
# sed.forward([[[float(i+j)] for i in range(13)] for j in range(5)])


(rate,sig) = wav.read("PropellerEngine.wav")
mfcc_feat = mfcc(sig,rate) # -> array[n][13]
mfcc_feat = [[float(i) for i in row] for row in mfcc_feat]
# fbank_feat = logfbank(sig, rate)
mfcc_feat = transpose(tensor(mfcc_feat), 0, 1)
print("Speech EncDec")
S = sed.forward(mfcc_feat)
# X = sed.forward(tensor([[float(i+j) for i in range(5)] for j in range(13)]))
print("Query EncDec")
Q = qed.forward("keyword")

