#!/usr/bin/env python

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav

(rate,sig) = wav.read("english.wav")

fbank_feat = logfbank(sig,dither=0)

mfcc_feat = mfcc(sig,dither=0,useEnergy=False,wintype='povey')

