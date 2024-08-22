import numpy as np
from scipy import signal
# import torchaudio
import torch

def filter(freq,data):
    fs=250
    filter_data=[]
    for bank in freq:
        parameter = signal.butter(N=5, Wn=bank, btype='bandpass', fs=fs)
        EEG_filtered = signal.lfilter(parameter[0], parameter[1], data)
        filter_data.append(EEG_filtered)
    numpy_array=np.array(filter_data)
    EEG = np.concatenate(numpy_array, axis=1).astype(np.float32)
    return EEG

