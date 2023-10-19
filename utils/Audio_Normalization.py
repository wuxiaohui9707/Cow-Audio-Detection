import numpy as np
from pyAudioAnalysis import audioBasicIO as aIO 

def audio_normalization(audio_path):
    fs, s = aIO .read_audio_file(audio_path)
    data_type = s.dtype
    if data_type == np.int16:
        bit_depth = 16
    elif data_type == np.int32:
        bit_depth = 32
    normalized_audio = s / (2**(bit_depth))
    return normalized_audio, fs, s