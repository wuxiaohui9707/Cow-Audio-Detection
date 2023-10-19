import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def MFCC_Coefficient_plot(audio,mfcc_num):
    s, fs = librosa.load(audio, sr=None)
    duration = librosa.get_duration(y=s, sr=fs) 
    mfccs = librosa.feature.mfcc(y=s, sr=fs, n_mfcc=mfcc_num)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time', cmap='viridis',x_coords=np.linspace(0, duration, mfccs.shape[1]))
    plt.colorbar()
    plt.xlabel('Time')
    plt.ylabel('MFCC Coefficient')
    plt.show()