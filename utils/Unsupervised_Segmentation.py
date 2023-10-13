from pyAudioAnalysis import audioBasicIO as aIO 
from pyAudioAnalysis.MidTermFeatures import mid_feature_extraction as mT
from pyAudioAnalysis.audioSegmentation import labels_to_segments
import scipy.io.wavfile as wavfile
import numpy as np
import IPython
import sklearn.cluster
import os

def Unsupervised_segmentation(input_audio, output_folder,mt_size=1, mt_step=0.5, st_win=0.1, n_clusters=6):
    fs, s = aIO .read_audio_file(input_audio)
    [mt_feats, st_feats, _] = mT(s, fs, mt_size * fs, mt_step * fs,
                            round(fs * st_win), round(fs * st_win * 0.5))
    mt_feats = mt_feats.T
    x_clusters = [np.zeros((fs, )) for i in range(n_clusters)]
    k_means = sklearn.cluster.KMeans(n_clusters=n_clusters)
    k_means.fit(mt_feats)
    cls = k_means.labels_
    segs, c = labels_to_segments(cls, mt_step)  # convert flags to segment limits
    for sp in range(n_clusters):                
        count_cl = 0
        for i in range(len(c)):     # for each segment in each cluster (>2 secs long)
            if c[i] == sp and segs[i, 1]-segs[i, 0] > 2:
                count_cl += 1
                # get the signal and append it to the cluster's signal (followed by some silence)
                cur_x = x[int(segs[i, 0] * fs): int(segs[i, 1] * fs)]
                x_clusters[sp] = np.append(x_clusters[sp], cur_x)
                x_clusters[sp] = np.append(x_clusters[sp], np.zeros((fs,)))
        # write cluster's signal into a WAV file
        output_file_path = os.path.join(output_folder, f'cluster_{sp}.wav')  # 使用os.path.join来生成完整路径
        print(f'cluster {sp}: {count_cl} segments {len(x_clusters[sp]) / float(fs)} sec total dur')
        wavfile.write(output_file_path, fs, np.int16(x_clusters[sp]))
        IPython.display.display(IPython.display.Audio(output_file_path))