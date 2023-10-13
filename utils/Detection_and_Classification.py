from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import audioSegmentation as aS
from . import Split_Audio as SA
from pyAudioAnalysis import audioTrainTest as aT
import os

def detection_and_classification(audio_file,output_folder,model_name,model_type,smooth_window = 0.1, weight = 0.05):
    [fs, s] = aIO.read_audio_file(audio_file )
    segments = aS.silence_removal(s, fs, 0.05, 0.05, smooth_window, weight, plot = True)
    print(segments)
    SA.split_audio_using_segments(audio_file,segments,output_folder)
    files_to_test = [os.path.join(output_folder, file) for file in os.listdir(output_folder) if file.endswith(".wav")]
    for f in files_to_test:
        print(f'{f}:')
        c, p, p_nam = aT.file_classification(f, model_name,model_type)
        print(f'P({p_nam[0]}={p[0]})')
        print(f'P({p_nam[1]}={p[1]})')
        print(f'P({p_nam[2]}={p[2]})')
        print()