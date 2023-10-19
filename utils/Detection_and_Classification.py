from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import audioSegmentation as aS
from . import Split_Audio as SA
from pyAudioAnalysis import audioTrainTest as aT
import os
import shutil

def detection_and_classification(audio_file,output_folder,model_name,model_type,smooth_window = 0.1, weight = 0.05):
    [fs, s] = aIO.read_audio_file(audio_file )# get audio signal and sample rate
    segments = aS.silence_removal(s, fs, 0.05, 0.05, smooth_window, weight, plot = True)
    #print(segments)
    SA.split_audio_using_segments(audio_file,segments,output_folder)#split audio according to segments
    files_to_test = [os.path.join(output_folder, file) for file in os.listdir(output_folder) if file.endswith(".wav")]
    for f in files_to_test:
        print(f'{f}:')
        c, p, p_nam = aT.file_classification(f, model_name, model_type)

        # 使用循环动态地打印每个种类的可能性
        for i in range(len(p_nam)):
            print(f'P({p_nam[i]}={p[i]})')
        
        # 获取可能性最大的种类的索引
        max_prob_index = p.argmax()
        max_prob_category = p_nam[max_prob_index]
        
        # 为每个种类创建文件夹（如果还没有的话）
        category_folder = os.path.join(output_folder, max_prob_category)
        if not os.path.exists(category_folder):
            os.makedirs(category_folder)
        
        # 将音频文件移动到对应的种类文件夹
        shutil.move(f, os.path.join(category_folder, os.path.basename(f)))
        
        print()