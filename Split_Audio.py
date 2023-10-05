from pydub import AudioSegment
import os

def split_audio_using_segments(audio_file, segments_file, output_folder):
    # 加载音频文件
    audio = AudioSegment.from_wav(audio_file)
    
    with open(segments_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            parts = line.strip().split()
            start_time = float(parts[0])
            end_time = float(parts[1])
            label = parts[2] 
            segment = audio[start_time * 1000:end_time * 1000]  # pydub 使用毫秒作为单位
            segment.export(os.path.join(output_folder, f"segment_{i}.wav"), format="wav")