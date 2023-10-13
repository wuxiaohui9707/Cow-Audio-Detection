from pydub import AudioSegment
import os

def split_audio_using_txt(audio_file, txt_file, output_folder):
    audio = AudioSegment.from_wav(audio_file)
    
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            parts = line.strip().split()
            start_time = float(parts[0])
            end_time = float(parts[1])
            label = parts[2] 
            segment = audio[start_time * 1000:end_time * 1000]  # pydub使用毫秒作为单位
            filename = f"segment_{i}_from_{start_time:.2f}to{end_time:.2f}.wav"
            segment.export(os.path.join(output_folder, filename), format="wav")

def split_audio_using_segments(audio_file, segments_file, output_folder):
    audio = AudioSegment.from_wav(audio_file)
    
    for i, segment_info in enumerate(segments_file):
        start_time = float(segment_info[0])
        end_time = float(segment_info[1])
        segment = audio[start_time * 1000:end_time * 1000]
        filename = f"segment_{i}_from_{start_time:.2f}to{end_time:.2f}.wav"
        segment.export(os.path.join(output_folder, filename), format="wav")
