import glob
import librosa

def generate_uem_files(audio_folder, outpuit_folder):
    # 遍历给定文件夹中的所有 .wav 文件
    for wav_file in glob.glob(os.path.join(audio_folder, '*.wav')):
        # 获取文件名（不包含扩展名）
        file_name = os.path.splitext(os.path.basename(wav_file))[0]

        # 使用 librosa 获取音频文件的时长
        duration = librosa.get_duration(filename=wav_file)
        formatted_duration = f"{duration:.3f}"

        # 构造 UEM 文件的内容
        uem_content = f"{file_name} 1 0.000 {formatted_duration}\n"

        # 构造 UEM 文件的完整路径
        uem_file = os.path.join(outpuit_folder, f'{file_name}.uem')

        # 将内容写入 UEM 文件
        with open(uem_file, 'w') as file:
            file.write(uem_content)

        print(f"UEM file created for {file_name}: {uem_file}")