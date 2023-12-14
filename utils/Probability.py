import numpy as np
from pyannote.audio import Inference

def save_probabilities(model, audio_file, audio_duration, start_time, end_time, window_duration, step, file_name):
    """
    计算并保存特定时间段内的概率数据到文本文件。

    :param model: 使用的模型。
    :param audio_file: 音频文件。
    :param audio_duration: 音频文件的时长（秒）。
    :param start_time: 开始时间（秒）。
    :param end_time: 结束时间（秒）。
    :param window_duration: 滑动窗口时长（秒）。
    :param step: 步长（秒）。
    :param file_name: 输出文件名。
    """
    # 创建 Inference 对象并获取概率
    probability = Inference(model, duration=window_duration,step=step)(audio_file)
    window = probability.sliding_window

    # 创建时间数组
    times = np.arange(window.start, audio_duration, window.step) + (window.duration / 2)

    # 截断概率数据
    probability_data = probability.data[:len(times)]

    # 结合时间和概率数据
    combined = np.hstack((times[:, np.newaxis], probability_data))

    # 选择特定时间段的数据
    start_index = int(start_time / window.step)
    end_index = int(end_time / window.step)
    selected_data = combined[start_index:end_index]

    # 格式化结果并保存到文件
    with open(file_name, 'w') as file:
        for row in selected_data:
            formatted_row = ' '.join([f"{x:.3f}" for x in row])
            file.write(formatted_row + '\n')

    print(f"Data saved to {file_name}")