import numpy as np
from pyannote.audio import Inference

class CowAudioInference:
    def __init__(self, model):
        self.model = model

    def run_inference(self, audio_file, window="sliding", duration=None, 
                             step=None, pre_aggregation_hook=None, 
                             skip_aggregation=False, skip_conversion=False, 
                             batch_size=32, use_auth_token=None):
        """
        运行自定义推理过程。

        :param audio_file: 要进行推理的音频文件。
        :param window: 使用的窗口类型，"sliding" 或 "whole"。
        :param duration: 每个块的持续时间。
        :param step: 块之间的步长。
        :param pre_aggregation_hook: 聚合前的钩子函数。
        :param skip_aggregation: 是否跳过聚合步骤。
        :param skip_conversion: 是否跳过转换步骤。
        :param batch_size: 批处理大小。
        :param use_auth_token: 使用授权令牌（如果有）。
        :return: 推理结果。
        """
        # 初始化 Inference 对象
        inference = Inference(
            model=self.model,
            window=window,
            duration=duration,
            step=step,
            pre_aggregation_hook=pre_aggregation_hook,
            skip_aggregation=skip_aggregation,
            skip_conversion=skip_conversion,
            batch_size=batch_size,
            use_auth_token=use_auth_token
        )

        # 执行推理
        return inference(audio_file)

    def save_probabilities(self, audio_file, audio_duration, start_time, end_time, window_duration, step, file_name):
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
        probability = Inference(self.model, duration=window_duration, step=step)(audio_file)
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
