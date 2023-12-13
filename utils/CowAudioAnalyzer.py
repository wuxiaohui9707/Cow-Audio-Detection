import os, torch
from pyannote.database import get_protocol, FileFinder, registry
from pyannote.audio import Model, Inference
from pyannote.audio.core.task import Resolution, Problem, Specifications
from pyannote.audio.models.segmentation import PyanNet
from pyannote.audio.tasks import MultiLabelSegmentation as MLS_T
from pyannote.audio.pipelines import MultiLabelSegmentation as MLS_P
import numpy as np
import pytorch_lightning as pl

class CowAudioAnalyzer:
    def __init__(self, database_path, output_dir, model_name="pyannote/segmentation"):
        self.database_path = database_path
        self.output_dir = output_dir
        self.model_name = model_name
        self.classes = None
        self.model = None
        self.pipeline = None
        self.protocol = None

    def load_dataset(self):
        registry.load_database(self.database_path)
        os.environ["PYANNOTE_DATABASE_CONFIG"] = self.database_path
        self.protocol = get_protocol('My_datasets.Segmentation.Classification', 
                                     preprocessors={"audio": FileFinder()})

    def load_original_model(self, sincnet_params=None, lstm_params=None, linear_params=None, sample_rate=16000, num_channels=1, classes=None):
        self.model = PyanNet(sincnet=sincnet_params, 
                             lstm=lstm_params, 
                             linear=linear_params, 
                             sample_rate=sample_rate, 
                             num_channels=num_channels)
        self.classes = classes
        self.model.specifications = Specifications(
            classes=classes,
            problem=Problem.MULTI_LABEL_CLASSIFICATION,
            resolution=Resolution.FRAME,
            duration=None,
            min_duration=None,
            warm_up=None
        )

    def load_pretrained_model(self,classes):
        self.classes = classes
        self.model = Model.from_pretrained(self.model_name)
        self.model.specifications = Specifications(
            classes=classes,
            problem=Problem.MULTI_LABEL_CLASSIFICATION,
            resolution=Resolution.FRAME,
            duration=None,
            min_duration=None,
            warm_up=None
        )

    def setup_task(self, duration=2.0, warm_up=0.0, balance=None, weight=None, 
                   batch_size=32, num_workers=None, pin_memory=False, 
                   augmentation=None, metric=None):
        # 创建 MultiLabelSegmentation 任务实例
        self.task = MLS_T(
            protocol=self.protocol,
            classes=self.classes, 
            duration=duration,
            warm_up=warm_up,
            balance=balance,
            weight=weight,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            augmentation=augmentation,
            metric=metric
        )
        self.model.task = self.task
        self.model.to('cuda')

    def train_model(self, max_epochs=100):
        # Make sure the model is already loaded
        if self.model is None:
            raise ValueError("Model has not been loaded.")
        
        trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=max_epochs, default_root_dir=self.output_dir)
        trainer.fit(self.model)

    def save_model(self, save_path):
        if self.model is None:
            raise ValueError("No model is available to save.")
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, model_path, map_location='cuda'):
        if self.model is None:
            raise ValueError("Model must be initialized before loading state dict.")
        self.model.load_state_dict(torch.load(model_path, map_location=map_location))

    def continue_training(self, model_path, new_data_protocol, max_epochs=100):
        self.load_model(model_path)  # 加载模型
        self.protocol = new_data_protocol  # 更新协议为新数据
        self.setup_task()  # 重新设置任务
        trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=max_epochs, default_root_dir=self.output_dir)
        trainer.fit(self.model)  # 继续训练

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

    def setup_pipeline(self, initial_params, fscore=False, share_min_duration=False, use_auth_token=None, **inference_kwargs):
        self.pipeline = MLS_P(
            segmentation=self.model,
            fscore=fscore,
            share_min_duration=share_min_duration,
            use_auth_token=use_auth_token,
            **inference_kwargs
        ).instantiate(initial_params)

    def run_segmentation(self, audio_file):
        return self.pipeline(audio_file)