import os, torch
from pyannote.database import get_protocol, FileFinder, registry
from pyannote.audio import Model, Inference
from pyannote.audio.core.task import Resolution, Problem, Specifications
from pyannote.audio.models.segmentation import PyanNet
from pyannote.audio.tasks import MultiLabelSegmentation as MLS_T
from pyannote.audio.pipelines import MultiLabelSegmentation as MLS_P
import Probability as Prob
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

    def run_inference(self, audio_file, duration=1.5, step=1):
        inference = Inference(self.model, duration=duration, step=step)
        return inference(audio_file)

    def save_probabilities(self, audio_file, audio_duration,start_time, end_time, window_duration, step, file_name):
        Prob.save_probabilities(self.model, audio_file, audio_duration, start_time, end_time, window_duration, step, file_name)

    def setup_pipeline(self, initial_params):
        self.pipeline = MLS_P(segmentation=self.model).instantiate(initial_params)

    def run_segmentation(self, audio_file):
        return self.pipeline(audio_file)