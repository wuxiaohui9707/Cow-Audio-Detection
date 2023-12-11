import os, torch
from pyannote.database import get_protocol, FileFinder, registry
from pyannote.audio import Model, Inference
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

    def load_pretrained_model(self,classes):
        self.classes = classes
        self.model = Model.from_pretrained(self.model_name)
        self.model.specifications.classes = classes

    def setup_task(self, duration=1.5, batch_size=32):
        task = MLS_T(
            self.protocol, duration=duration, batch_size=batch_size,
            classes=self.classes
        )
        self.model.task = task
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