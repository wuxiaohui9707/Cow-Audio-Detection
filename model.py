import torch
import pytorch_lightning as pl
from pyannote.audio import Model
from pyannote.audio.models.segmentation import PyanNet
from pyannote.audio.core.task import Specifications, Problem, Resolution
from pyannote.audio.tasks import MultiLabelSegmentation as MLS_T

class CowAudioModelManager:
    def __init__(self, model_name="pyannote/segmentation", output_dir=None):
        self.model_name = model_name
        self.output_dir = output_dir
        self.model = None

    def load_original_model(self, sincnet_params, lstm_params, linear_params, sample_rate, num_channels, classes):
        self.model = PyanNet(sincnet=sincnet_params, lstm=lstm_params, linear=linear_params, sample_rate=sample_rate, num_channels=num_channels)
        self.model.specifications = Specifications(classes=classes, problem=Problem.MULTI_LABEL_CLASSIFICATION, resolution=Resolution.FRAME, duration=None, min_duration=None, warm_up=None)

    def load_pretrained_model(self, classes):
        self.model = Model.from_pretrained(self.model_name)
        self.model.specifications = Specifications(classes=classes, problem=Problem.MULTI_LABEL_CLASSIFICATION, resolution=Resolution.FRAME, duration=None, min_duration=None, warm_up=None)

    def train_model(self, protocol, task_params, max_epochs=100):
        self.model.task = MLS_T(protocol=protocol, **task_params)
        self.model.to('cuda')
        trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=max_epochs, default_root_dir=self.output_dir)
        trainer.fit(self.model)

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, model_path, map_location='cuda'):
        self.model.load_state_dict(torch.load(model_path, map_location=map_location))
