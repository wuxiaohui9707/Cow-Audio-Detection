from typing import Optional
import torch
import torch.nn as nn
from pyannote.audio import Model
from pyannote.audio.core.task import Task, Resolution
from torchaudio.transforms import MFCC

class SoundEventDetection(Model):

    def __init__(
        self,
        sample_rate: int = 16000, 
        num_channels: int = 1, 
        task: Optional[Task] = None,
        param1: int = 32,
        param2: int = 16,
    ):

        super().__init__(sample_rate=sample_rate, 
                         num_channels=num_channels, 
                         task=task)

        self.save_hyperparameters("param1", "param2")
        self.mfcc = MFCC()
        self.linear1 = nn.Linear(self.mfcc.n_mfcc, self.hparams.param1)
        self.linear2 = nn.Linear(self.hparams.param1, self.hparams.param2)

    def build(self):    
        num_classes = len(self.specifications.classes)
        self.classifier = nn.Linear(self.hparams.param2, num_classes)

        self.activation = self.default_activation()

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        mfcc = self.mfcc(waveforms).squeeze(dim=1).transpose(1, 2)
        output = self.linear1(mfcc)
        output = self.linear2(output)

        if self.specifications.resolution == Resolution.CHUNK:
            output = torch.mean(output, dim=-1)
        elif self.specifications.resolution == Resolution.FRAME:
            pass
        
        output = self.classifier(output)
        return self.activation(output)   