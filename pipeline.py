import numpy as np
from pyannote.audio.pipelines import MultiLabelSegmentation as MLS_P
from pyannote.audio.utils.metric import MacroAverageFMeasure

class CowAudioPipelineManager:
    def __init__(self, model):
        self.model = model
        self.pipeline = None

    def setup_pipeline(self, initial_params, fscore=False, share_min_duration=False, use_auth_token=None, **inference_kwargs):
        self.pipeline = MLS_P(segmentation=self.model, fscore=fscore, share_min_duration=share_min_duration, use_auth_token=use_auth_token, **inference_kwargs).instantiate(initial_params)

    def run_segmentation(self, audio_file):
        return self.pipeline(audio_file)

    def evaluate_performance(self, protocol, classes):
        metric = MacroAverageFMeasure(classes=classes)
        for test_file in protocol.test():
            reference = test_file['annotation']
            hypothesis = self.pipeline(test_file)
            metric(reference, hypothesis)
        return abs(metric)
