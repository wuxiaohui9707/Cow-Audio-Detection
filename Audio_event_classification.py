from utils import prepare
from pyannote.database import get_protocol, FileFinder
from pyannote.audio import Model, Inference
from pyannote.audio.tasks import MultiLabelSegmentation as MLS_T
from pyannote.audio.pipelines import MultiLabelSegmentation as MLS_P
from utils import Probability as Prob
import pytorch_lightning as pl
import os

#load datasets and get protocol
my_database = os.path.join('/mnt/', 'e', 'Files', 'Acoustic_Data', 'Datasets_combination', 'yaml','My_Databases.yml')
prepare.database(my_database)
cow_audio = get_protocol('My_datasets.Segmentation.Classification', 
                         preprocessors={"audio": FileFinder()})

#load pretrained model
model = Model.from_pretrained("pyannote/segmentation")
model.specifications.classes = ["Rumination", "Breath","Burp"]#

#load task
task = MLS_T(
    cow_audio, duration=1.5, batch_size=32,
    classes= ["rumination", "Breath","Burp"]#
    )
model.task = task
model.to('cuda')

#train model
output_directory = os.path.join('/mnt/', 'e', 'Files', 'Acoustic_Data', 'Datasets_combination')
trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=100,default_root_dir=output_directory)
trainer.fit(model)

#get test_file_1's probability 
test_iter = cow_audio.test()
first_audio_file = next(test_iter)
inference_1=Inference(model,duration=1.5,step=1)(first_audio_file)
inference_1
Prob.save_probabilities(model,first_audio_file,60,0,60,1.5,1,'probability_1')

#load pipeline 
pipeline = MLS_P(segmentation=model)
initial_params = {
    "thresholds": {
        "rumination": {"onset": 0.7, "offset": 0.6, "min_duration_on": 0.0, "min_duration_off": 0.0},
        "Breath": {"onset": 0.7, "offset": 0.6, "min_duration_on": 0.0, "min_duration_off": 0.0},
        #"Grazing": {"onset": 0.5, "offset": 0.4, "min_duration_on":0.0, "min_duration_off": 0.0},
        "Burp": {"onset": 0.7, "offset": 0.6, "min_duration_on": 0.0, "min_duration_off": 0.0},
        #"gulp": {"onset": 0.8, "offset": 0.6, "min_duration_on": 0.005, "min_duration_off": 0.005},
    }
}

#get test_file_1's timeline and annotation
segmentation_1= pipeline.instantiate(initial_params)(first_audio_file)
segmentation_1
first_audio_file["annotation"]

#test_file_2
second_audio_file = next(test_iter)
inference_2=Inference(model,duration=1.5,step=1)(second_audio_file)
inference_2
Prob.save_probabilities(model,second_audio_file,23,0,23,1.5,1,'probability_2')
segmentation_2 = pipeline.instantiate(initial_params)(second_audio_file)
segmentation_2
second_audio_file["annotation"]