from utils import prepare
from pyannote.database import get_protocol, FileFinder
from pyannote.audio import Model, Inference
from pyannote.audio.tasks import MultiLabelSegmentation as MLS_T
from pyannote.audio.pipelines import MultiLabelSegmentation as MLS_P
import pytorch_lightning as pl
import os

#load database
my_database = os.path.join('/mnt/', 'e', 'Files', 'Acoustic_Data', 'Datasets_specified', 'yaml','My_Databases_specified.yml')
prepare.database(my_database)
cow_audio = get_protocol('My_datasets_specified.Segmentation.Classification', 
                         preprocessors={"audio": FileFinder()})

#load model
model = Model.from_pretrained("pyannote/segmentation")
model.specifications.classes = ["rumination", "hoofbeat", "breath"]
model.to('cuda')

#
audio_file = next(cow_audio.test())
inference_1=Inference(model,duration=2,step=1)(audio_file)
inference_1

#load task
task = MLS_T(
    cow_audio, duration=2.0, batch_size=32,
    classes= ["rumination","hoofbeat","breath"]
    )
model.task = task
inference_2=Inference(model,duration=2,step=1)(audio_file)
inference_2

#train model
output_directory = os.path.join('/mnt/', 'e', 'Files', 'Acoustic_Data', 'Datasets_specified')
trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=10,default_root_dir=output_directory)
trainer.fit(model)

inference_3=Inference(model,duration=2,step=1)(audio_file)
inference_3

#load pipeline 
pipeline = MLS_P(segmentation=model)
initial_params = {
    "thresholds": {
        "rumination": {"onset": 0.3, "offset": 0.2, "min_duration_on": 0, "min_duration_off": 0.00},
        "breath": {"onset": 0.1, "offset": 0.05, "min_duration_on": 0, "min_duration_off": 0.0},
        "hoofbeat": {"onset": 0.3, "offset": 0.2, "min_duration_on":00, "min_duration_off": 0.00},
        #"grazing": {"onset": 0.8, "offset": 0.6, "min_duration_on": 0.005, "min_duration_off": 0.005},
        #"gulp": {"onset": 0.8, "offset": 0.6, "min_duration_on": 0.005, "min_duration_off": 0.005},
    }
}
segmentation = pipeline.instantiate(initial_params)(audio_file)
segmentation