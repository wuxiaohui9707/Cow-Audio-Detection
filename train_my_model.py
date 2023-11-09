#%%
from pyannote.audio.tasks import __all__ as TASKS; print('\n'.join(TASKS))

from pyannote.database import registry
import os
registry.load_database(
    os.path.join('/mnt/', 'e', 'Files', 'Acoustic_Data', 'Datasets_specified', 'yaml','My_Databases_specified.yml')
    )
os.environ["PYANNOTE_DATABASE_CONFIG"] = os.path.join('/mnt/', 'e', 'Files', 'Acoustic_Data', 'Datasets_specified','yaml', 'My_Databases_specified.yml')

from pyannote.database import get_protocol, FileFinder
preprocessors = {"audio": FileFinder()}
cow_audio = get_protocol('My_datasets_specified.SpeakerDiarization.Detection', 
                   preprocessors=preprocessors)

#%%
from pyannote.audio.tasks import MultiLabelSegmentation
MLS_task = MultiLabelSegmentation(
    cow_audio, duration=2.0, batch_size=32,
    classes= ["rumination","hoofbeat","breath","gulp","grazing"]
    )

#%%
from huggingface_hub import notebook_login
notebook_login()
#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
#%%
from pyannote.audio import Model
token = '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'
model = Model.from_pretrained("pyannote/segmentation", use_auth_token=token)
model.task = MLS_task
model.to("cuda")

#%%
import pytorch_lightning as pl
output_directory = os.path.join('/mnt/', 'e', 'Files', 'Acoustic_Data', 'Datasets_specified')
trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=10, default_root_dir=output_directory)
trainer.fit(model)
# %%
from pyannote.audio.pipelines import MultiLabelSegmentation
test_file = next(cow_audio.test())
pipeline = MultiLabelSegmentation(segmentation=model)
initial_params = {
    "thresholds": {
        "rumination": {"onset": 0.8, "offset": 0.6, "min_duration_on": 0.01, "min_duration_off": 0.00},
        "breath": {"onset": 0.8, "offset": 0.6, "min_duration_on": 0.01, "min_duration_off": 0.01},
        "hoofbeat": {"onset": 0.8, "offset": 0.6, "min_duration_on": 0.005, "min_duration_off": 0.005},
        "grazing": {"onset": 0.8, "offset": 0.6, "min_duration_on": 0.005, "min_duration_off": 0.005},
        "gulp": {"onset": 0.8, "offset": 0.6, "min_duration_on": 0.005, "min_duration_off": 0.005},
    }
}
pipeline.instantiate(initial_params)
detection = pipeline(test_file)
# %%
from pyannote.core import Annotation
assert isinstance(detection, Annotation)

for speech_turn, track, speaker in detection.itertracks(yield_label=True):
    print(f"{speech_turn.start:4.1f} {speech_turn.end:4.1f} {speaker}")
# %%
from pyannote.audio import Inference
inference = Inference(model, step=2)
output = inference(test_file)
# %%
output
# %%
