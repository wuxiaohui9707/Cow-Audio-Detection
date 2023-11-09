from pyannote.core import notebook, Segment
notebook.crop = Segment(210,240)

from pyannote.database import registry
import os
registry.load_database(
    os.path.join('/mnt/', 'e', 'Files', 'Github', 'AMI-diarization-setup', 'pyannote', 'database.yml')
    )
os.environ["PYANNOTE_DATABASE_CONFIG"] = os.path.join('/mnt/', 'e', 'Files', 'Github', 'AMI-diarization-setup', 'pyannote', 'database.yml')

#get protocol
from pyannote.database import get_protocol
ami = get_protocol('AMI.SpeakerDiarization.only_words')

#define task
from pyannote.audio.tasks import VoiceActivityDetection
vad_task = VoiceActivityDetection(ami,duration=2.0,batch_size=128)

#load model
from pyannote.audio.models.segmentation import PyanNet
vad_model = PyanNet(task=vad_task, sincnet={'stride':10})
vad_model.to("cuda")

#train model
import pytorch_lightning as pl
trainer = pl.Trainer(devices=1, accelerator='gpu', max_epochs=1)
trainer.fit(vad_model)

#test
test_file = next(ami.test())
from pyannote.audio import Inference
vad = Inference(vad_model)

vad_probability = vad(test_file)
vad_probability

expected_output = test_file["annotation"].get_timeline().support()
expected_output