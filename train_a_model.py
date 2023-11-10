#%%
# preparing notebook for visualization purposes
# (only show outputs between t=180s and t=240s)
from pyannote.core import notebook, Segment
notebook.crop = Segment(210, 240)

from pyannote.database import registry
import os
registry.load_database(
    os.path.join('/mnt/', 'e', 'Files', 'Github', 'AMI-diarization-setup', 'pyannote', 'database.yml')
    )
os.environ["PYANNOTE_DATABASE_CONFIG"] = os.path.join('/mnt/', 'e', 'Files', 'Github', 'AMI-diarization-setup', 'pyannote', 'database.yml')

#%%
from pyannote.database import get_protocol
ami = get_protocol('AMI.SpeakerDiarization.only_words')

from pyannote.audio.tasks import VoiceActivityDetection
vad_task = VoiceActivityDetection(ami, duration=2.0, batch_size=128)

from pyannote.audio.models.segmentation import PyanNet
vad_model = PyanNet(task=vad_task, sincnet={'stride': 10})
vad_model.to('cuda')
#%%
import pytorch_lightning as pl
trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=1)
trainer.fit(vad_model)

#%%
test_file = next(ami.test())

from pyannote.audio import Inference
vad = Inference(vad_model)

vad_probability = vad(test_file)
vad_probability

expected_output = test_file["annotation"].get_timeline().support()
expected_output
#%%
from huggingface_hub import notebook_login
notebook_login()

#%%
from pyannote.audio import Model
pretrained = Model.from_pretrained("pyannote/segmentation", use_auth_token=True)

spk_probability = Inference(pretrained, step=2.5)(test_file)
spk_probability

#%%
test_file["annotation"].discretize(notebook.crop, resolution=0.010)


#%%
# fine-tune pretrained model 
from pyannote.audio.tasks import SpeakerDiarization
seg_task = SpeakerDiarization(ami, duration=5.0, max_num_speakers=4)

#%%
def test(model, protocol, subset="test"):
    from pyannote.audio.utils.signal import binarize
    from pyannote.audio.utils.metric import DiscreteDiarizationErrorRate
    from pyannote.audio.pipelines.utils import get_devices

    (device,) = get_devices(needs=1)
    metric = DiscreteDiarizationErrorRate()
    files = list(getattr(protocol, subset)())

    inference = Inference(model, device=device)

    for file in files:
        reference = file["annotation"]
        hypothesis = binarize(inference(file))# 使用模型对文件进行推断，并将输出二值化
        uem = file["annotated"]
        _ = metric(reference, hypothesis, uem=uem)
        
    return abs(metric)

# %%
der_pretrained = test(model=pretrained, protocol=ami, subset="test")
print(f"Local DER (pretrained) = {der_pretrained * 100:.1f}%")

# %%
from copy import deepcopy
finetuned = deepcopy(pretrained)
finetuned.task = seg_task

# %%
trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=1)
trainer.fit(finetuned)
# %%
der_finetuned = test(model=finetuned, protocol=ami, subset="test")
print(f"Local DER (finetuned) = {der_finetuned * 100:.1f}%")

# %%
Inference('pyannote/segmentation', use_auth_token=True, step=2.5)(test_file)

# %%
test_file["annotation"]

# %%
from pyannote.audio.tasks import OverlappedSpeechDetection
osd_task = OverlappedSpeechDetection(ami, duration=2.0)

osd_model = Model.from_pretrained("pyannote/segmentation", use_auth_token=True)
osd_model.to('cuda')
osd_model.task = osd_task
# %%
osd_model.freeze_up_to('lstm')

# %%
trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=1)
trainer.fit(osd_model)
# %%
from pyannote.audio.utils.signal import binarize
binarize(Inference(osd_model)(test_file))
# %%
test_file["annotation"]