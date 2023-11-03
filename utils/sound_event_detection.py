import math
import torch
import torch.nn as nn
import numpy as np
from pyannote.core import Annotation, Segment, SlidingWindow
from pyannote.audio import Model
from pyannote.audio.core.task import Task, Resolution
from pyannote.database import Protocol
from pyannote.audio.core.task import Specifications
from pyannote.audio.core.task import Problem
from pyannote.audio.utils.loss import binary_cross_entropy, nll_loss
from typing import Dict, List, Literal, Optional, Sequence, Text, Tuple, Union
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torchmetrics import Metric, Precision, Recall, F1Score, MetricCollection
from torch.utils.data import DataLoader, Dataset, IterableDataset
from functools import cached_property, partial

class TrainDataset(IterableDataset):
    def __init__(self, task: Task):
        super().__init__()
        self.task = task

    def __iter__(self):
        return self.task.train__iter__()

    def __len__(self):
        return self.task.train__len__()


class ValDataset(Dataset):
    def __init__(self, task: Task):
        super().__init__()
        self.task = task

    def __getitem__(self, idx):
        return self.task.val__getitem__(idx)

    def __len__(self):
        return self.task.val__len__()

class SoundEventDetection(Task):
    """Sound event detection"""

    def __init__(
        self,
        protocol: Protocol,
        classes: Optional[List[str]] = None,
        balance: Sequence[Text] = None,
        duration: float = 1.5,
        min_duration: Optional[float] = None,
        warm_up: Union[float, Tuple[float, float]] = 0.0,#Mostly use for segmentation tasks.
        batch_size: int = 32,
        num_workers: int = None,#Number of workers used for generating training samples.
        pin_memory: bool = False,
        augmentation: BaseWaveformTransform = None,
    ):

        super().__init__(
            protocol,
            duration=duration,
            min_duration=min_duration,
            warm_up=warm_up,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            augmentation=augmentation,
        )
        self.balance = balance
        self.classes = classes 

    def setup(self):
        
        self.training_metadata_ = list()
        # 加载训练子集的元数据
        for training_file in self.protocol.train():
            training_file_duration = training_file["annotated"].duration()
            self.training_metadata_.append({
                "audio": training_file["audio"],
                "duration": training_file_duration,
                "annotation": training_file["annotation"],
            })


        # gather the list of classes
        if self.classes is None:
            inferred_classes = set()
            for training_file in self.training_metadata_:
                inferred_classes.update(training_file["annotation"].labels())
            self.classes = sorted(inferred_classes)
        else:
            # Validate or filter the provided classes against the dataset if necessary
            pass

        # specify the addressed problem
        self.specifications = Specifications(
            # it is a multi-label classification problem
            problem=Problem.MULTI_LABEL_CLASSIFICATION,
            # FRAME = 1 model outputs a sequence of frames;  CHUNK = 2 model outputs 
            # just one vector for the whole chunk
            resolution=Resolution.FRAME,
            # the model will ingest chunks with that duration (in seconds)
            duration=self.duration,
            # human-readable names of classes
            classes=self.classes)

        # `has_validation` is True iff protocol defines a development set
        if not self.has_validation:
            return

        # load metadata for validation subset
        self.validation_metadata_ = list()
        for validation_file in self.protocol.development():
            validation_file_duration = validation_file["annotated"].duration()
            num_samples = math.floor(validation_file_duration / self.duration)
            self.validation_metadata_.append({
                "audio": validation_file["audio"],
                "num_samples": num_samples,
                "annotation": validation_file["annotation"],
            })

    def default_loss(
            self, specifications: Specifications, target, prediction, weight=None
            ) -> torch.Tensor:
        """Guess and compute default loss according to task specification

        Parameters
        ----------
        specifications : Specifications
            Task specifications
        target : torch.Tensor
            * (batch_size, num_frames) for binary classification
            * (batch_size, num_frames) for multi-class classification
            * (batch_size, num_frames, num_classes) for multi-label classification
        prediction : torch.Tensor
            (batch_size, num_frames, num_classes)
        weight : torch.Tensor, optional
            (batch_size, num_frames, 1)

        Returns
        -------
        loss : torch.Tensor
            Binary cross-entropy loss in case of binary and multi-label classification,
            Negative log-likelihood loss in case of multi-class classification.

        """

        if specifications.problem in [
            Problem.BINARY_CLASSIFICATION,
            Problem.MULTI_LABEL_CLASSIFICATION,
        ]:
            return binary_cross_entropy(prediction, target, weight=weight)

        elif specifications.problem in [Problem.MONO_LABEL_CLASSIFICATION]:
            return nll_loss(prediction, target, weight=weight)

        else:
            msg = "TODO: implement for other types of problems"
            raise NotImplementedError(msg)

            
    def common_step(self, batch, batch_idx: int, stage: Literal["train", "val"]):
        """Default training or validation step according to task specification

            * binary cross-entropy loss for binary or multi-label classification
            * negative log-likelihood loss for regular classification

        If "weight" attribute exists, batch[self.weight] is also passed to the loss function
        during training (but has no effect in validation).

        Parameters
        ----------
        batch : (usually) dict of torch.Tensor
            Current batch.
        batch_idx: int
            Batch index.
        stage : {"train", "val"}
            "train" for training step, "val" for validation step

        Returns
        -------
        loss : {str: torch.tensor}
            {"loss": loss}
        """

        if isinstance(self.specifications, tuple):
            raise NotImplementedError(
                "Default training/validation step is not implemented for multi-task."
            )

        # forward pass
        y_pred = self.model(batch["X"])

        batch_size, num_frames, _ = y_pred.shape
        # (batch_size, num_frames, num_classes)

        # target
        y = batch["y"]

        # frames weight
        weight_key = getattr(self, "weight", None) if stage == "train" else None
        weight = batch.get(
            weight_key,
            torch.ones(batch_size, num_frames, 1, device=self.model.device),
        )
        # (batch_size, num_frames, 1)

        # warm-up
        warm_up_left = round(self.warm_up[0] / self.duration * num_frames)
        weight[:, :warm_up_left] = 0.0
        warm_up_right = round(self.warm_up[1] / self.duration * num_frames)
        weight[:, num_frames - warm_up_right :] = 0.0

        # compute loss
        loss = self.default_loss(self.specifications, y, y_pred, weight=weight)

        # skip batch if something went wrong for some reason
        if torch.isnan(loss):
            return None

        self.model.log(
            f"loss/{stage}",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        return {"loss": loss}

    def train__iter__(self):
        # create worker-specific random number generator (RNG):
        rng = create_rng_for_worker(self.model.current_epoch)

        # load list and number of classes
        classes = self.specifications.classes
        num_classes = len(classes)

        # yield training samples "ad infinitum"
        while True:

            # select training file at random
            random_training_file, *_ = rng.choices(self.training_metadata_, k=1)

            # select one chunk at random 
            random_training_file_duration = random_training_file["annotated"].duration() 
            random_start_time = rng.uniform(0, random_training_file_duration - self.duration)
            random_chunk = Segment(random_start_time, random_start_time + self.duration)

            # load audio excerpt corresponding to random chunk
            X = self.model.audio.crop(random_training_file["audio"], 
                                      random_chunk, 
                                      self.duration)
            
            # load labels corresponding to random chunk as {0|1} numpy array
            # y[k] = 1 means that kth class is active
            y = np.zeros((num_classes,))
            active_classes = random_training_file["annotation"].crop(random_chunk).labels()
            for active_class in active_classes:
                y[classes.index(active_class)] = 1
        
            # yield training samples as a dict (use 'X' for input and 'y' for target)
            yield {'X': X, 'y': y}
              
        
    def train__len__(self):
        # outputs the number of training samples that make an epoch by compute this number as the
        # total duration of the training set divided by duration of training chunks. 
        train_duration = sum(training_file["annotated"].duration() for training_file in self.training_metadata_)
        return max(self.batch_size, math.ceil(train_duration / self.duration))
    """
    def collate_fn(self, batch, stage="train"):
        msg = f"Missing '{self.__class__.__name__}.collate_fn' method."
        raise NotImplementedError(msg)
    """        

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            TrainDataset(self),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            #collate_fn=partial(self.collate_fn, stage="train"),
        )

    def training_step(self, batch, batch_idx: int):
        return self.common_step(batch, batch_idx, "train")   

    def val__getitem__(self, sample_idx):

        # load list and number of classes
        classes = self.specifications.classes
        num_classes = len(classes)


        # find which part of the validation set corresponds to sample_idx
        num_samples = np.cumsum([
            validation_file["num_samples"] for validation_file in self.validation_metadata_])
        file_idx = np.where(num_samples >= sample_idx)[0][0]
        validation_file = self.validation_metadata_[file_idx]
        idx = sample_idx - (num_samples[file_idx] - validation_file["num_samples"]) 
        chunk = SlidingWindow(start=0., duration=self.duration, step=self.duration)[idx]

        # load audio excerpt corresponding to current chunk
        X = self.model.audio.crop(validation_file["audio"], chunk, self.duration)

        # load labels corresponding to random chunk as {0|1} numpy array
        # y[k] = 1 means that kth class is active
        y = np.zeros((num_classes,))
        active_classes = validation_file["annotation"].crop(chunk).labels()
        for active_class in active_classes:
            y[classes.index(active_class)] = 1

        return {'X': X, 'y': y}

    def val__len__(self):
        return sum(validation_file["num_samples"] 
                   for validation_file in self.validation_metadata_)
    
    def val_dataloader(self) -> Optional[DataLoader]:
        if self.has_validation:
            return DataLoader(
                ValDataset(self),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=False,
                #collate_fn=partial(self.collate_fn, stage="val"),
            )
        else:
            return None

    # default validation_step provided for convenience
    # can obviously be overriden for each task
    def validation_step(self, batch, batch_idx: int):
        return self.common_step(batch, batch_idx, "val")
    
    def default_metric(self) -> Union[Metric, Sequence[Metric], Dict[str, Metric]]:
        """Default validation metric"""
        task_type = 'multiclass' 
        precision = Precision(num_classes=len(self.classes), average='macro', task=task_type)
        recall = Recall(num_classes=len(self.classes), average='macro', task=task_type)
        f1 = F1Score(num_classes=len(self.classes), average='macro', task=task_type)

        return {'precision': precision,'recall': recall,'f1': f1,}

    @cached_property
    def metric(self) -> MetricCollection:
        if self._metric is None:
            self._metric = self.default_metric()

        return MetricCollection(self._metric)

    def setup_validation_metric(self):
        metric = self.metric
        if metric is not None:
            self.model.validation_metric = metric
            self.model.validation_metric.to(self.model.device)

    @property
    def val_monitor(self):
        """Quantity (and direction) to monitor

        Useful for model checkpointing or early stopping.

        Returns
        -------
        monitor : str
            Name of quantity to monitor.
        mode : {'min', 'max}
            Minimize

        See also
        --------
        pytorch_lightning.callbacks.ModelCheckpoint
        pytorch_lightning.callbacks.EarlyStopping
        """

        name, metric = next(iter(self.metric.items()))
        return name, "max" if metric.higher_is_better else "min"