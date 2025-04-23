from typing import Protocol, TypedDict, Sequence, Callable, Optional

import torch
from torch import nn
from torch import device
import numpy as np
import numpy.typing as npt


class Peak(TypedDict):
    mz: str
    intensity: npt.NDArray


class MetaData(TypedDict):
    peaks: Sequence[Peak]
    smiles: str


class TokenSequence(TypedDict):
    mz: npt.NDArray[np.int32]
    intensity: npt.NDArray[np.float32]


class OptimizerConfig(TypedDict):
    lr: float
    weight_decay: float


class TokenizerConfig(TypedDict):
    max_len: int
    n_decimals: int
    show_progress_bar: bool


class TrainerConfig(TypedDict):
    n_epoch: int
    device: device
    show_progress_bar: bool


class DataLoaderConfig(TypedDict):
    batch_size: int
    val_ratio: float


class LossConfig(TypedDict):
    temperature: float
    ignore_index: int
    reduction: Optional[str]


class Embedder(Protocol):
    def embed(self, mz: torch.Tensor, intensity: torch.Tensor) -> torch.Tensor:
        pass

    def eval(self):
        pass


BatchType = Sequence[torch.Tensor]
EmbedBatchFunc = Callable[[Embedder, BatchType, device], torch.Tensor]
StepTrain = Callable[[nn.Module, device, nn.Module, BatchType], torch.Tensor]
StepVal = Callable[[nn.Module, device, nn.Module, BatchType], torch.Tensor]
