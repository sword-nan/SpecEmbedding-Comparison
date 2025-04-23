import torch
import torch.nn as nn
from torch import device

from type import BatchType


def step_train(
    model: nn.Module,
    device: device,
    criterion: nn.Module,
    batch: BatchType,
):
    mz, intensity = batch
    mz, intensity = mz.to(device), intensity.to(device)
    logits_lm1, mask_token1, pool1, logits_lm2, mask_token2, pool2 = model(
        mz, intensity)

    loss: torch.Tensor = criterion(
        logits_lm1, mask_token1, logits_lm2,
        mask_token2, pool1, pool2
    )
    return loss


def step_evaluate(
    model: nn.Module,
    device: device,
    criterion: nn.Module,
    batch: BatchType,
):
    mz, intensity = batch
    mz, intensity = mz.to(device), intensity.to(device)
    logits_lm1, mask_token1, pool1, logits_lm2, mask_token2, pool2 = model(
        mz, intensity)

    loss: torch.Tensor = criterion(
        logits_lm1, mask_token1, logits_lm2,
        mask_token2, pool1, pool2
    )
    return loss
