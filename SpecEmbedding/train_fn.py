from typing import Callable

import torch
import torch.nn as nn
from torch import device

from type import BatchType


def step_train(
    model: nn.Module,
    criterion: nn.Module,
    device: device,
    batch: BatchType,
    custom_fn: Callable[..., torch.Tensor] = None
):
    x, y = batch
    y = y.to(device)
    pred = []
    for item in x:
        item = [d.to(device) for d in item]
        res: torch.Tensor = model(*item).unsqueeze(dim=1)
        pred.append(res)
    pred = torch.cat(pred, dim=1)
    loss: torch.Tensor = criterion(pred, y)
    if custom_fn is not None:
        custom_metric = custom_fn(pred, y)
        return loss, custom_metric

    return loss


def step_evaluate(
    model: nn.Module,
    criterion: nn.Module,
    device: device,
    batch: BatchType,
    custom_fn: Callable[..., torch.Tensor] = None
):
    x, y = batch
    y = y.to(device)
    pred = []
    for item in x:
        item = [d.to(device) for d in item]
        res: torch.Tensor = model(*item).unsqueeze(dim=1)
        pred.append(res)
    pred = torch.cat(pred, dim=1)
    loss: torch.Tensor = criterion(pred, y)
    if custom_fn is not None:
        custom_metric = custom_fn(pred, y)
        return loss, custom_metric

    return loss
