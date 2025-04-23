import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from timm.scheduler.scheduler import Scheduler

from type import StepTrain, StepVal, TrainerConfig


def train(
    model: nn.Module,
    trainer_config: TrainerConfig,
    criterion: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: Scheduler,
    step_train: StepTrain,
    step_val: StepVal,
):
    train_loss = []
    val_loss = []
    step_count = 0

    for epoch in range(trainer_config["n_epoch"]):
        # train
        model.train()
        epoch_loss = []
        if trainer_config["show_progress_bar"]:
            pbar = tqdm(train_loader, total=len(train_loader))
        else:
            pbar = train_loader

        for batch in pbar:
            step_loss = step_train(
                model, trainer_config["device"],
                criterion, batch
            )
            epoch_loss.append(step_loss.item())
            optimizer.zero_grad()
            step_loss.backward()
            optimizer.step()
            scheduler.step(step_count)
            step_count += 1
            if trainer_config["show_progress_bar"]:
                pbar.set_description_str(
                    f"Train Epoch[{epoch + 1}/{trainer_config["n_epoch"]}]"
                )
                pbar.set_postfix(loss=f"{step_loss:.2f}")

        train_loss.append(epoch_loss)
        print(f'train epoch={
            epoch + 1}, loss={np.mean(epoch_loss):.4f}')

        # evaluate
        model.eval()
        with torch.no_grad():
            val_step_loss = []
            if trainer_config["show_progress_bar"]:
                pbar = tqdm(val_loader, total=len(val_loader))
            else:
                pbar = val_loader
            for batch in val_loader:
                step_loss = step_val(
                    model, trainer_config["device"],
                    criterion, batch
                )
                val_step_loss.append(step_loss.item())
            val_loss.append(val_step_loss)
            print(f"validation epoch={
                  epoch + 1}, loss={np.mean(step_loss):.4f}")
    return model, train_loss, val_loss
