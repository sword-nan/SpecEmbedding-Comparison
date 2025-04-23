import sys
sys.path.append("../")
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from train import Trainer, set_seed
from data import TrainDataset, Tokenizer, get_classified_tokenset
from type import (
    AugmentationConfig,
    StorageConfig,
    StepFuncConfig,
    DescriptionConfig,
    TokenizerConfig,
    DataLoaderConfig,
    SchedulerConfig,
    TrainerConfig,
    OptimizerConfig,
    SupConLossConfig,
    TanimotoLossConfig,
    SupConLossWithTanimotoScoreConfig
)
from train_fn import step_train, step_evaluate
from loss import SupConLossWithTanimotoScore, SupConLoss, TanimotoScoreLoss
from model import SiameseModel
from const import gnps


# load data
train_spectra = np.load(gnps.ORBITRAP_TRAIN_REF, allow_pickle=True)
unique_smiles = np.load(
    gnps.DIR / gnps.UNIQUE_SMILES,
    allow_pickle=True
)
print("load data success")

# tokenization
tokenizer_config = TokenizerConfig(
    max_len=100,
    show_progress_bar=True
)
tokenizer = Tokenizer(**tokenizer_config)
sequences = tokenizer.tokenize_sequence(train_spectra)
data, labels = get_classified_tokenset(unique_smiles, sequences)
print("finish the tokenize process")

### set the seed
seed = set_seed(42)
is_augment = True

augment_config = AugmentationConfig(
    prob=0.5,
    removal_max=0.2,
    removal_intensity=0.3,
    rate_intensity=0.15,
)

# dataset and dataloader
dataloader_config = DataLoaderConfig(
    batch_size=512,
    val_ratio=0.2
)

length = len(labels)
val_indices = np.random.choice(
    np.arange(length), 
    int(dataloader_config["val_ratio"] * length), 
    replace=False
)
val_keys = labels[val_indices]
train_keys = np.delete(labels, val_indices)

train_dataset = TrainDataset(
    data=data,
    keys=train_keys,
    n_views=2,
    is_augment=is_augment,
    augment_config=augment_config
)

val_dataset = TrainDataset(
    data=data,
    keys=val_keys,
    n_views=2
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=dataloader_config["batch_size"],
    shuffle=True
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=dataloader_config["batch_size"],
    shuffle=False
)

loss_type = "SupConWithTanimotoLoss"
model_backbone = "transformer"

# model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SiameseModel(
    512,
    16,
    4,
    512,
    512,
    feedward_activation="selu",
)
model = model.to(device)

# trainer
trainer_config = TrainerConfig(
    n_epoch=50,
    device=device,
    early_stop=20,
    show_progress_bar=True
)
# scheduler
n_step = trainer_config["n_epoch"] * len(train_dataloader)
scheduler_config = SchedulerConfig(
    warmup_steps=int(n_step * 0.1),
    total_steps=n_step
)

# loss
reduction = "mean"
tanimoto_score_path = "/data1/xp/data/GNPS/tanimoto_score.npy"
if loss_type == "SupConLoss":
    loss_config = SupConLossConfig(
        device=device,
        temperature=0.01,
        base_temperature=0.5,
        contrast_mode='all',
        reduction=reduction
    )
    criterion = SupConLoss(**loss_config)
elif loss_type == "TanimotoLoss":
    loss_config = TanimotoLossConfig(
        score_path=tanimoto_score_path,
        device=device,
        reduction=reduction
    )
    criterion = TanimotoScoreLoss(**loss_config)
elif loss_type == "SupConWithTanimotoLoss":
    loss_config = SupConLossWithTanimotoScoreConfig(
        alpha=30,
        score_path=tanimoto_score_path,
        device=device,
        temperature=0.05,
        contrast_mode="all",
        base_temperature=0.05,
        reduction=reduction
    )
    criterion = SupConLossWithTanimotoScore(**loss_config)

# optimizer
optimizer_config = OptimizerConfig(
    lr=7.5e-5,
    weight_decay=0.1
)
optimizer = AdamW(model.parameters(), **optimizer_config)

# desc
desc_config = DescriptionConfig(
    train="train, epoch={}, loss={:.4f}",
    val="validation, epoch={}, loss={:.4f}, save the model",
    end="model train end, best model loss={:.4f}"
)
# step func
stepfunc_config = StepFuncConfig(
    train=step_train,
    val=step_evaluate
)

suffix = ""
if is_augment:
    suffix = "-Augmentation"

# storage
model_dir = Path(f"./model/{model_backbone}-{loss_type}{suffix}")

model_dir.mkdir(parents=True, exist_ok=True)
storage_config = StorageConfig(
    model=model_dir / f"model.ckpt",
    lr=model_dir / f"lr.npy",
    step_loss=model_dir / f"step_loss.npy",
    loss=model_dir / "epoch_loss.npy",
    custom=model_dir / "custom_metric.npy"
)

# model train
torch.cuda.empty_cache()
trainer = Trainer(
    model,
    reduction,
    train_dataloader,
    val_dataloader,
    optimizer,
    criterion,
    device,
    stepfunc_config,
    desc_config,
    storage_config,
    trainer_config,
    scheduler_config
)

trainer.train()