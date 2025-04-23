import sys
sys.path.append("../")

import numpy as np
from ms2deepscore.SettingsMS2Deepscore import SettingsMS2Deepscore
from ms2deepscore.wrapper_functions.training_wrapper_functions import train_ms2ds_model

from const import gnps

train_spectra = np.load(gnps.ORBITRAP_TRAIN_REF, allow_pickle=True)
val_spectra = np.load(gnps.ORBITRAP_TRAIN_QUERY, allow_pickle=True)

setting = SettingsMS2Deepscore(
    **{
        "epochs": 500,
        "base_dims": (1000, 1000, 1000),
        "embedding_dim": 512,
        "ionisation_mode": "positive",
        "batch_size": 2048,
        "learning_rate": 0.00025,
        "patience": 30,
    }
)

model, history = train_ms2ds_model(
    train_spectra,
    val_spectra,
    './res',
    setting,
)
