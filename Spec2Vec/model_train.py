import sys
sys.path.append("../")

import numpy as np
from spec2vec import SpectrumDocument
from spec2vec.model_building import train_new_word2vec_model

from const import gnps

model_files = {
    "orbitrap": {
        "model": "orbitrap.model",
        "data": gnps.ORBITRAP_TRAIN_REF,
    },
    "qtof": {
        "model": "qtof.model",
        "data": gnps.QTOF_TEST_REF,
    },
    "other": {
        "model": "other.model",
        "data": gnps.OTHER_TEST_REF
    }
}

print("train_model")

for desc, metadata in model_files.items():
    model = metadata["model"]
    data_path = metadata["data"]
    spectra = np.load(data_path, allow_pickle=True)
    spectrum_documents = [
        SpectrumDocument(s, n_decimals=2)
        for s in spectra
    ]
    train_new_word2vec_model(
        spectrum_documents,
        10,
        filename=model,
        vector_size=512,
        workers=10,
        progress_logger=True
    )
