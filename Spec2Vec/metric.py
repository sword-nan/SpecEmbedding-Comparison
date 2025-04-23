import sys
sys.path.append("../")
from typing import Optional
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm
from gensim.models import Word2Vec

from const import gnps
from utils import metric, embedding, cosine_similarity


def search_with_spectra(
    desc: str, model: Word2Vec,
    query_spectra, ref_spectra,
    k_metric: list[int],
    show_progress_bar: bool = True, batch_size: Optional[int] = None
):
    query_embedding, query_smiles = embedding(
        model,  query_spectra, show_progress_bar)
    ref_embedding, ref_smiles = embedding(
        model, ref_spectra, show_progress_bar)
    infos, values = metric(
        k_metric,
        query_embedding,
        ref_embedding,
        query_smiles,
        ref_smiles,
        show_progress_bar,
        batch_size
    )
    df = pd.DataFrame([values], columns=infos, index=[desc])
    return df


def search(
    desc: str, model: Word2Vec,
    query_path: Path, ref_path: Path,
    k_metric: list[int],
    show_progress_bar: bool = True, batch_size: Optional[int] = None
):
    query_spectra = np.load(query_path, allow_pickle=True)
    ref_spectra = np.load(ref_path, allow_pickle=True)
    query_embedding, query_smiles = embedding(
        model,  query_spectra, show_progress_bar)
    ref_embedding, ref_smiles = embedding(
        model, ref_spectra, show_progress_bar)
    infos, values = metric(
        k_metric,
        query_embedding,
        ref_embedding,
        query_smiles,
        ref_smiles,
        show_progress_bar,
        batch_size
    )
    df = pd.DataFrame([values], columns=infos, index=[desc])
    return df


model_files = {
    "orbitrap": {
        "model": "orbitrap.model",
    },
    "qtof": {
        "model": "qtof.model",
    },
    "other": {
        "model": "other.model",
    }
}

spectra_paths = {
    "gnps": {
        "orbitrap": {
            "train": (gnps.ORBITRAP_TRAIN_QUERY, gnps.ORBITRAP_TEST_REF),
            "test": (gnps.ORBITRAP_TEST_QUERY, gnps.ORBITRAP_TEST_REF)
        },
        "qtof": {
            "test": (gnps.QTOF_TEST_QUERY, gnps.QTOF_TEST_REF)
        },
        "other": {
            "test": (gnps.OTHER_TEST_QUERY, gnps.OTHER_TEST_REF)
        }
    },
}

gnps_train_ref = np.load(gnps.ORBITRAP_TRAIN_REF, allow_pickle=True)

batch_size = None
k_metric = [5, 1, 10]
show_progress_bar = False

replica_suffix = "-replication-{}"

replica_df_seq = []

models = {
    desc: Word2Vec.load(metadata["model"])
    for desc, metadata in model_files.items()
}

for i in tqdm(range(10)):
    df_seq = []
    for db, db_metadata in spectra_paths.items():
        for desc, path_metadata in db_metadata.items():
            model = models[desc]
            for info, paths in path_metadata.items():
                print("-" * 40, f"{db}-{desc}-{info}", "-" * 40)
                query_path, ref_path = paths
                query_path = query_path.with_stem(
                    query_path.stem + replica_suffix.format(i + 1))
                ref_path = ref_path.with_stem(
                    ref_path.stem + replica_suffix.format(i + 1))
                if db == "gnps" and desc == "orbitrap":
                    if info == "train":
                        query_path = gnps.ORBITRAP_TRAIN_QUERY

                    ref_spectra = np.load(ref_path, allow_pickle=True)
                    query_spectra = np.load(query_path, allow_pickle=True)
                    ref_spectra = np.hstack((gnps_train_ref, ref_spectra))
                    df = search_with_spectra(
                        f"{db}-{desc}-{info}", model,
                        query_spectra, ref_spectra,
                        k_metric,
                        show_progress_bar, batch_size
                    )
                else:
                    df = search(
                        f"{db}-{desc}-{info}", model,
                        query_path, ref_path,
                        k_metric,
                        show_progress_bar, batch_size
                    )
                df_seq.append(df)
    df = pd.concat(df_seq, axis=0)
    print(df)
    replica_df_seq.append(df)

data = []
indices = replica_df_seq[0].index
columns = replica_df_seq[0].columns
for item in replica_df_seq:
    data.append([item.values])

data = np.concatenate(data, axis=0)

np.set_printoptions(precision=2, suppress=True)
print(np.mean(data, axis=0) * 100)
print(np.std(data, axis=0) * 100)
mean_df = pd.DataFrame(np.mean(data, axis=0) * 100,
                       index=indices, columns=columns)
std_df = pd.DataFrame(np.std(data, axis=0) * 100,
                      index=indices, columns=columns)
mean_df.to_csv("./mean.tsv", sep='\t')
std_df.to_csv("./std.tsv", sep='\t')


model = Word2Vec.load("./qtof.model")
batch_size = None
k_metric = [5, 1, 10]
show_progress_bar = False

query_spectra = np.load(
    "/data1/xp/data/MSBert/MTBLS1572/query.npy", allow_pickle=True)
ref_spectra = np.load(
    "/data1/xp/data/MSBert/MTBLS1572/ref.npy", allow_pickle=True)

df = search_with_spectra(
    "MTBLS1572",
    model,
    query_spectra,
    ref_spectra,
    k_metric,
    show_progress_bar,
    batch_size
)

print(df)

model = Word2Vec.load("./qtof.model")
batch_size = None
k_metric = [5, 1, 10]
show_progress_bar = False

query_spectra = np.load(
    "/data1/xp/data/MSBert/MTBLS1572/query.npy", allow_pickle=True)
ref_spectra = np.load(
    "/data1/xp/data/MSBert/MTBLS1572/ref.npy", allow_pickle=True)

query_embedding, _ = embedding(
    model,
    query_spectra,
    False
)

ref_embedding, _ = embedding(
    model,
    ref_spectra,
    False
)

scores = cosine_similarity(query_embedding, ref_embedding)
print(np.argsort(scores, axis=1)[:, ::-1][:, 0])
