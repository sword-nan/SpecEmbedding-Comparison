from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import numpy as np
from tqdm import tqdm
import numpy.typing as npt
from numba import njit, prange
from matchms import Spectrum
from spec2vec import calc_vector, SpectrumDocument
from gensim.models import Word2Vec


@njit
def eq(a: npt.NDArray, b: npt.NDArray):
    return np.equal(a, b)


@njit
def cosine_similarity(A: npt.NDArray, B: npt.NDArray):
    norm_A = np.sqrt(np.sum(A ** 2, axis=1)) + 1e-8
    norm_B = np.sqrt(np.sum(B ** 2, axis=1)) + 1e-8
    normalize_A = A / norm_A[:, np.newaxis]
    normalize_B = B / norm_B[:, np.newaxis]
    scores = np.dot(normalize_A, normalize_B.T)
    return scores


@njit(parallel=True)
def top_k_indices(score, top_k):
    rows, cols = score.shape
    indices = np.empty((rows, top_k), dtype=np.int64)
    for i in prange(rows):
        row = score[i]
        sorted_idx = np.argsort(row)[::-1]
        indices[i] = sorted_idx[:top_k]
    return indices


def metric(
    k_metric: list[int],
    query_embedding: npt.NDArray,
    ref_embedding: npt.NDArray,
    query_smiles: npt.NDArray,
    ref_smiles: npt.NDArray,
    show_progress_bar: bool = True,
    batch_size: Optional[int] = None
):
    def init_count():
        count = {
            f"top{k}": 0
            for k in k_metric
        }
        return count

    def calculate_hit_count(start: int, end: int, indices: npt.NDArray):
        for query_index, ref_index in zip(range(start, end), indices):
            q = query_smiles[query_index]
            r = ref_smiles[ref_index]
            for k in k_metric:
                if q in r[:k]:
                    hit_count[f"top{k}"] += 1

    # def calculate_recall_count(start: int, end: int, indices: npt.NDArray):
    #     for query_index, ref_index in zip(range(start, end), indices):
    #         q = query_smiles[query_index]
    #         r = ref_smiles[ref_index]
    #         for k in k_metric:
    #             recall_count[f"top{k}"] += np.sum(r[:k] == q)

    k_metric = sorted(list(set(k_metric)))
    k_max = max(k_metric)
    if batch_size is None:
        batch_size = len(query_embedding)

    # candidate_recall_num = 0
    # pbar = query_smiles
    # if show_progress_bar:
    #     pbar = tqdm(query_smiles, total=len(query_smiles), desc="calculate all the candidates compounds")
    # for q in pbar:
    #     candidate_recall_num += np.sum(ref_smiles == q)
    # print(f"the recall count is {candidate_recall_num}")

    hit_count = init_count()
    # recall_count = init_count()
    start_seq = range(0, len(query_embedding), batch_size)
    end_seq = range(batch_size, len(query_embedding) + batch_size, batch_size)
    pbar = zip(start_seq, end_seq)
    if show_progress_bar:
        pbar = tqdm(pbar, total=len(start_seq),
                    desc="calculate hit and recall count")

    for start, end in pbar:
        score = cosine_similarity(query_embedding[start:end], ref_embedding)
        indices = top_k_indices(score, k_max)
        calculate_hit_count(start, end, indices)
        # calculate_recall_count(start, end, indices)

    for key in hit_count:
        hit_count[key] /= len(query_embedding)
        # recall_count[key] /= candidate_recall_num

    infos = []
    values = []
    for info, value in hit_count.items():
        infos.append(info)
        values.append(value)
    return infos, values


def embedding(
    model: Word2Vec,
    spectra: Sequence[Spectrum],
    show_progress_bar: bool = True
):
    smiles = []
    embeddings = []
    pbar = spectra
    if show_progress_bar:
        pbar = tqdm(spectra, total=len(spectra), desc="get smiles, embedding")

    for s in pbar:
        document = SpectrumDocument(s, 2)
        vec = calc_vector(
            model,
            document,
            0.5,
            50,
        )
        embeddings.append(vec)
        smiles.append(s.get("smiles"))
    return np.array(embeddings), np.array(smiles)


def search_with_spectra(
    desc: str, model: Word2Vec,
    query_spectra: Sequence[Spectrum], ref_spectra: Sequence[Spectrum],
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


def get_labels(
    unique_smiles,
    smiles_seq
):
    labels = []
    for smiles in smiles_seq:
        labels.append(np.where(unique_smiles == smiles)[0][0])
    return np.array(labels)


def load_model(model_path):
    model = Word2Vec.load(model_path)
    return model


def most_similar(
    query_embedding: npt.NDArray,
    ref_embedding: npt.NDArray,
    batch_size: int,
    show_progress_bar: bool = True
):
    start_seq = range(0, len(query_embedding), batch_size)
    end_seq = range(batch_size, len(query_embedding) + batch_size, batch_size)
    pbar = zip(start_seq, end_seq)
    if show_progress_bar:
        pbar = tqdm(pbar, total=len(start_seq),
                    desc="processing")

    scores = []
    most_similar_indices = []

    for start, end in pbar:
        score = cosine_similarity(query_embedding[start:end], ref_embedding)
        indices = top_k_indices(score, 1).flatten()
        for i, j in zip(range(score.shape[0]), indices):
            scores.append(score[i][j])
            most_similar_indices.append(j)
    return np.array(scores), np.array(most_similar_indices)
