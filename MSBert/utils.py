from typing import Optional, Sequence

import pandas as pd
import torch
from torch import device
from torch.utils.data import DataLoader
from numba import prange, njit
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from matchms import Spectrum

from type import Embedder
from model import MSBERT
from data import Tokenizer, TokenSequenceDataset


def embedding(
    model: Embedder,
    device: device,
    tokenizer: Tokenizer,
    batch_size: int,
    spectra: Sequence[Spectrum],
    show_progress_bar: bool = True
):
    sequences = tokenizer.tokenize_sequence(spectra)
    smiles_seq = np.array([
        s.get("smiles")
        for s in spectra
    ])
    dataset = TokenSequenceDataset(sequences)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )
    spectra_embedding = ModelEmbed(
        model, dataloader, device, show_progress_bar)
    return spectra_embedding, smiles_seq


def ModelEmbed(
    model: Embedder,
    dataloader: DataLoader,
    device: device,
    show_progress_bar: bool = False
):
    model.eval()
    embed_list = []
    pbar = dataloader
    if show_progress_bar:
        pbar = tqdm(dataloader, total=len(dataloader))

    with torch.no_grad():
        for batch in pbar:
            mz, intensity = batch
            mz, intensity = mz.to(device), intensity.to(device)
            emebedded_vectors = model.embed(mz, intensity)
            embed_list.append(emebedded_vectors.cpu().numpy())
    embed_arr = np.concatenate(embed_list, axis=0)
    return embed_arr


def SearchTop(
    ref_arr: npt.NDArray, query_arr: npt.NDArray,
    ref_smiles, query_smiles,
    batch_size: int
):
    '''
    Top-n metrics for computing library matching
    '''
    top1 = []
    top5 = []
    top10 = []
    start = 0
    ref_square_sum: npt.NDArray = np.linalg.norm(ref_arr, axis=1)
    # n * 1
    ref_square_sum = ref_square_sum.reshape(ref_square_sum.shape[0], 1)
    while start < query_arr.shape[0]:
        end = start+batch_size
        batch_query = query_arr[start:end, :]
        batch_query_square_sum: npt.NDArray = np.linalg.norm(
            batch_query, axis=1)
        # 1 * batch
        batch_query_square_sum = batch_query_square_sum.reshape(
            1, batch_query_square_sum.shape[0])
        # n * batch
        n_q = np.repeat(
            batch_query_square_sum, ref_square_sum.shape[0], axis=0)
        # n * batch
        dot = np.dot(ref_arr, batch_query.T)
        n_d = np.repeat(ref_square_sum, batch_query.shape[0], axis=1)
        sim = dot/(n_d*n_q)
        # n * batch
        sort = np.argsort(sim, axis=0)
        sort = np.flipud(sort)
        for s in range(sort.shape[1]):
            smi_q = query_smiles[(s+start)]
            smi_dataset = [ref_smiles[i] for i in sort[0:10, s]]
            if smi_q in smi_dataset:
                top10.append(1)
            smi_dataset = [ref_smiles[i] for i in sort[0:5, s]]
            if smi_q in smi_dataset:
                top5.append(1)
            smi_dataset = [ref_smiles[i] for i in sort[0:1, s]]
            if smi_q in smi_dataset:
                top1.append(1)
        start += batch_size
    top1 = len(top1)/len(query_smiles)
    top5 = len(top5)/len(query_smiles)
    top10 = len(top10)/len(query_smiles)
    return [top1, top5, top10]


@njit
def eq(a: npt.NDArray, b: npt.NDArray):
    return np.equal(a, b)


@njit(parallel=True)
def top_k_indices(score, top_k):
    rows, cols = score.shape
    indices = np.empty((rows, top_k), dtype=np.int64)
    for i in prange(rows):
        row = score[i]
        sorted_idx = np.argsort(row)[::-1]
        indices[i] = sorted_idx[:top_k]
    return indices


def get_labels(
    unique_smiles,
    smiles_seq
):
    labels = []
    for smiles in smiles_seq:
        labels.append(np.where(unique_smiles == smiles)[0][0])
    return np.array(labels)


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


def load_model(device):
    model_state_path = "/data1/xp/data/MSBert/MSBERT.pkl"
    model_state = torch.load(model_state_path, map_location=device)
    model = MSBERT(
        100002,
        512,
        6,
        16,
        0,
        100,
        3
    )
    model.load_state_dict(model_state)
    model = model.to(device)
    return model


def search_with_spectra(
    desc: str, tokenizer: Tokenizer,
    model, device,
    query_spectra: npt.NDArray, ref_spectra: npt.NDArray,
    k_metric: list[int],
    loader_batch_size: int,
    batch_size: Optional[int] = None,
    show_progress_bar: bool = True
):
    query_smiles = np.array([
        s.get("smiles")
        for s in query_spectra
    ])
    ref_smiles = np.array([
        s.get("smiles")
        for s in ref_spectra
    ])
    query_sequences = tokenizer.tokenize_sequence(query_spectra)
    ref_sequences = tokenizer.tokenize_sequence(ref_spectra)
    print("tokenize the query and reference data success")
    query_dataset = TokenSequenceDataset(query_sequences)
    ref_dataset = TokenSequenceDataset(ref_sequences)

    ref_loader = DataLoader(
        ref_dataset,
        batch_size=loader_batch_size,
        shuffle=False
    )
    query_loader = DataLoader(
        query_dataset,
        batch_size=loader_batch_size,
        shuffle=False
    )
    ref_embedding = ModelEmbed(model, ref_loader, device)
    query_embedding = ModelEmbed(model, query_loader, device)

    infos, values = metric(
        k_metric,
        query_embedding, ref_embedding,
        query_smiles, ref_smiles,
        show_progress_bar,
        batch_size
    )
    hit_df = pd.DataFrame([values], columns=infos, index=[desc])
    return hit_df


def search(
    desc: str, tokenizer: Tokenizer,
    model, device,
    query_path: str, ref_path: str,
    k_metric: list[int],
    loader_batch_size: int,
    batch_size: Optional[int] = None,
    show_progress_bar: bool = True
):
    query_spectra = np.load(query_path, allow_pickle=True)
    ref_spectra = np.load(ref_path, allow_pickle=True)
    query_smiles = np.array([
        s.get("smiles")
        for s in query_spectra
    ])
    ref_smiles = np.array([
        s.get("smiles")
        for s in ref_spectra
    ])
    query_sequences = tokenizer.tokenize_sequence(query_spectra)
    ref_sequences = tokenizer.tokenize_sequence(ref_spectra)
    print("tokenize the query and reference data success")
    query_dataset = TokenSequenceDataset(query_sequences)
    ref_dataset = TokenSequenceDataset(ref_sequences)

    ref_loader = DataLoader(
        ref_dataset,
        batch_size=loader_batch_size,
        shuffle=False
    )
    query_loader = DataLoader(
        query_dataset,
        batch_size=loader_batch_size,
        shuffle=False
    )
    model.eval()
    with torch.no_grad():
        ref_embedding = ModelEmbed(model, ref_loader, device)
        query_embedding = ModelEmbed(model, query_loader, device)

    infos, values = metric(
        k_metric,
        query_embedding, ref_embedding,
        query_smiles, ref_smiles,
        show_progress_bar,
        batch_size
    )
    hit_df = pd.DataFrame([values], columns=infos, index=[desc])
    return hit_df


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
