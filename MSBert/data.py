from collections.abc import Sequence

import numpy as np
from matchms import Spectrum
from torch.utils.data import Dataset
from tqdm import tqdm

from type import TokenSequence, Peak, MetaData

SpecialToken = {
    "PAD": 0,
    "MASK": 1,
}


def get_smiles(sequences: list[TokenSequence], show_progress_bar: bool = True):
    pbar = sequences
    if show_progress_bar:
        pbar = tqdm(sequences, total=len(sequences), desc="get smiles")

    smiles_seq = []
    for sequence in pbar:
        smiles_seq.append(sequence["smiles"])
    return np.array(smiles_seq)


class TokenSequenceDataset(Dataset):
    def __init__(self, tokens: list[TokenSequence]) -> None:
        super().__init__()
        self.tokens = tokens
        self.length = len(tokens)

    def __getitem__(self, index: int):
        return self.tokens[index]["mz"], self.tokens[index]["intensity"]

    def __len__(self):
        return self.length


class Tokenizer:
    def __init__(self, n_decimals: int, max_len: int, show_progress_bar: bool = True) -> None:
        self.n_decimals = n_decimals
        self.max_len = max_len
        self.show_progress_bar = show_progress_bar

    def tokenize(self, s: Spectrum):
        metadata = self.get_metadata(s)
        mz = []
        intensity = []
        for peak in metadata["peaks"]:
            mz.append(self.get_word_index(peak["mz"]))
            intensity.append(peak["intensity"])

        mz = np.array(mz)
        intensity = np.array(intensity)
        intensity = intensity / max(intensity)
        if len(mz) < self.max_len:
            mz = np.pad(
                mz, (0, self.max_len - len(mz)),
                mode='constant', constant_values=SpecialToken['PAD']
            )

            intensity = np.pad(
                intensity, (0, self.max_len - len(intensity)),
                mode='constant', constant_values=0
            )

        return TokenSequence(
            mz=np.array(mz, np.int32),
            intensity=np.array(intensity, np.float32).reshape(1, -1)
        )

    def tokenize_sequence(self, spectra: Sequence[Spectrum]):
        sequences: list[TokenSequence] = []
        if self.show_progress_bar:
            pbar = tqdm(spectra, total=len(spectra))
        else:
            pbar = spectra
        for s in pbar:
            sequences.append(self.tokenize(s))

        return sequences

    def get_metadata(self, s: Spectrum):
        precursor_mz = s.get("precursor_mz")
        smiles = s.get("smiles")
        peaks = s.peaks.to_numpy
        intensity = peaks[:, 1]
        argmaxsort_index = np.sort(np.argsort(
            intensity)[::-1][:self.max_len - 1]
        )
        peaks = peaks[argmaxsort_index]
        packaged_peaks: list[Peak] = [
            Peak(
                mz=str(round(precursor_mz, self.n_decimals)),
                intensity=2
            )
        ]
        for mz, intensity in peaks:
            packaged_peaks.append(
                Peak(
                    mz=str(round(mz, self.n_decimals)),
                    intensity=intensity
                )
            )

        metadata = MetaData(
            smiles=smiles,
            peaks=packaged_peaks
        )
        return metadata

    def get_word_index(self, mz: str):
        multiple = pow(10, self.n_decimals)

        if '.' not in mz:
            return int(mz) * multiple

        integer_part, frac_part = mz.split(".")
        return int(integer_part) * multiple + int(frac_part) + len(SpecialToken)
