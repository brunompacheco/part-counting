"""Fundamental structures and utils.
"""
import h5py

from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Union

class FimacDataset(Dataset):
    def __init__(self, hdf5_fpath: Union[Path, str],
                 hdf5_dataset_name='renders') -> None:
        super().__init__()

        self._data = h5py.File(str(hdf5_fpath), "r")[hdf5_dataset_name]

    def __len__(self) -> int:
        return self._data.shape[0] * self._data.shape[1]

    def __getitem__(self, index) -> Tuple[np.ndarray, int]:
        i = index // self._data.shape[1]  # render index
        j = index % self._data.shape[1]  # render step (# of parts) index

        image = self._data[i,j,:,:,:]
        label = j

        return image, label

    def __del__(self) -> None:
        self._data.close()
        super().__del__()
