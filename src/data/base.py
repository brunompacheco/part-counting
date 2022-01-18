"""Fundamental structures and utils.
"""
from pathlib import Path
from typing import Tuple, Union

import h5py
import numpy as np

from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms as T

class FimacDataset(Dataset):
    def __init__(self, hdf5_fpath: Union[Path, str],
                 hdf5_dataset_name='renders') -> None:
        super().__init__()

        self._data = h5py.File(str(hdf5_fpath), "r")[hdf5_dataset_name]

        self.transform = T.Compose([
            T.ToTensor(),
            # normalize required for pre-trained image models,
            # check https://pytorch.org/vision/stable/models.html
            T.Normalize(mean=[0.485, 0.456], std=[0.229, 0.224]),
        ])

    def __len__(self) -> int:
        return self._data.shape[0] * self._data.shape[1]

    def __getitem__(self, index) -> Tuple[Tensor, int]:
        i = index // self._data.shape[1]  # render index
        j = index % self._data.shape[1]  # render step (# of parts) index

        image = self._data[i,j,:,:,:]
        # torchvision's normalize expects numpy array in shape (H x W x C)
        image = self.transform(np.moveaxis(image, 0, -1))

        label = j

        return image, label

    def __del__(self) -> None:
        self._data.close()
        super().__del__()
