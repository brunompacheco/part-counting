"""Fundamental structures and utils.
"""
from pathlib import Path
from typing import Tuple, Union

import h5py
import numpy as np

from torch import Tensor
from torch.utils.data import Dataset, Subset
from torchvision import transforms as T


class FimacDataset(Dataset):
    def __init__(self, hdf5_fpath: Union[Path, str],
                 test=False) -> None:
        super().__init__()

        self._hdf5_file = h5py.File(str(hdf5_fpath), "r")
        self._hdf5_dataset_name = 'test' if test else 'train'

        self.transform = T.Compose([
            T.ToTensor(),
            # normalize required for pre-trained image models,
            # check https://pytorch.org/vision/stable/models.html
            T.Normalize(mean=[0.485, 0.456], std=[0.229, 0.224]),
        ])

    @property
    def _data(self) -> h5py.Dataset:
        return self._hdf5_file[self._hdf5_dataset_name]

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
        self._hdf5_file.close()

    def subset(self, frac: float) -> Subset:
        """Return a fraction of this dataset.

        Args:
            frac: percentage of the dataset to be returned.
        """
        assert frac <= 1 and frac > 0, '`frac` must be <=1 and >0'

        indices = np.random.choice(np.arange(len(self)),
                                   int(frac*len(self)),
                                   replace=False)

        return Subset(self, indices)
