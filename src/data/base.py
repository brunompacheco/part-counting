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

    def _subset(self, frac: float) -> np.array:
        assert frac <= 1 and frac > 0, '`frac` must be <=1 and >0'

        renders = np.arange(self._data.shape[0])
        parts_per_render = self._data.shape[1]

        frac_renders = np.random.choice(renders, int(frac*len(renders)),
                                        replace=False)

        indices = [np.arange(parts_per_render * render,
                             parts_per_render * (render + 1))
                   for render in frac_renders]
        indices = np.concatenate(indices)

        return indices

    def subset(self, frac: float) -> Subset:
        """Return a fraction of this dataset without contamining the remaining.

        The fraction is extracted in such a way that for a given render, either
        all of its images are in the subset or none is.

        Args:
            frac: percentage of the dataset to be returned.
        """
        indices = self._subset(frac)

        return Subset(self, indices)
    
    def split(self, split_size: float) -> Tuple[Subset, Subset]:
        """Generate a split (train-test) of the dataset.

        Partitions the dataset into two subsets in such a way that for all
        renders, their images are all in either one of the subsets, that is,
        avoiding two subsets containing (distinct) images of the same render.

        Args:
            split_size: split_size of the renders that the first subset will contain.
        """
        assert split_size <= 1 and split_size > 0, '`split_size` must be <=1 and >0'

        indices_a = self._subset(split_size)

        indices_b = filter(lambda i: i not in indices_a, np.arange(len(self)))

        return Subset(self, indices_a), Subset(self, indices_b)
    
    def subset_split(self, frac: float, split_size: float) -> Tuple[Subset, Subset]:
        """Generate a split (train-test) of a fraction of the dataset.

        A combination of `self.subset()` and `self.split`.

        Args:
            frac: percentage of the dataset to be used for the split.
            split_size: controls percentage of renders in the dataset fraction
            that will be in the first subset.
        """
        assert frac <= 1 and frac > 0, '`frac` must be <=1 and >0'

        renders = np.arange(self._data.shape[0])
        parts_per_render = self._data.shape[1]

        frac_renders = np.random.choice(
            renders,
            int(frac*len(renders)),
            replace=False
        )

        renders_a = np.random.choice(
            frac_renders,
            int(split_size*len(frac_renders)),
            replace=False
        )
        renders_b = filter(lambda r: r not in renders_a, frac_renders)

        indices_a = np.concatenate([
            np.arange(parts_per_render * render,
                      parts_per_render * (render + 1))
            for render in renders_a
        ])
        indices_b = np.concatenate([
            np.arange(parts_per_render * render,
                      parts_per_render * (render + 1))
            for render in renders_b
        ])

        return Subset(self, indices_a), Subset(self, indices_b)
