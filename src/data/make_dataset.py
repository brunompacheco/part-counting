# -*- coding: utf-8 -*-
from typing import List
import click
import json
import logging

import h5py
import numpy as np
import open3d as o3d

from dotenv import find_dotenv, load_dotenv
from pathlib import Path

load_dotenv(find_dotenv())


def add_simulation_to_dataset(dataset: h5py.Dataset, sim_dir: Path, sim_i: int
    ) -> h5py.Dataset:
    """Add simulation images to a HDF5 dataset.

    Args:
        dataset: dataset of an OPEN (in writtable mode) HDF5 file.
        sim_dir: directory containing simulation .png images.
        sim_i: index at which to put the images.

    Returns:
        dataset: modified dataset.
    """
    for img_fpath in sim_dir.glob('*.png'):
        img_j = int(img_fpath.name.replace('.png','').split('_')[-1])

        data = np.array(o3d.io.read_image(str(img_fpath)))
        # swap axis so that channels is the first axis
        data = np.moveaxis(data[:,:,1:3], -1, 0)

        dataset[sim_i,img_j] = data

    return dataset

def create_dataset_from_dirs(sim_dirs: List[Path], h5_file: h5py.File,
                             compression=None, dataset_name='renders',
                             log=print) -> h5py.Dataset:
    """Create h5py.Dataset in `h5_file` and add images in `sim_dirs`.

    Args:
        sim_dir: directories containing folders of simulations.
        h5_file: open (in writtable mode) HDF5 file.
        dataset_name: (optional) name of the dataset, default: 'renders'.
        compression: see h5py.File.create_dataset.
        log: logger function, default: print.

    Returns:
        dataset: created dataset with images from `sim_dirs`.
    """
    dataset = h5_file.create_dataset(
        dataset_name,
        (len(sim_dirs),101,2,512,512),
        dtype='uint8',
        chunks=(1,1,2,512,512),  # each image as a chunk
        compression=compression,
    )

    for i, sim_dir in enumerate(sim_dirs):
        log(f"adding {sim_dir.name} ({i+1}/{len(sim_dirs)}) to {dataset_name}")
        dataset = add_simulation_to_dataset(dataset, sim_dir, i)

    return dataset

@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.option('--compress', '-c', is_flag=True, default=False)
@click.option('--split', '-s', is_flag=True, default=False)
def main(input_dir, output_filepath, compress, split):
    """Turn render results images into an HDF5 dataset.

    Args:
        input_dir: path to the directory containing the simulation renders.
        output_filepath: filepath of the desired .hdf5 file.
        compress: wether to use compression (gzip, default compression level)
        or not on the HDF5.
    """
    logger = logging.getLogger(__name__)
    logger.info('making HDF5 dataset from raw data')

    input_dir = Path(input_dir)
    output_filepath = Path(output_filepath)

    sim_dirs = list(input_dir.glob('simulacao*'))

    if split:
        logger.info('making an 80/20 train-test split')

        train_sim_dirs = np.random.choice(sim_dirs, int(0.8 * len(sim_dirs)),
                                          replace=False)
        test_sim_dirs = [d for d in sim_dirs if d not in train_sim_dirs]

        split = {
            'train': [f.name for f in train_sim_dirs],
            'test': [f.name for f in test_sim_dirs],
        }
        with open(output_filepath.parent/'split.json', 'w') as f:
            json.dump(split, f)

    logger.info(f'creating HDF5 file at {output_filepath}')

    with h5py.File(str(output_filepath), "w") as h:
        compression = 'gzip' if compress else None

        if split:
            create_dataset_from_dirs(train_sim_dirs, h, compression=compression,
                                     dataset_name='train', log=logger.info)
            create_dataset_from_dirs(test_sim_dirs, h, compression=compression,
                                     dataset_name='test', log=logger.info)
        else:
            create_dataset_from_dirs(sim_dirs, h, compression=compression,
                                     log=logger.info)

    logger.info('finished!')
    logger.info(f'data saved at {output_filepath}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
