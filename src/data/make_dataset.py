# -*- coding: utf-8 -*-
import click
import logging

import h5py
import numpy as np
import open3d as o3d

from dotenv import find_dotenv, load_dotenv
from pathlib import Path

load_dotenv(find_dotenv())


def add_simulation_to_dataset(dataset: h5py.Dataset, sim_dir: Path
    ) -> h5py.Dataset:
    """Add simulation images to a HDF5 dataset.

    Args:
        dataset: dataset of an OPEN (in writtable mode) HDF5 file.
        sim_dir: directory containing simulation .png images.

    Returns:
        dataset: modified dataset.
    """
    sim_i = int(sim_dir.name.replace('simulacao',''))

    # in case the simulation is 
    if sim_i > dataset.shape[0]:
        dataset.resize(dataset.shape[0]+1, axis=0)

    for img_fpath in sim_dir.glob('*.png'):
        img_j = int(img_fpath.name.replace('.png','').split('_')[-1])

        data = np.array(o3d.io.read_image(str(img_fpath)))
        # swap axis so that channels is the first axis
        data = np.moveaxis(data[:,:,1:3], -1, 0)

        dataset[sim_i,img_j] = data

    return dataset

@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.option('--compress', '-c', default=False)
def main(input_dir, output_filepath, compress):
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

    logger.info(f'creating HDF5 file at {output_filepath}')

    with h5py.File(str(output_filepath), "w") as h:
        compression = 'gzip' if compress else None

        # create resizable dataset just in case not all images are available
        dataset = h.create_dataset(
            'renders',
            (len(sim_dirs),101,2,512,512),
            maxshape=(1000,101,2,512,512),
            dtype='uint8',
            chunks=(1,1,2,512,512),  # each image as a chunk
            compression=compression,
        )

        for i, sim_dir in enumerate(sim_dirs):
            logger.info(f"adding {sim_dir.name} ({i+1}/{len(sim_dirs)})")
            dataset = add_simulation_to_dataset(dataset, sim_dir)

        logger.info('finished!')
        logger.info(f'data saved at {output_filepath}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
