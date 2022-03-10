import json
import logging
import os

from pathlib import Path
from time import time

import numpy as np
import open3d as o3d

from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm

from src.data.rgbd import load_rgbd
from src.data.pcd import load_pcd
from src.features.cropping import mask_selection_volume, box_mask_from_rgbd
from pso import dig_and_predict


# find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)  # load up the entries as environment variables

project_dir = Path(dotenv_path).parent


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=os.environ['log_fmt'])
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(os.environ['log_fmt'])

    fileh = logging.FileHandler(os.environ['log_file'], 'a')
    fileh.setFormatter(formatter)
    logger.addHandler(fileh)

    preds_fpath = project_dir/'models/pso_test_preds.json'
    times_fpath = project_dir/'models/pso_test_times.json'
    raw_data_dir = project_dir/'data/raw/render_results_imov_cam_mist_simple_only_validation'

    test_images = list(raw_data_dir.glob('*/*.exr'))

    frac = 0.1
    test_images = np.random.choice(test_images, int(frac * len(test_images)),
                                   replace=False)

    # load predictions already made
    if not preds_fpath.exists():
        preds = dict()
        times = dict()
        logger.info(f'starting predictions of {len(test_images)} images.')
    else:
        with open(preds_fpath, 'r') as f:
            preds = json.load(f)
        with open(times_fpath, 'r') as f:
            times = json.load(f)
        logger.info(f'continuing predictions, already done {len(preds)}/{len(test_images)}.')

    logger.info(f'storing results at {preds_fpath}.')
    logger.info(f'storing performance (time) at {times_fpath}.')

    for img_fpath in tqdm(test_images):
        # avoid overwriting
        if img_fpath.name in preds.keys():
            continue

        logger.debug(str(img_fpath))

        start_time = time()

        ## LOADING
        rgbd = load_rgbd(img_fpath)

        box_mask = box_mask_from_rgbd(rgbd)

        vol = mask_selection_volume(rgbd, box_mask)

        pcd = load_pcd(rgbd)

        box = vol.crop_point_cloud(pcd)

        part_fpath = project_dir/'data/raw/part.stl'

        part_mesh = o3d.io.read_triangle_mesh(str(part_fpath), enable_post_processing=True)

        part_mesh.paint_uniform_color([1., 0., 0.,])

        part = part_mesh.sample_points_uniformly(number_of_points=10000)

        part_points = np.array(part.points) / 1000  # mm to meter conversion
        part_points = part_points + np.array([0,0,0.3])
        part_points = o3d.utility.Vector3dVector(part_points)
        part.points = part_points

        # PREDICTING
        n = dig_and_predict(box, part, 0.02)
 
        # STORING
        preds[img_fpath.name] = n
        times[img_fpath.name] = time() - start_time

        with open(preds_fpath, 'w') as f:
            json.dump(preds, f)

        with open(times_fpath, 'w') as f:
            json.dump(times, f)
