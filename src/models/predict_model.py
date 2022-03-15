import json
import logging
import os
import sys

from pathlib import Path
from time import time

import numpy as np
import open3d as o3d
import torch

from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm

from src.features.base import preprocess_box_for_cv, preprocess_box_for_dl
from src.models.base import load_dl_model, load_pso_model


# find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)  # load up the entries as environment variables

project_dir = Path(dotenv_path).parent


if __name__ == '__main__':
    ## get model
    model_list = ['pso', 'effnet', 'linreg', 'polyfit']

    try:
        model_name = sys.argv[1].lower()
    except IndexError:
        print('Please, select a MODEL for predicting.')
        print('MODEL can be any of ', model_list)

        sys.exit(-1)

    if model_name not in model_list:
        print('MODEL must be one of ', model_list)

        sys.exit(-1)

    ## get model methods
    model_loads = {
        'pso': load_pso_model,
        'effnet': load_dl_model,
        # TODO:
        # 'linreg': ,
        # 'polyfit': ,
    }
    model_load = model_loads[model_name]

    preprocesss = {
        'pso': preprocess_box_for_cv,
        'effnet': preprocess_box_for_dl,
        'linreg': preprocess_box_for_cv,
        'polyfit': preprocess_box_for_cv,
    }
    preprocess = preprocesss[model_name]

    ## init logging
    logging.basicConfig(level=logging.INFO, format=os.environ['log_fmt'])
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(os.environ['log_fmt'])

    fileh = logging.FileHandler(os.environ['log_file'], 'a')
    fileh.setFormatter(formatter)
    logger.addHandler(fileh)

    logger.info(f'predicting with {model_name} model')

    ## definitions
    preds_fpath = project_dir/f'reports/{model_name}_test_preds.json'
    times_fpath = project_dir/f'reports/{model_name}_test_times.json'

    if model_name == 'effnet':
        raw_data_dir = project_dir/'data/raw/render_results'
        test_images = list(raw_data_dir.glob('*/*.png'))

        with open(project_dir/'split.json', 'r') as f:
            split = json.load(f)

        test_images = [t for t in test_images if t.parent.name in split['test']]
    else:
        raw_data_dir = project_dir/'data/raw/render_results_imov_cam_mist_simple_only_validation'
        test_images = list(raw_data_dir.glob('*/*.exr'))

    if model_name == 'pso':
        # PSO is too slow to test on the whole set
        frac = 0.1
        test_images = np.random.choice(test_images, int(frac * len(test_images)),
                                       replace=False)

    ## load predictions already made
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

    logger.info(f'loading model')

    model = model_load()

    logger.info(f'storing results at {preds_fpath}.')
    logger.info(f'storing performance (time) at {times_fpath}.')
    for img_fpath in tqdm(test_images):
        # avoid overwriting
        if img_fpath.name in preds.keys():
            continue

        logger.debug(str(img_fpath))

        start_time = time()

        data = preprocess(img_fpath)

        preds[img_fpath.name] = model(data)
        times[img_fpath.name] = time() - start_time

        with open(preds_fpath, 'w') as f:
            json.dump(preds, f)

        with open(times_fpath, 'w') as f:
            json.dump(times, f)
