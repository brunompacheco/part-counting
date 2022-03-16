import json
import logging
import os

from pathlib import Path
from time import time

import numpy as np
import open3d as o3d
import torch

from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm

from model import EffNetRegressor, load_from_wandb


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

    preds_fpath = project_dir/'models/effnet_test_preds.json'
    times_fpath = project_dir/'models/effnet_test_times.json'
    raw_data_dir = project_dir/'data/raw/render_results'

    test_images = list(raw_data_dir.glob('*/*.exr'))

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

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    logger.info(f'loading model in {device}')

    net = load_from_wandb(
        EffNetRegressor(freeze=False, pretrained=False, effnet_size='b0', hidden_layer_size=60),
        '13mhcjex',
        'part-counting-fine-tuning',
    )
    net.eval().to(device)

    logger.info(f'storing results at {preds_fpath}.')
    logger.info(f'storing performance (time) at {times_fpath}.')
    for img_fpath in tqdm(test_images):
        # avoid overwriting
        if img_fpath.name in preds.keys():
            continue

        logger.debug(str(img_fpath))

        start_time = time()

        ## LOADING
        data = np.array(o3d.io.read_image(str(img_fpath)))
        # swap axis so that channels is the first axis
        data = np.moveaxis(data[:,:,1:3], -1, 0)

        ## PREDICTING
        X = torch.from_numpy(data).unsqueeze(0)
        X = X.type(torch.FloatTensor)
        X.to(device)
        with torch.no_grad():
            y = net(X)

        y = y.item()

        # STORING
        preds[img_fpath.name] = y
        times[img_fpath.name] = time() - start_time

        with open(preds_fpath, 'w') as f:
            json.dump(preds, f)

        with open(times_fpath, 'w') as f:
            json.dump(times, f)
