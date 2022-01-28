import os
import logging

from pathlib import Path

from dotenv import load_dotenv, find_dotenv

from model import EffNetRegressor
from trainer import Trainer

# find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)  # load up the entries as environment variables

project_dir = Path(dotenv_path).parent


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=os.environ['log_fmt'])
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(os.environ['log_fmt'])

    # streamh = logging.StreamHandler(sys.stdout)
    # streamh.setFormatter(formatter)
    # logger.addHandler(streamh)

    fileh = logging.FileHandler(os.environ['log_file'], 'a')
    fileh.setFormatter(formatter)
    logger.addHandler(fileh)

    logger.info('start')

    Trainer(
        EffNetRegressor(),
        epochs=20,
        lr=0.005,
        frac=1.0,
        lr_scheduler='ExponentialLR',
        lr_scheduler_params={'gamma': 0.9,},
        logger=logger,
    ).run()
