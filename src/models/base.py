from pathlib import Path
from typing import Callable

from dotenv import load_dotenv, find_dotenv

from src.models.regression import estimate_volume

# find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)  # load up the entries as environment variables

project_dir = Path(dotenv_path).parent

def load_pso_model(part_fpath: Path = project_dir/'data/raw/part.stl',
                   voxel_size: float = 0.02) -> Callable:
    """Sets up the PSO model for prediction. Returns a callable that does so.

    Args:
        part_fpath: Filepath to the .stl model file.

    Returns:
        model: Callable that returns the (estimated) number of parts in the box
        (input argument).
    """
    from src.features.base import load_part_model
    from src.models.pso import dig_and_predict

    part = load_part_model(part_fpath)

    model = lambda box: dig_and_predict(box, part, voxel_size)

    return model

def load_dl_model(run_id: str = '13mhcjex',
                  wandb_project: str = 'part-counting-fine-tuning',
                  device=None) -> Callable:
    """Load DL model for prediction. Returns a callable that does so.

    Args:
        run_id: W&B run id of the desired model.
        wandb_project: Name of W&B project where to fetch the run from.
        device: Where to store and run the model. See torch.device. If None
        (default) will use `cuda` if available.

    Returns:
        model: Callable that returns number of parts in the box.
    """
    import torch

    from src.models.model import EffNetRegressor, load_from_wandb

    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    net = load_from_wandb(
        EffNetRegressor(freeze=False, pretrained=False, effnet_size='b0', hidden_layer_size=60),
        run_id,
        wandb_project,
    )
    net.eval().to(device)

    def model(X: torch.Tensor):
        X = X.to(device)
        with torch.no_grad():
            y = net(X)

        return y.item()

    return model

def load_linreg_model(
        model_fpath: Path = project_dir/'models/linear_regression.pkl',
        voxel_size: float = 0.005,
    ) -> Callable:
    """Load Linear Regression model for prediction. Returns a callable that does so.

    Args:
        model_fpaht: Filepath of the .pkl file containing an sklearn's
        LinearRegresion model already fitted.
        voxel_size: Resolution of the voxel grid used to estimate the volume.

    Returns:
        model: Callable that returns number of parts in the box.
    """
    from joblib import load

    import numpy as np


    with open(model_fpath, 'rb') as f:
        lr = load(f)

    def model(box):
        vol = estimate_volume(box, voxel_size=voxel_size)

        X = np.array(vol).reshape(-1,1)

        return lr.predict(X)[0] * 100

    return model

def load_polyfit_model(
        model_fpath: Path = project_dir/'models/polynomial_fit.pkl',
        voxel_size: float = 0.005,
    ) -> Callable:
    """Load polynomial fitted to the volume-parts curve.
    
    Args:
        model_fpaht: Filepath of the .pkl file containing the polynomial's
        coefficients.
        voxel_size: Resolution of the voxel grid used to estimate the volume.

    Returns:
        model: Callable that returns number of parts in the box.
    """
    from joblib import load

    with open(model_fpath, 'rb') as f:
        p = load(f)

    def model(box):
        vol = estimate_volume(box, voxel_size=voxel_size)

        return sum([p[i] * vol**i for i in range(len(p))])

    return model
