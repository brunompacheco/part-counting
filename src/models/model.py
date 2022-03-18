"""Network definitions and utilitary functions to load them.

    Typical usage example:

    >>> blank_net = EffNetRegressor(
    ...     freeze=False,
    ...     pretrained=False,
    ...     effnet_size='b0',
    ...     hidden_layer_size=60,
    ... )
    >>> net = load_from_wandb(blank_net, '13mhcjex', 'part-counting-fine-tuning')
    >>> y_hat = net(X)
"""
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb


class TestNet(nn.Module):
    """Simple NN used to test the routines.
    """
    def __init__(self):
        super().__init__()

        # two basic convolution layers and two fully-connected for regression
        self.conv1 = nn.Conv2d(2, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(1048576, 128)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Pass data through dropout1
        x = self.dropout1(x)

        # Flatten x with start_dim=1
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return x

class EffNetRegressor(nn.Module):
    """Regression model that uses EfficientNet as a (fixed) feature extractor.

    An input layer is added (Conv2d+BN) to adapt our 2-channel image to the 3
    channels EffNet is expecting. Also, the classifier is replaced by a
    regressor, for obvious reasons.

    Attributes:
        freeze: Freeze the EfficientNet weights *except* for the `stem` block.
        Used for transfer learning.
        pretrained: If true, loads EfficientNet weights.
        effnet_size: Size of the EfficientNet to use. Must be `b0` or `b4`.
        hidden_layer_size: Number of node in the FC hidden layer in the
        regressor.
    """
    def __init__(self, freeze=True, pretrained=True, effnet_size='b0',
                 hidden_layer_size=40):
        super().__init__()

        self.input = nn.Sequential(
            # adapt our 2-channel images to effnet 3 channels
            nn.Conv2d(2, 3, 3, 1),
            nn.BatchNorm2d(3),
        )

        self.effnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub',
                                     f'nvidia_efficientnet_{effnet_size}',
                                     pretrained=pretrained)

        # change EffNet classifier to a regressor
        sequential_in = 1280 if effnet_size == 'b0' else 1792
        self.effnet.classifier.fc = nn.Sequential(
            nn.Linear(sequential_in, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, 1),
        )

        if freeze:
            # freeze EffNet's layers and features
            for param in self.effnet.layers.parameters():
                param.requires_grad = False
            for param in self.effnet.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.input(x)
        x = self.effnet(x)

        return x

def get_model_class(model: str) -> nn.Module:
    return eval(model)

def load_model(model_fpath: Union[str, Path]) -> nn.Module:
    """Instantiate model and load parameters.

    Args:
        model_fpath: path to the saved `state_dict` of a model. Expected in
        format `{wandb_run_name}__{model_class_name}.pth`.
    """
    model_fpath = Path(model_fpath)

    model_name = model_fpath.name.split('__')[-1].split('.')[0]
    model = get_model_class(model_name)()
    model.load_state_dict(torch.load(model_fpath))

    return model

def load_from_wandb(net: nn.Module, run_id: str,
                    project='part-counting-transfer-learning'):
    best_model_file = wandb.restore(
        'model_best.pth',
        run_path=f"brunompac/{project}/{run_id}",
    )
    net.load_state_dict(torch.load(best_model_file.name))

    return net
