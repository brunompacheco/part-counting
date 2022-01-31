import logging
from pathlib import Path
import random

from time import time
import numpy as np

import torch
import torch.nn as nn

from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import wandb
from dotenv import load_dotenv, find_dotenv

from src.data import FimacDataset
from src.models import get_model_class

# find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)  # load up the entries as environment variables

project_dir = Path(dotenv_path).parent


def int_MAE(y, y_hat):
    y_hat_int = y_hat.squeeze().type(torch.IntTensor) + 0.5
    y_int = y.type(torch.IntTensor)

    err = y_int - y_hat_int
    err = err.abs()
    err = err.type(torch.FloatTensor).mean()

    return err.item()

def acc(y, y_hat):
    y_hat_int = (y_hat.squeeze() + 0.5).type(torch.IntTensor)
    y_int = y.type(torch.IntTensor)

    matches = y_int == y_hat_int

    return matches.sum().item() / len(matches)

def timeit(fun):
    def fun_(*args, **kwargs):
        start_time = time()
        f_ret = fun(*args, **kwargs)
        end_time = time()

        return end_time - start_time, f_ret

    return fun_

class Trainer():
    """Generice trainer for PyTorch NNs.

    Attributes:
        net: the neural network to be trained.
        epochs: number of epochs to train the network.
        lr: learning rate.
        optimizer: optimizer (name of a optimizer inside `torch.optim`).
        loss_func: a valid PyTorch loss function.
        lr_scheduler: if a scheduler is to be used, provide the name of a valid
        `torch.optim.lr_scheduler`.
        lr_scheduler_params: parameters of selected `lr_scheduler`.
        frac: fraction of the data to be used (train + validation).
        batch_size: batch_size for training.
        device: see `torch.device`.
        logger: see `logging`.
        random_seed: if not None (default = 42), fixes randomness for Python,
        NumPy as PyTorch (makes trainig reproducible).
    """
    def __init__(self, net: nn.Module, epochs=5, lr= 0.01,
                 optimizer: str = 'SGD', loss_func: str = 'L1Loss',
                 lr_scheduler: str = None, lr_scheduler_params: dict = None,
                 frac=1.0, batch_size=16, device=None, logger=None,
                 random_seed=42) -> None:
        self._is_initalized = False

        self._e = 0  # inital epoch

        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

        if device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        self.net = net.to(self.device)
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params

        self.frac = frac

        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.l = logging.getLogger(__name__)
        else:
            self.l = logger

        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
            torch.manual_seed(random_seed)

        self.best_val = float('inf')

    @classmethod
    def load_trainer(cls, run_id: str, logger=None):
        """Load a previously initialized trainer from wandb.

        Loads checkpoint from wandb and create the instance.

        Args:
            run_id: valid wandb id.
            logger: same as the attribute.
        """
        wandb.init(
            project="part-counting",
            entity="brunompac",
            id=run_id,
            resume='must',
        )

        # load checkpoint file
        checkpoint_file = wandb.restore('checkpoint.tar')
        checkpoint = torch.load(checkpoint_file.name)

        # load model
        net = get_model_class(wandb.run.config['model'])()
        net = net.to(wandb.config['device'])
        net.load_state_dict(checkpoint['model_state_dict'])

        # fix for older versions
        if 'lr_scheduler' not in wandb.config.keys():
            wandb.config['lr_scheduler'] = None
            wandb.config['lr_scheduler_params'] = None

        # create trainer instance
        self = cls(
            epochs=wandb.config['epochs'],
            net=net,
            lr=wandb.config['learning_rate'],
            optimizer=wandb.config['optimizer'],
            loss_func=wandb.config['loss_func'],
            lr_scheduler=wandb.config['lr_scheduler'],
            lr_scheduler_params=wandb.config['lr_scheduler_params'],
            frac=wandb.config['data_fraction'],
            batch_size=wandb.config['batch_size'],
            device=wandb.config['device'],
            logger=logger,
            random_seed=wandb.config['random_seed'],
        )

        if 'best_val' in checkpoint.keys():
            self.best_val = checkpoint['best_val']

        self._e = checkpoint['epoch'] + 1

        self.l.info(f'Resuming training of {wandb.run.name} at epoch {self._e}')

        # load optimizer
        Optimizer = eval(f"torch.optim.{self.optimizer}")
        self._optim = Optimizer(
            filter(lambda p: p.requires_grad, self.net.parameters()),
            lr=self.lr
        )
        self._optim.load_state_dict(checkpoint['optimizer_state_dict'])

        return self

    def setup_training(self):
        self.l.info('Setting up training')

        Optimizer = eval(f"torch.optim.{self.optimizer}")
        self._optim = Optimizer(
            filter(lambda p: p.requires_grad, self.net.parameters()),
            lr=self.lr
        )

        if self.lr_scheduler is not None:
            Scheduler = eval(f"torch.optim.lr_scheduler.{self.lr_scheduler}")
            self._scheduler = Scheduler(self._optim, **self.lr_scheduler_params)

        self._loss_func = eval(f"nn.{self.loss_func}()")

        self.l.info('Initializing wandb.')
        self.initialize_wandb()

        self.l.info('Preparing data')
        self.prepare_data()

        self._is_initalized = True

    def initialize_wandb(self):
        wandb.init(
            project="part-counting",
            entity="brunompac",
            config={
                "learning_rate": self.lr,
                "data_fraction": self.frac,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "model": type(self.net).__name__,
                "optimizer": self.optimizer,
                "lr_scheduler": self.lr_scheduler,
                "lr_scheduler_params": self.lr_scheduler_params,
                "loss_func": self.loss_func,
                "random_seed": self.random_seed,
                "device": self.device,
            },
        )

        wandb.watch(self.net)

        self._id = wandb.run.id

        self.l.info(f"Wandb set up. Run ID: {self._id}")

    def prepare_data(self, split_size=.8):
        self.dataset = FimacDataset(project_dir/'data/interim/renders.hdf5')

        train_data, val_data = self.dataset.subset_split(self.frac, split_size)

        # instantiate DataLoaders
        self._dataloader = {
            'train': DataLoader(train_data, batch_size=self.batch_size,
                                shuffle=True),
            'val': DataLoader(val_data, batch_size=32),
        }

    def run(self):
        if not self._is_initalized:
            self.setup_training()

        scaler = GradScaler()
        while self._e < self.epochs:
            self.l.info(f"Epoch {self._e} started ({self._e+1}/{self.epochs})")
            epoch_start_time = time()

            # train
            train_time, train_loss = timeit(self.train_pass)(scaler)

            self.l.info(f"Training pass took {train_time:.3f} seconds")
            self.l.info(f"Training loss = {train_loss}")

            # validation
            val_time, (val_loss, val_MAE, val_acc) = timeit(self.validation_pass)()

            self.l.info(f"Validation pass took {val_time:.3f} seconds")
            self.l.info(f"Validation loss = {val_loss}")
            self.l.info(f"Validation MAE = {val_MAE}")
            self.l.info(f"Validation accuracy = {100 * val_acc:.2f} %")

            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_MAE": val_MAE,
                "val_acc": val_acc,
            }, step=self._e, commit=True)

            self.l.info(f"Saving checkpoint")
            self.save_checkpoint()

            if val_MAE < self.best_val:
                self.l.info(f"Saving best model")
                self.save_model(name='model_best')

                self.best_val = val_MAE

            epoch_end_time = time()
            self.l.info(
                f"Epoch {self._e} finished and took "
                f"{epoch_end_time - epoch_start_time:.2f} seconds"
            )

            self._e += 1

        self.l.info(f"Saving model")
        self.save_model(name='model_last')

        wandb.finish()
        self.l.info('Training finished!')

    def train_pass(self, scaler):
        train_loss = 0
        self.net.train()
        with torch.set_grad_enabled(True):
            for X, y in self._dataloader['train']:
                X = X.to(self.device)
                y = y.to(self.device)

                self._optim.zero_grad()

                with autocast():
                    y_hat = self.net(X)
                    loss = self._loss_func(
                        y_hat.squeeze(),
                        y.type(torch.float32) / 100  # scale to 0-1 range
                    )

                scaler.scale(loss).backward()

                train_loss += loss.item() * len(y)

                scaler.step(self._optim)
                scaler.update()

            if self.lr_scheduler is not None:
                self._scheduler.step()

        # scale to data size
        train_loss = train_loss / len(self._dataloader['train'].dataset)

        return train_loss

    def validation_pass(self):
        val_loss = 0
        val_MAE = 0
        val_acc = 0

        self.net.eval()
        with torch.set_grad_enabled(False):
            for X, y in self._dataloader['val']:
                X = X.to(self.device)
                y = y.to(self.device)

                with autocast():
                    y_hat = self.net(X)
                    loss_value = self._loss_func(
                        y_hat.squeeze(),
                        y.type(torch.float32) / 100  # scale to 0-1 range
                    ).item()

                val_loss += loss_value * len(y)  # scales to data size

                y_hat = y_hat.squeeze() * 100  # scale to 0-100 range
                val_MAE += int_MAE(y, y_hat) * len(y)
                val_acc += acc(y, y_hat) * len(y)

        # scale to data size
        len_data = len(self._dataloader['val'].dataset)
        val_loss = val_loss / len_data
        val_MAE = val_MAE / len_data
        val_acc = val_acc / len_data

        return val_loss, val_MAE, val_acc

    def save_checkpoint(self):
        checkpoint = {
            'epoch': self._e,
            'best_val': self.best_val,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self._optim.state_dict(),
        }

        torch.save(checkpoint, Path(wandb.run.dir)/'checkpoint.tar')
        wandb.save('checkpoint.tar')

    def save_model(self, name='model'):
        fname = f"{name}.pth"
        fpath = Path(wandb.run.dir)/fname

        torch.save(self.net.state_dict(), fpath)
        wandb.save(fname)

        return fpath
