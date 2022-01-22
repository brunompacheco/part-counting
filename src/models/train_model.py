import os
import click
import logging
import random

from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn as nn
import wandb

from dotenv import load_dotenv, find_dotenv
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler

from src.data import FimacDataset
from src.models import TestNet
from src.models.base import EffNetRegressor, get_model_class

# find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)  # load up the entries as environment variables

project_dir = Path(dotenv_path).parent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


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

def train_pass(net, dataloader, criterion, optimizer, scaler):
    train_loss = 0
    net.train()
    with torch.set_grad_enabled(True):
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            with autocast():
                y_hat = net(X)
                loss = criterion(y_hat.squeeze(), y.type(torch.float32))

            scaler.scale(loss).backward()

            train_loss += loss.item() * len(y)

            scaler.step(optimizer)
            scaler.update()

    return train_loss / len(dataloader.dataset)  # scales to data size

def validation_pass(net, dataloader, criterion):
    val_loss = 0
    val_MAE = 0
    val_acc = 0
    net.eval()
    with torch.set_grad_enabled(False):
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            with autocast():
                y_hat = net(X)
                loss_value = criterion(
                    y_hat.squeeze(),
                    y.type(torch.float32)
                ).item()

            val_loss += loss_value * len(y)  # scales to data size
            val_MAE += int_MAE(y, y_hat.squeeze()) * len(y)
            val_acc += acc(y, y_hat.squeeze()) * len(y)

    len_data = len(dataloader.dataset)
    return val_loss / len_data, val_MAE / len_data, val_acc / len_data

def timeit(fun):
    def fun_(*args, **kwargs):
        start_time = time()
        f_ret = fun(*args, **kwargs)
        end_time = time()

        return end_time - start_time, f_ret

    return fun_

@click.command()
@click.argument('epochs', type=click.INT)
@click.option('--frac', type=click.FLOAT, default=1.0, help=(
    'Fraction of the data to be used (train + validation). This option is '
    'overwritten if a checkpoint is loaded.'
))
@click.option('--resume', type=click.STRING,
              help='Wandb run id of the experiment to resume.')
def main(epochs, frac, resume):
    """Train a model in the Fimac render dataset.
    """
    logger = logging.getLogger(__name__)

    fileh = logging.FileHandler(os.environ['log_file'], 'a')
    fileh.setFormatter(logging.Formatter(os.environ['log_fmt']))
    logger.addHandler(fileh)

    if resume is not None:
        run_id = resume
        wandb.init(
            project="part-counting",
            entity="brunompac",
            id=run_id,
            resume='must',
        )
        checkpoint_file = wandb.restore('checkpoint.tar')
        checkpoint = torch.load(checkpoint_file.name)

        epoch = checkpoint['epoch'] + 1

        logger.info(f'Resuming training of {wandb.run.name} at epoch {epoch}')

        # I'll just fix number of epochs for each training (experiments POV)
        epochs = wandb.run.config['epochs']

        # load hyperparameters
        lr = wandb.run.config['learning_rate']
        frac = wandb.run.config['data_fraction']
        batch_size = wandb.run.config['batch_size']

        criterion = eval(f"nn.{wandb.run.config['loss_func']}()")

        net = get_model_class(wandb.run.config['model'])().to(device)
        net.load_state_dict(checkpoint['model_state_dict'])

        optimizer = eval(f"torch.optim.{wandb.run.config['optimizer']}")
        optimizer = optimizer(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=lr
        )
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logger.info('Setting up training')

        epoch = 0

        # hyperparameters
        batch_size = 16
        lr = 0.05
        criterion = nn.L1Loss()

        net = EffNetRegressor().to(device)

        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=lr
        )

        # initialize wandb
        wandb.init(
            project="part-counting",
            entity="brunompac",
            config={
                "learning_rate": lr,
                "data_fraction": frac,
                "epochs": epochs,
                "batch_size": batch_size,
                "model": type(net).__name__,
                "optimizer": type(optimizer).__name__,
                "loss_func": type(criterion).__name__,
            },
        )

    wandb.watch(net)

    logger.info(f"Wandb set up. Run ID: {wandb.run.id}")

    logger.info(f"Training {wandb.run.config['model']} model")
    logger.info(f"optimizer = {wandb.run.config['optimizer']}")
    logger.info(f"learning rate = {wandb.run.config['learning_rate']}")
    logger.info(f"batch size = {wandb.run.config['batch_size']}")
    logger.info(f"loss function = {wandb.run.config['loss_func']}")

    logger.info('Preparing data')
    dataset = FimacDataset(project_dir/'data/interim/renders.hdf5')

    split_size = .8
    train_data, val_data = dataset.subset_split(frac, split_size)

    # instantiate DataLoaders
    train_dataloader = DataLoader(train_data, batch_size=batch_size,
                                  shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=32)

    scaler = GradScaler()

    while epoch < epochs:
        logger.info(f"Epoch {epoch} started ({epoch+1}/{epochs})")
        epoch_start_time = time()

        # train
        train_time, train_loss = timeit(train_pass)(net, train_dataloader,
                                                    criterion, optimizer,
                                                    scaler)

        logger.info(f"Training pass took {train_time:.3f} seconds")
        logger.info(f"Training loss = {train_loss}")

        # validation
        val_time, (val_loss, val_MAE, val_acc) = timeit(validation_pass)(
            net,
            val_dataloader,
            criterion,
        )

        logger.info(f"Validation pass took {val_time:.3f} seconds")
        logger.info(f"Validation loss = {val_loss}")
        logger.info(f"Validation MAE = {val_MAE}")
        logger.info(f"Validation accuracy = {100 * val_acc:.2f} %")

        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_MAE": val_MAE,
            "val_acc": val_acc,
        }, step=epoch, commit=True)

        logger.info(f"Saving checkpoint")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        # torch.save(checkpoint, checkpoint_fpath)
        torch.save(checkpoint, Path(wandb.run.dir)/'checkpoint.tar')
        wandb.save('checkpoint.tar')

        epoch_end_time = time()
        logger.info(
            f"Epoch {epoch} finished and took "
            f"{epoch_end_time - epoch_start_time:.2f} seconds"
        )

        epoch += 1

    logger.info(f"Saving model")
    torch.save(net.state_dict(), Path(wandb.run.dir)/'model.tar')

    wandb.finish()
    logger.info('Training finished!')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=os.environ['log_fmt'])

    main()
