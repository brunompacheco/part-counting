import click
import logging

from pathlib import Path
from time import time

import torch
import torch.nn as nn
import wandb

from dotenv import load_dotenv, find_dotenv
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler

from src.data import FimacDataset
from src.models import TestNet

# find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)  # load up the entries as environment variables

project_dir = Path(dotenv_path).parent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def int_MAE(y, y_hat):
    err = y.type(torch.IntTensor) - y_hat.type(torch.IntTensor)    
    err = err.abs()
    err = err.type(torch.FloatTensor).mean()

    return err.item()

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

            train_loss += loss.item() / len(y)  # scales to data size

            scaler.step(optimizer)
            scaler.update()

    return train_loss

def validation_pass(net, dataloader, criterion):
    val_loss = 0
    val_MAE = 0
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

            val_loss += loss_value / len(y)  # scales to data size
            val_MAE += int_MAE(y, y_hat.squeeze()) / len(y)

    return val_loss,val_MAE

@click.command()
@click.argument('epochs', type=click.INT)
def main(epochs):
    """Train a model in the Fimac render dataset.
    """
    logger = logging.getLogger(__name__)

    # training hyperparameters
    batch_size = 16
    lr = 0.001
    criterion = nn.L1Loss()

    dataset = FimacDataset(project_dir/'data/interim/renders.hdf5')

    # split train into train and validation
    train_val_split = .8
    train_size = int(len(dataset) * train_val_split)
    train_data, val_data = random_split(
        dataset,
        (train_size, len(dataset) - train_size),
    )

    # instantiate DataLoaders
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=32)

    # instantiate model
    net = TestNet().to(device)

    # more "hyperparameters"
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    scaler = GradScaler()

    config = {
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "model": type(net).__name__,
        "optimizer": type(optimizer).__name__,
        "loss_func": type(criterion).__name__,
    }

    logger.info(f"Training {config['model']} model")
    logger.info(f"optimizer = {config['optimizer']}")
    logger.info(f"learning rate = {config['learning_rate']}")
    logger.info(f"batch size = {config['batch_size']}")
    logger.info(f"loss function = {config['loss_func']}")

    # initialize wandb
    wandb.init(
        project="part-counting",
        entity="brunompac",
        config=config,
    )

    wandb.watch(net)

    for epoch in range(epochs):
        logger.info(f"Started epoch {epoch+1}/{epochs}")
        epoch_start_time = time()

        # train
        start_time = time()
        train_loss = train_pass(net, train_dataloader, criterion, optimizer,
                                scaler)
        end_time = time()

        logger.info(f"Training pass took {end_time - start_time:.3f} seconds")
        logger.info(f"Training loss = {train_loss}")

        wandb.log({
            "train_loss": train_loss,
        }, step=epoch)

        # validation
        start_time = time()
        val_loss, val_MAE = validation_pass(net, val_dataloader, criterion)
        end_time = time()

        logger.info(f"Validation pass took {end_time - start_time:.3f} seconds")
        logger.info(f"Validation loss = {val_loss}")
        logger.info(f"Validation MAE = {val_MAE}")

        wandb.log({
            "val_loss": val_loss,
            "val_MAE": val_MAE,
        }, step=epoch, commit=True)

        epoch_end_time = time()
        logger.info(
            f"Epoch {epoch+1} finished and took "
            f"{epoch_end_time - epoch_start_time:.2f} seconds"
        )

        # TODO: save model and checkpoints, see https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_models_for_inference.html and https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
        # TODO: implement continue training (sync wandb and torch)

    logger.info('Training finished!')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
