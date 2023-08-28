# -*- coding: utf-8 -*-

"""Train a neural network with single-sampling training"""

import argparse
import glob
import logging
import os
import time

import numpy as np
import torch

from sucpy.dual_initialisers import SingleSamplingNetworkInitialiser
from sucpy.utils import logging_utils, network_utils


def main():
    """Run the main routine of this script"""
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="path to an experiment root directry",
    )
    parser.add_argument(
        "--smoketest",
        action="store_true",
        help="run smoketest",
    )
    args = parser.parse_args()
    path = args.path

    model_path = f"{path}/out/single_sampling_network.pth"
    ratio_training_data = 0.8

    log_filepath = f"{path}/out/logs/single_sampling_network.txt"

    if os.path.exists(model_path):
        print(f"{model_path=} exists")
        return

    logging_utils.initialise(log_filepath)
    logger = logging.getLogger(__name__)

    data_filepath = glob.glob(f"{path}/out/nearest_neighbour_data/*.npz")[0]
    full_data = np.load(data_filepath)

    logger.info(f"full data size: {full_data['demand'].shape[0]}")

    training_budget = 24 * 60 * 60

    mask = full_data["instance_solution_walltime"] < training_budget

    x = full_data["demand"][mask]
    y = full_data["y"][mask]

    np.testing.assert_allclose(x.ndim, 2)
    np.testing.assert_allclose(y.ndim, 2)

    training_data_size = int(x.shape[0] * ratio_training_data)
    validation_data_size = x.shape[0] - training_data_size

    training_data_selector = np.sort(
        np.random.choice(x.shape[0], training_data_size, replace=False)
    )
    validation_data_selector = np.setdiff1d(
        np.arange(x.shape[0]), training_data_selector
    )

    np.testing.assert_allclose(
        validation_data_selector.size, validation_data_size
    )

    training_x = x[training_data_selector]
    training_y = y[training_data_selector]
    validation_x = x[validation_data_selector]
    validation_y = y[validation_data_selector]

    logger.info(f"used data size: {x.shape[0]}")

    dual_initialiser = SingleSamplingNetworkInitialiser(
        dim_parameter=x.shape[1],
        dim_dual=y.shape[1],
        network_kwargs=dict(activation="tanh"),
    )
    dual_initialiser.set_normalizer(
        parameter=training_x,
        dual=training_y,
    )

    optimizer = torch.optim.Adam(
        dual_initialiser.parameters(),
        lr=1e-3,
    )

    loss = torch.nn.MSELoss()

    train_neural_network(
        dual_initialiser,
        optimizer,
        loss,
        training_x,
        training_y,
        validation_x,
        validation_y,
    )

    logger.info(f"saving model to: {model_path}")
    torch.save(dual_initialiser.state_dict(), model_path)


def train_neural_network(
    model,
    optimizer,
    loss,
    training_x,
    training_y,
    validation_x,
    validation_y,
    half_lr_after=4,
    terminate_training_after=12,
    mini_batch_config=None,
):
    """Train a neural network model

    Parameters
    ----------
    model : torch.nn.Module
    optimizer : torch.optim.optimizer.Optimizer
    training_x : (training_data_size, dim_x) array
    training_y : (training_data_size, dim_y) array
    validation_x : (validation_data_size, dim_x) array
    validation_y : (validation_data_size, dim_y) array
    half_lr_after : int, default 4
        Halve the learning rate after this number of epochs without
        improvement.
    terminate_training_after : int, default 12
        Terminate the training after this number of epochs without
        improvement.
    mini_batch_config : dict, optional
        This may has keyword arguments passed to network_utils.get_batches.
        By default batch_size is 64.
    """
    torch.set_num_threads(1)
    logger = logging.getLogger(__name__)

    if not isinstance(training_x, torch.Tensor):
        training_x = torch.Tensor(training_x)
    if not isinstance(training_y, torch.Tensor):
        training_y = torch.Tensor(training_y)
    if not isinstance(validation_x, torch.Tensor):
        validation_x = torch.Tensor(validation_x)
    if not isinstance(validation_y, torch.Tensor):
        validation_y = torch.Tensor(validation_y)

    if mini_batch_config is None:
        mini_batch_config = {}

    mini_batch_default_config = {"batch_size": 64}
    # Populate the default config.
    mini_batch_config = {**mini_batch_default_config, **mini_batch_config}

    losses = []
    best_loss = np.inf
    n_no_progress = 0

    starttime = time.perf_counter()

    for epoch_index in range(1000):
        model.train()
        minibatches = network_utils.get_batches(
            rng=epoch_index,
            x=training_x,
            y=training_y,
            **mini_batch_config,
        )
        for batch in minibatches:
            input = batch.x
            target = batch.y
            prediction = model(input)
            epoch_loss = loss(prediction, target)
            optimizer.zero_grad()
            epoch_loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            input = validation_x
            target = validation_y
            prediction = model(input)
            validation_loss = loss(prediction, target)
            losses.append(validation_loss.detach())
            if best_loss > validation_loss:
                n_no_progress = 0
                best_loss = validation_loss
                best_loss_updated = True
            else:
                n_no_progress += 1
                best_loss_updated = False
        elapse = time.perf_counter() - starttime
        logger.info(
            f"{elapse:6.1f}  "
            f"{losses[-1]:7.4f}{'*' if best_loss_updated else ' '}  "
            f"{network_utils.get_lr(optimizer):.1e}"
        )

        if n_no_progress > terminate_training_after:
            break
        elif (n_no_progress + 1) % half_lr_after == 0:
            network_utils.multiply_lr(optimizer, 0.5)


if __name__ == "__main__":
    main()
