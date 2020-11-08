import argparse
import math
from random import sample

import torch
from tqdm import tqdm

from layers import FullyConnected
from load_data import Dataset, load_iris_data
from model import Model, RMSProp


def train(model, optimiser, data, targets, size=64, use_tqdm=False):

    it = minibatches(data, targets, size=size)
    if use_tqdm:
        it = tqdm(it)
    for batch, targets in it:
        optimiser.zero_grad()
        predictions = model(batch)

        loss = cross_entropy(predictions, targets)

        loss.backward()

        optimiser.step()


def test(model, data, targets):
    predictions = model(data)

    loss = cross_entropy(predictions, targets)

    _, predicted_classes = torch.max(predictions, dim=1)

    correct = torch.sum(predicted_classes == targets)

    return loss, correct / len(targets)


def cross_entropy(predictions, targets, dim=1):
    # convert targets to one hot

    one_hot_targets = torch.zeros(*targets.shape, predictions.size(1))
    one_hot_targets.scatter_(dim=1, index=targets[:, None], value=1)

    # compute the log softmax, subtracting max for numerical
    # stability. Heavily inspired by the numpy implementation.
    pred_max, _ = torch.max(predictions, dim=dim)
    preds = predictions - pred_max[:, None]
    exp_preds = torch.exp(preds)
    pred_sum = torch.log(torch.sum(exp_preds, dim=dim))
    logsoftmax = preds - pred_sum[:, None]

    return torch.sum(-one_hot_targets * logsoftmax)


def minibatches(data, targets, size=64):
    """Randomly generates minibatches of the given
    size"""
    N = data.size(0)

    no_batches = math.ceil(N / size)
    indices = set(range(N))
    for i in range(no_batches):
        no_indices = size if N - i * size > size else N - i * size
        # randomly draw and remove these indices
        drawn_indices = sample(indices, no_indices)
        indices -= set(drawn_indices)
        yield data[drawn_indices], targets[drawn_indices]
