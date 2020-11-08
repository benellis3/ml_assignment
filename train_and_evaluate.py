import math
from random import sample

import torch
from tqdm import tqdm


def train(model, optimiser, data, targets, size=64, use_tqdm=False):

    it = minibatches(data, targets, size=size)
    if use_tqdm:
        it = tqdm(it)
    cum_loss = 0
    count = 0
    for batch, targets in it:
        optimiser.zero_grad()
        predictions = model(batch)

        loss = cross_entropy(predictions, targets)

        cum_loss += loss
        count += 1
        loss.backward()

        optimiser.step()
    # return average loss over the epoch
    return cum_loss / count


def test(model, data, targets):
    predictions = model(data)

    no_classes = torch.max(targets) + 1

    loss = cross_entropy(predictions, targets)

    _, predicted_classes = torch.max(predictions, dim=1)

    correct_targets = targets[predicted_classes == targets]
    correct = torch.sum(predicted_classes == targets)

    # compute the correct number per class
    accuracy_per_class = []
    for i in range(no_classes):
        accuracy_per_class.append(
            torch.sum(correct_targets == i) / torch.sum(targets == i)
        )

    return loss, correct / len(targets), accuracy_per_class


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
