import argparse

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
from tqdm import tqdm

from layers import Conv2d, Flatten, FullyConnected, MaxPool2d
from load_data import Dataset, load_iris_data, load_kmnist_data
from model import Model, RMSProp
from train_and_evaluate import test, train


def make_image_grid(dataset, labels, savefig=False):
    # find 20 examples of each class
    classes = labels.max()
    images = []
    fig, ax = plt.subplots(1, 1)
    for c in range(classes + 1):
        images.append(dataset[labels == c][:20])
    images = torch.stack(images)
    images = images.flatten(start_dim=0, end_dim=1)
    images = images.unsqueeze(dim=1)
    grid = make_grid(images, nrow=20)
    numpy_grid = grid.numpy()
    ax.imshow(numpy_grid.transpose((1, 2, 0)), interpolation="nearest")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if savefig:
        plt.savefig("figures/image_grid.pdf")


def plot_loss(losses, dataset=None, savefig=False):
    fig, ax = plt.subplots(1, 1)

    ax.plot(range(len(losses)), losses)
    ax.set_ylabel("Mean training loss")
    ax.set_xlabel("Epoch #")
    if savefig:
        plt.savefig(f"figures/{dataset}_training_loss.pdf")


def plot_bar(class_accuracy, classes, dataset=None, savefig=False):
    fig, ax = plt.subplots(1, 1)

    ax.bar(range(len(classes)), class_accuracy, tick_label=classes)
    ax.set_xlabel("Classes")
    ax.set_ylabel("Accuracy")
    if savefig:
        plt.savefig(f"figures/{dataset}_classification_accuracy.pdf")


def main(args):

    parser = argparse.ArgumentParser(
        description="Trains a simple neural net on simple data"
    )

    parser.add_argument(
        "--dataset", type=Dataset, choices=Dataset, default=Dataset.IRIS
    )
    parser.add_argument("--plot-image-grid", action="store_true")
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not show any of the generated plots",
    )

    parser.add_argument(
        "--savefig", action="store_true", help="Save the figures as pdfs"
    )

    parsed_args = parser.parse_args()
    epochs = parsed_args.epochs

    if parsed_args.dataset == Dataset.IRIS:
        train_data, test_data, train_targets, test_targets, classes = load_iris_data()
    elif parsed_args.dataset == Dataset.KMNIST:
        train_data, test_data, train_targets, test_targets, classes = load_kmnist_data()
        if parsed_args.plot_image_grid:
            make_image_grid(train_data, train_targets, savefig=parsed_args.savefig)
            plt.show()
            return

    features = train_data.size(1)

    no_classes = train_targets.max() + 1

    if parsed_args.dataset == Dataset.IRIS:
        layers = [
            FullyConnected(features, 32),
            FullyConnected(32, no_classes, nonlinearity="linear"),
        ]
    elif parsed_args.dataset == Dataset.KMNIST:
        layers = [
            Conv2d(1, 8, 3),
            Conv2d(8, 8, 3),
            MaxPool2d(2),
            Conv2d(8, 16, 3),
            Conv2d(16, 16, 3),
            MaxPool2d(2),
            Flatten(start_dim=1, end_dim=-1),
            FullyConnected(256, no_classes, nonlinearity="linear"),
        ]
    model = Model(layers)
    optimiser = RMSProp(model.parameters(), parsed_args.lr)

    it = range(epochs)
    if parsed_args.dataset != Dataset.KMNIST:
        it = tqdm(it)

    epochs_loss = []
    for epoch in it:
        epochs_loss.append(
            train(model, optimiser, train_data, train_targets, size=128, use_tqdm=True)
        )

    plot_loss(epochs_loss, dataset=parsed_args.dataset, savefig=parsed_args.savefig)

    loss, accuracy, accuracy_per_class = test(model, test_data, test_targets)
    plot_bar(
        accuracy_per_class,
        classes,
        dataset=parsed_args.dataset,
        savefig=parsed_args.savefig,
    )
    if not parsed_args.quiet:
        plt.show()
    print(f"Final Loss: {loss}, Final Accuracy: {accuracy}")


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
