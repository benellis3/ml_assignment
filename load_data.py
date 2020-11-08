from enum import Enum

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch import float32, from_numpy, zeros
from torchvision.datasets import KMNIST
from torchvision.transforms import (Compose, ConvertImageDtype, Normalize,
                                    ToTensor)


class Dataset(Enum):
    IRIS = "iris"
    KMNIST = "kmnist"


def load_iris_data(split=0.2):
    dataset = load_iris()
    train_data, test_data, train_targets, test_targets = train_test_split(
        dataset["data"], dataset["target"], test_size=split
    )
    train_data = from_numpy(train_data).float()
    train_targets = from_numpy(train_targets)
    test_data = from_numpy(test_data).float()
    test_targets = from_numpy(test_targets)

    return train_data, test_data, train_targets, test_targets


def load_kmnist_data():
    transforms = Compose(
        [
            ConvertImageDtype(float32),
            Normalize(zeros(28, 28) + 0.5, zeros(28, 28) + 0.5),
        ]
    )
    train_set = KMNIST(root="./data/", download=True, train=True, transform=transforms)
    test_set = KMNIST(root="./data/", train=False, download=True, transform=transforms)

    train_data = transforms(train_set.data)
    test_data = transforms(test_set.data)

    return (
        train_data.unsqueeze(1),
        test_data.unsqueeze(1),
        train_set.targets,
        test_set.targets,
    )
