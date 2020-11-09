import pytest
import torch
from torch import nn as N
from torch.nn.functional import conv2d

from layers import Conv2d, FullyConnected, MaxPool2d


def test_initialisation():
    torch.manual_seed(100)

    layer = FullyConnected(4, 4)
    expected = torch.tensor(
        [
            [0.0897, 0.9591, 0.3983, -0.0735],
            [-0.2528, 0.2770, -0.4809, 0.1704],
            [0.3322, 0.8787, 0.3821, -0.8099],
            [-1.0318, -1.1512, 0.2711, -0.1215],
        ]
    )
    assert torch.allclose(layer.weights, expected, atol=1e-4)


def test_fully_connected():
    data = torch.tensor([1.0, 2.0, 3.0, 4.0])

    layer = FullyConnected(4, 2)
    layer.weights = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [-1.0, -40.0],
        ]
    )
    layer.bias = torch.tensor([1.0, 2.0])

    out = layer(data[None, :])
    expected = torch.tensor([[19.0, 0.0]])
    assert torch.allclose(out, expected)


@pytest.mark.parametrize("slow", [False, True])
def test_conv2d(slow):
    # 3 x 1 x 2 x 2
    data = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[-1.0, -2.0], [-3.0, -4.0]],
        ]
    )
    data = data.unsqueeze(dim=1)
    # perform 1x1 convolutions on data
    layer = Conv2d(1, 3, 1)
    layer.weights = torch.tensor([[[[10.0, 20.0, 30.0]]]])
    layer.bias = torch.tensor([1.0, 2.0, 3.0])
    out = layer(data, slow=slow)

    torch_out = conv2d(data, layer.weights.view(3, 1, 1, 1), layer.bias)
    torch_out = torch.relu(torch_out)
    assert torch.allclose(out, torch_out)


def test_maxpool2d():

    data = torch.randn(3, 2, 6, 6)
    layer = MaxPool2d(3)
    out = layer(data)

    torch_layer = N.MaxPool2d(3)
    torch_out = torch_layer(data)

    assert torch.allclose(out, torch_out)
