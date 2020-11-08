import torch
from unittest.mock import Mock

from layers import FullyConnected
from model import Model, RMSProp


def test_optimiser_step():
    layers = [FullyConnected(4, 2)]
    model = Model(layers)
    model.layers[0].weights = torch.tensor(
        [[9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]], requires_grad=True
    )
    optim = RMSProp(model.parameters(), lr=2)

    data = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

    out = model(data)
    total = torch.sum(out)
    total.backward()
    optim.step()

    expected = torch.tensor(
        [[2.6754, 3.6754], [4.6754, 5.6754], [6.6754, 7.6754], [8.6754, 9.6754]]
    )

    assert torch.allclose(expected, model.layers[0].weights, atol=1e-4)


def test_zero_grad():
    data = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    optim = RMSProp([data], 1)

    loss = torch.sum(data * 3)
    loss.backward()
    optim.step()
    optim.zero_grad()

    assert torch.equal(torch.zeros(2, 2, dtype=torch.float32), optim.parameters[0].grad)


def test_layers():
    """Tests that the model calls the layers in order"""
    int_value = Mock()
    mock_layer_1 = Mock(return_value=int_value)
    mock_layer_2 = Mock()
    mock_input = Mock()
    layers = [mock_layer_1, mock_layer_2]
    model = Model(layers)
    model(mock_input)
    mock_layer_1.assert_called_once()
    mock_layer_2.assert_called_once()
    mock_layer_2.assert_called_with(int_value)
