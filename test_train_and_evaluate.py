import torch
import torch.nn.functional as F

from train_and_evaluate import cross_entropy


def test_cross_entropy():

    targets = torch.tensor([0, 1, 2])
    predictions = torch.tensor([[10.0, 1.0, -4.0], [3.0, 4.0, 5.0], [7.0, 10.0, 15.0]])
    loss = cross_entropy(predictions, targets)

    torch_loss = F.cross_entropy(predictions, targets, reduction="sum")

    assert torch.allclose(loss, torch_loss)
