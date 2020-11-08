import math

import torch

NONLINEARITY_MAP = {
    "relu": (torch.relu, math.sqrt(2)),
    "linear": (lambda x: x, 1),
    "sigmoid": (torch.sigmoid, 1),
}


def _fan_in_fan_out(tensor):
    """
    Returns
    -------
    fan_in: The maximum number of inputs this can accept
    fan_out: The max number of outputs it can produce
    """
    size = 1
    if tensor.dim() > 2:
        size = tensor[0][0].numel()
    if tensor.dim() >= 2:
        return tensor.size(1) * size, tensor.size(0) * size
    else:
        return tensor.size(0) * size, tensor.size(0) * size


class Layer:
    def _init_weights(self, weights, gain, mode="fan_in"):
        fan_in, fan_out = _fan_in_fan_out(weights)
        if mode == "fan_in":
            fan = fan_in
        elif mode == "fan_out":
            fan = fan_out
        else:
            raise NotImplementedError(
                "Only fan_in and fan_out are valid modes of weight initialisation"
            )
        var = gain / math.sqrt(fan)
        eps = torch.randn_like(weights)
        weights = eps * var
        return weights


class Layer2d(Layer):
    def _slow_im_batch_to_column(self, x, stride=1, flatten_channels=True):

        rows = []
        for row in range(0, x.size(-1) - self.kernel_size + 1, stride):
            for col in range(0, x.size(-2) - self.kernel_size + 1, stride):
                window = x[
                    :, :, row : row + self.kernel_size, col : col + self.kernel_size
                ]
                if flatten_channels:
                    rows.append(window.reshape(window.size(0), -1))
                else:
                    rows.append(window.reshape(window.size(0), window.size(1), -1))

        dim = 1 if flatten_channels else 2
        return torch.stack(rows, dim=dim)

    def _fast_im_batch_to_column(self, x, stride=1, flatten_channels=True):
        """Transforms the image batch x from shape (N, C_in, H, W) to
        (N, (H - K + 1) * (W - K + 1), C_in * K**2) where K is the kernel size.
        This allows convolution to be done by matrix multiplication"""
        # unfold the rows of x. Shapes assume stride == 1.
        # shape N, C_in, H - K + 1, W, K
        rows = x.unfold(-2, self.kernel_size, stride)
        # shape N, C_in, H - K + 1, W - K + 1, K, K
        patches = rows.unfold(-2, self.kernel_size, stride)
        # reshape to flatten the right dimensions
        if flatten_channels:
            patches = patches.permute(0, 2, 3, 1, 4, 5)
            patches = patches.reshape(
                patches.size(0), patches.size(1), patches.size(2), -1
            )
            patches = patches.reshape(patches.size(0), -1, patches.size(-1))
        else:
            patches = patches.reshape(
                patches.size(0), patches.size(1), patches.size(2), patches.size(3), -1
            )
            patches = patches.reshape(
                patches.size(0), patches.size(1), -1, patches.size(-1)
            )

        return patches

    def _im_batch_to_column(self, x, stride=1, flatten_channels=True, slow=False):
        if slow:
            return self._slow_im_batch_to_column(
                x, stride=stride, flatten_channels=flatten_channels
            )
        else:
            return self._fast_im_batch_to_column(
                x, stride=stride, flatten_channels=flatten_channels
            )


class FullyConnected(Layer):
    def __init__(self, *args, nonlinearity="relu"):
        try:
            self.nonlinearity, gain = NONLINEARITY_MAP[nonlinearity]
        except KeyError:
            raise Exception(f"Not a recognised nonlinearity {nonlinearity}")
        self.weights = self._init_weights(torch.zeros(*args), gain)
        self.weights.requires_grad_()
        self.bias = self._init_weights(
            torch.zeros(*args[-(self.weights.dim() - 1) :]), gain
        )
        self.bias.requires_grad_()

    def __call__(self, x):
        return self.nonlinearity(x @ self.weights + self.bias)


class Conv2d(Layer2d):
    """A 2D convolution layer that makes a passable attempt at being
    performant by rearranging the image receptive fields to column format.
    It does not modify the strides of anything to improve performance further though
    and so is likely to not scale to v. large datasets."""

    def __init__(self, in_channels, out_channels, kernel_size, nonlinearity="relu"):
        try:
            self.nonlinearity, gain = NONLINEARITY_MAP[nonlinearity]
        except KeyError:
            raise Exception(f"Not a recognised nonlinearity {nonlinearity}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weights = self._init_weights(
            torch.zeros(in_channels, kernel_size, kernel_size, out_channels), gain
        )
        self.weights.requires_grad_()
        self.bias = self._init_weights(torch.zeros(out_channels), gain)
        self.bias.requires_grad_()

    def __call__(self, x, slow=False):
        # shape (N, (H - K + 1)*(W - K + 1), C_in * K**2)
        col_x = self._im_batch_to_column(x, slow=slow)

        # shape (K ** 2 * C_in, C_out)
        weights_tmp = self.weights.view(
            self.kernel_size ** 2 * self.in_channels, self.out_channels
        )

        # out shape (N, (H - K + 1) * (W - K + 1), C_out)
        out = col_x @ weights_tmp + self.bias
        out = out.transpose(2, 1)

        out_size = x.size(-1) - self.kernel_size + 1
        return self.nonlinearity(
            out.reshape(out.size(0), self.out_channels, out_size, out_size)
        )


class MaxPool2d(Layer2d):
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def __call__(self, x, slow=False):
        col_x = self._im_batch_to_column(
            x, stride=self.kernel_size, flatten_channels=False, slow=slow
        )
        # take the maximum over the last dimension
        out, _ = torch.max(col_x, dim=-1)
        # flatten out to be the output size again
        out_size = x.size(-1) // self.kernel_size
        return out.reshape(out.size(0), out.size(1), out_size, out_size)


class Flatten(Layer):
    def __init__(self, start_dim=0, end_dim=-1):
        self.start_dim = start_dim
        self.end_dim = end_dim

    def __call__(self, x):
        return torch.flatten(x, start_dim=self.start_dim, end_dim=self.end_dim)
