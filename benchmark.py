from timeit import default_timer as timer

import torch
from torch.nn.functional import conv2d

from layers import Conv2d


def main():

    # time the slow version
    repeats = 100

    data = torch.randn(64, 16, 28, 28)
    conv = Conv2d(16, 16, 3)
    start_slow_time = timer()
    for i in range(repeats):
        conv(data, slow=True)
    end_slow_time = timer()

    start_fast_time = timer()
    for i in range(repeats):
        conv(data)
    end_fast_time = timer()

    reshaped_weights = conv.weights.reshape(16, 16, 3, 3)
    bias = conv.bias
    start_torch_time = timer()
    for i in range(repeats):
        torch.relu(conv2d(data, reshaped_weights, bias))
    end_torch_time = timer()

    print("Performance Results")
    print("===================")
    print()
    print()
    print(
        f"Slow Convolutional Method: {(end_slow_time - start_slow_time) / repeats} s/it"
    )
    print(
        f"Faster Convolutional Method: {(end_fast_time - start_fast_time) / repeats} s/it"
    )
    print(
        f"Pytorch Convolutional Method: {(end_torch_time - start_torch_time) / repeats} s/it"
    )

if __name__ == "__main__":
    main()
