from abc import ABC, abstractmethod

import torch


class Model:
    """Defines a neural network model"""

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def parameters(self):
        params = [getattr(layer, "weights", None) for layer in self.layers]
        bias_params = [getattr(layer, "bias", None) for layer in self.layers]
        params.extend(bias_params)
        return [param for param in params if param is not None]


class Optimiser(ABC):
    @abstractmethod
    def step(self):
        pass


class RMSProp(Optimiser):
    def __init__(self, parameters, lr, avg_weight=0.9, eps=1e-8):
        self.parameters = parameters
        self.hyperparams = {
            "lr": lr,
            "avg_weight": avg_weight,
            "eps": eps,
        }
        self.moving_average = [
            torch.zeros_like(param).detach() for param in self.parameters
        ]

    def zero_grad(self):
        for parameter in self.parameters:
            if parameter.grad is not None:
                parameter.grad.zero_()

    def step(self):
        with torch.no_grad():
            for i, parameter in enumerate(self.parameters):
                moving_average = self.moving_average[i]
                avg_weight = self.hyperparams["avg_weight"]
                eps = self.hyperparams["eps"]
                grad = parameter.grad
                moving_average.mul_(avg_weight).addcmul_(
                    grad, grad, value=1 - avg_weight
                )

                parameter.addcdiv_(
                    parameter.grad,
                    moving_average.sqrt() + eps,
                    value=-self.hyperparams["lr"],
                )
