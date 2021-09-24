"""
Model Based Reinforcement Learning with Analytic Mechanic
Fei, Cheung

Function for picking non-linearity
"""

import torch
import torch.nn as nn


def get_activation_function(activation: str, arg=None):
    if activation == 'tanh':
        non_linear_activation = nn.Tanh()
    elif activation == 'relu':
        non_linear_activation = nn.ReLU()
    elif activation == 'sigmoid':
        non_linear_activation = nn.Sigmoid()
    elif activation == 'softmax':
        non_linear_activation = nn.Softmax(dim=arg)  # normalisation across dim
    elif activation == 'softplus':
        non_linear_activation = nn.Softplus()
    elif activation == 'logsigmoid':
        non_linear_activation = nn.LogSigmoid()
    elif activation == 'gelu':
        non_linear_activation = nn.GELU()
    else:
        raise ValueError("Activation not recognized.")
    return non_linear_activation


# For Debugging
if __name__ == "__main__":
    # For reproducibility
    torch.manual_seed(0)

    # Testing Tensor
    test_1 = torch.randint(-5, 5, (5, 2)).float()
    test_activation = get_activation_function('softmax', 1)
    test_out_1 = test_activation(test_1)

    print(test_1)
    print(test_out_1)
    print("done")
