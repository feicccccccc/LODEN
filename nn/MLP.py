"""
Class for simple Multi Layer Perceptron
"""

import torch
import torch.nn as nn

from utils.activation import get_activation_function


class MLP(torch.nn.Module):
    def __init__(self, dims: list, activation='sigmoid', gain=1):
        """
        :param dims: Array defining the network structure.
        :param activation: Activation used in the network
        """
        super(MLP, self).__init__()
        layer_list = []
        for i in range(len(dims) - 1):
            cur_layer = nn.Linear(dims[i], dims[i + 1])
            torch.nn.init.orthogonal_(cur_layer.weight, gain=gain)
            layer_list.append(cur_layer)
            # TODO: Standard init, Can try different weight init

        self.layers = nn.ModuleList(layer_list)  # Register parameters
        self.activiation = get_activation_function(activation)

    def forward(self, x):
        h = x
        for layer_idx in range(len(self.layers) - 1):
            z = self.layers[layer_idx](h)
            h = self.activiation(z)
        z = self.layers[-1](h)
        return z


# For Debugging
if __name__ == "__main__":
    # For reproducibility
    torch.manual_seed(0)

    mlp = MLP([4, 3, 4, 1], activation='sigmoid')
    mlp.double()
    # Testing Tensor
    test_1 = torch.randint(-5, 5, (5, 4)).double()
    print(test_1)
    print(mlp(test_1))

    print("done")
