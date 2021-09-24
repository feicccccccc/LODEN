"""
Class for Making Control Matrix
"""

import torch
import torch.nn as nn

from utils.activation import get_activation_function


class MatrixNet(torch.nn.Module):
    def __init__(self, net_dims: list, control_dim: int, activation='sigmoid', gain=1):
        """
        :param net_dims: Array defining the network structure. Input should be state dim,
                         Output should be dq dim.
        :param activation: Activation used in the network
        """
        super(MatrixNet, self).__init__()
        layer_list = []
        self._q_dim = net_dims[0]
        self._dq_dim = net_dims[-1]
        self._control_dim = control_dim

        for i in range(len(net_dims) - 2):
            cur_layer = nn.Linear(net_dims[i], net_dims[i + 1])
            torch.nn.init.orthogonal_(cur_layer.weight, gain=gain)
            layer_list.append(cur_layer)

        last_layer = nn.Linear(net_dims[-2], self._dq_dim * self._control_dim)
        layer_list.append(last_layer)

        self.layers = nn.ModuleList(layer_list)  # Register parameters
        self.activiation = get_activation_function(activation)

    def forward(self, x):
        bs, dim = x.shape
        h = x
        for layer_idx in range(len(self.layers) - 1):
            z = self.layers[layer_idx](h)
            h = self.activiation(z)

        z = self.layers[-1](h)
        G = torch.reshape(z, (bs, self._dq_dim, self._control_dim))
        return G


# For Debugging
if __name__ == "__main__":
    # For reproducibility
    torch.manual_seed(0)

    test = MatrixNet([8, 3, 7, 4], control_dim=1, activation='sigmoid')

    # Testing Tensor
    test_1 = torch.randint(-5, 5, (5, 8)).float()
    test_c = torch.randint(-5, 5, (5, 2)).float()

    print(test_1.shape)
    print(test(test_1).shape)
    print(test_c.shape)
    print(torch.einsum('ijk,ik->ij', test(test_1), test_c).shape)
    print("done")

