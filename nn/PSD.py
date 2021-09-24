"""
Class for generate a PSD matrix using NN
Generate through Cholesky decomposition
M = L L.T (assume real value matrix)
"""

import torch
import torch.nn as nn
import numpy as np

from utils.activation import get_activation_function


class PSD(nn.Module):

    def __init__(self, dims: list, activation='sigmoid', gain=1):
        """
        :param dims: Array defining the network structure. Last dimension specify the diag dimension.
        :param activation: Activation used in the network
        """
        super(PSD, self).__init__()
        layer_list = []

        self._diag_dim = dims[-1]
        self._off_diag_dim = (self._diag_dim ** 2 - self._diag_dim) // 2
        self.eps = 0.01  # for stabilise the Cholesky decomposition

        for i in range(len(dims) - 1):  # ignore the last layer
            cur_layer = nn.Linear(dims[i], dims[i + 1])
            torch.nn.init.orthogonal_(cur_layer.weight, gain=gain)
            layer_list.append(cur_layer)

        last_layer = nn.Linear(dims[-1], self._diag_dim + self._off_diag_dim)
        torch.nn.init.orthogonal_(last_layer.weight)
        layer_list.append(last_layer)

        self.layers = nn.ModuleList(layer_list)
        self.activiation = get_activation_function(activation)

    def forward(self, x):
        """
        Output should always have dimension (bs, row, col) for torch.bmm batch matrix multiplication
        """
        bs = x.shape[0]

        h = x
        for layer_idx in range(len(self.layers) - 1):
            z = self.layers[layer_idx](h)
            h = self.activiation(z)
        z = self.layers[-1](h)

        diag, off_diag = torch.split(z, [self._diag_dim, self._off_diag_dim], dim=1)

        # Generate PSD using Cholesky decomposition
        L = torch.diag_embed(diag)

        # k = -1 to offset from main diag
        # indices of lower tri for a flatten matrix
        ind = np.tril_indices(self._diag_dim, k=-1)

        # get the flatten index from the target square matrix
        flat_ind = np.ravel_multi_index(ind, (self._diag_dim, self._diag_dim))

        L = torch.flatten(L, start_dim=1)
        L[:, flat_ind] = off_diag
        L = torch.reshape(L, (bs, self._diag_dim, self._diag_dim))

        # batch matrix product, L @ L.T Cholesky decomposition
        D = torch.bmm(L, L.transpose(2, 1))
        D = D + torch.eye(self._diag_dim) * self.eps  # for stability
        return D


# For Debugging
if __name__ == "__main__":
    # For reproducibility
    torch.manual_seed(0)

    psd = PSD([4, 3, 4, 5], activation='sigmoid').double()

    # Testing Tensor
    test_1 = torch.randint(-5, 5, (5, 4)).double()
    print(test_1.shape)
    print(psd(test_1).shape)

    print("done")
