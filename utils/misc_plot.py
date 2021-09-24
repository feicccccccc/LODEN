"""
Model Based Reinforcement Learning with Analytic Mechanic
Fei, Cheung

Function for plotting graph for Mass network, and Potential network
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator

import torch
import torch.nn as nn

import numpy as np


def plot_2D_massMatrixNet(model: nn.Module):

    n_grid = 50
    x = torch.linspace(-np.pi, np.pi, n_grid)
    Q1, Q2 = torch.meshgrid(x, x)  # all (q, dq)

    M = torch.zeros([*Q1.shape, 4])

    for i in range(n_grid):
        for j in range(n_grid):
            # The current canonical coordinate
            x = torch.cat([Q1[i, j].reshape(1, 1), Q1[i, j].reshape(1, 1)], 1)
            M[i, j, :] = model.defunc.m.L(x).detach().cpu()

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    CS = ax.contourf(Q1, Q2, M, 100, cmap='RdYlBu')
    fig.colorbar(CS)

    ax.set_xlim([Q1.min(), Q1.max()])
    ax.set_ylim([Q2.min(), Q2.max()])
    ax.set_xlabel("q")
    ax.set_ylabel("dq")
    ax.set_title("Mass Matrix output")
    fig.show()


def plot_LNN_1D_surface(model, ground_truth=None):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    n_grid = 50
    x = torch.linspace(-np.pi, np.pi, n_grid)
    Q, dQ = torch.meshgrid(x, x)  # all (q, dQ)

    L = torch.zeros(Q.shape)

    for i in range(n_grid):
        for j in range(n_grid):
            x = torch.cat([Q[i, j].reshape(1, 1), dQ[i, j].reshape(1, 1)], 1)
            L[i, j] = model.defunc.m.L(x).detach()

    scale = 1
    offset = 0
    if ground_truth is not None:
        # will not change if multiplied by some constant or add to a time derivative of f(q, t)
        L_true = ground_truth(Q, dQ)
        offset = (L_true[int(n_grid / 2), int(n_grid / 2)] - L[int(n_grid / 2), int(n_grid / 2)]).numpy()
        L += offset
        scale = (L_true.max() / L.max()).numpy()
        L *= scale
        _ = ax.plot_surface(Q.numpy(), dQ.numpy(), L_true.numpy(),
                            cmap='coolwarm', alpha=0.2)

    # Plot the surface
    surf = ax.plot_surface(Q.numpy(), dQ.numpy(), L.numpy(),
                           cmap='coolwarm')

    # Customize the z axis.
    diff = L.max() - L.min()
    ax.set_zlim(L.min() - diff * 0.1, L.max() + diff * 0.1)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')
    ax.view_init(30, 30)

    ax.set_title("Lagrangian Surface plot")

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    fig.show()