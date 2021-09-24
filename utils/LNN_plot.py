"""
Function for plotting graph for LNN
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator

import torch
import torch.nn as nn

import numpy as np


def plot_LNN_1D_traj(model: nn.Module, sampling_points=None):
    """
    Use random point to show trajectory
    :param sampling_points: Points we want to sample the trajectory
    :param model: Our Model of the ODE
    :return: None
    """
    time_span = 1
    steps = 100
    if sampling_points is None:
        points = torch.randn((100, 2))
    else:
        points = sampling_points

    # point dim: (bs, (q, dq))

    t_span = torch.linspace(0, time_span, steps)
    traj = model.trajectory(points, t_span).detach()  # (time, points, dim)

    # Plot the trajectory generated from the model
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)

    for point in range(len(points)):
        ax.plot(traj[:, point, 0], traj[:, point, 1], color='blue')  # traj
        ax.plot(traj[0, point, 0], traj[0, point, 1], marker='x', color="red", alpha=0.5)  # init pos

    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([-2.5, 2.5])
    ax.set_xlabel("q")
    ax.set_ylabel("p")
    ax.set_title("Trajectories of model dynamic")
    fig.show()


def plot_LNN_1D_vector_field(model: nn.Module):
    # Create grid to plot the function
    n_grid = 50
    x = torch.linspace(-np.pi, np.pi, n_grid)
    Q, dQ = torch.meshgrid(x, x)  # all (q, dq)

    L, U, V = torch.zeros(Q.shape), torch.zeros(Q.shape), torch.zeros(dQ.shape)

    for i in range(n_grid):
        for j in range(n_grid):
            # The current canonical coordinate
            x = torch.cat([Q[i, j].reshape(1, 1), dQ[i, j].reshape(1, 1)], 1)
            L[i, j] = model.defunc.m.L(x).detach().cpu()
            grad = model.defunc(0, x).detach().cpu()
            U[i, j], V[i, j] = grad[:, 0], grad[:, 1]

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    CS = ax.contourf(Q, dQ, L, 100, cmap='RdYlBu')
    fig.colorbar(CS)

    ax.streamplot(Q.T.numpy(), dQ.T.numpy(), U.T.numpy(), V.T.numpy(), color='black')

    ax.set_xlim([Q.min(), Q.max()])
    ax.set_ylim([dQ.min(), dQ.max()])
    ax.set_xlabel("q")
    ax.set_ylabel("dq")
    ax.set_title("lagrangian & Vector Field")
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