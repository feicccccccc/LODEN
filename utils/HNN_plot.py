"""
Function for plotting graph for HNN
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator

import torch
import torch.nn as nn

import numpy as np


def plot_HNN_1D_traj(model: nn.Module, sampling_points=None):
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


def plot_HNN_1D_vector_field(model: nn.Module):
    # Create grid to plot the function
    n_grid = 50
    x = torch.linspace(-np.pi, np.pi, n_grid)
    Q, P = torch.meshgrid(x, x)  # all (q, p)

    # H is Hamiltonian
    # (U, V) are the symplectic gradient
    H, U, V = torch.zeros(Q.shape), torch.zeros(Q.shape), torch.zeros(Q.shape)

    for i in range(n_grid):
        for j in range(n_grid):
            # The current canonical coordinate
            x = torch.cat([Q[i, j].reshape(1, 1), P[i, j].reshape(1, 1)], 1)
            H[i, j] = model.defunc.m.H(x).detach().cpu()
            grad = model.defunc(0, x).detach().cpu()
            U[i, j], V[i, j] = grad[:, 0], grad[:, 1]

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    CS = ax.contourf(Q, P, H, 100, cmap='RdYlBu')
    fig.colorbar(CS)

    ax.streamplot(Q.T.numpy(), P.T.numpy(), U.T.numpy(), V.T.numpy(), color='black')

    ax.set_xlim([Q.min(), Q.max()])
    ax.set_ylim([P.min(), P.max()])
    ax.set_xlabel("q")
    ax.set_ylabel("p")
    ax.set_title("Hamiltonian & Vector Field")
    fig.show()


def plot_HNN_1D_surface(model, ground_truth=None):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    n_grid = 50
    x = torch.linspace(-np.pi, np.pi, n_grid)
    Q, P = torch.meshgrid(x, x)  # all (q, p)

    H = torch.zeros(Q.shape)

    for i in range(n_grid):
        for j in range(n_grid):
            # The current canonical coordinate
            x = torch.cat([Q[i, j].reshape(1, 1), P[i, j].reshape(1, 1)], 1)
            H[i, j] = model.defunc.m.H(x).detach()

    # Plot the surface
    surf = ax.plot_surface(Q.numpy(), P.numpy(), H.numpy(),
                           cmap='coolwarm')

    if ground_truth is not None:
        H_true = ground_truth(Q, P) + H[int(n_grid / 2), int(n_grid / 2)]  # offset to see to compare
        _ = ax.plot_surface(Q.numpy(), P.numpy(), H_true.numpy(),
                            cmap='coolwarm', alpha=0.2)

    # Customize the z axis.
    ax.set_zlim(H.min() - 0.5, H.max() + 0.5)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')
    ax.view_init(30, 30)

    ax.set_title("Hamiltonian Surface plot")

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    fig.show()


# TODO: Move when restructured folder
def plot_2D_vector_field(model: nn.Module):
    # plot (time/depth independent) vector field
    n_grid = 50
    x = torch.linspace(-2.5, 2.5, n_grid)
    X, Y = torch.meshgrid(x, x)
    z = torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1)], 1)

    f = model.defunc(0, z).cpu().detach()  # output the gradient
    fx, fy = f[:, 0], f[:, 1]
    fx, fy = fx.reshape(n_grid, n_grid), fy.reshape(n_grid, n_grid)

    # plot vector field and its intensity
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)

    ax.streamplot(X.numpy().T, Y.numpy().T, fx.numpy().T, fy.numpy().T, color='black')

    CS = ax.contourf(X.T, Y.T, torch.sqrt(fx.T ** 2 + fy.T ** 2), cmap='RdYlBu')
    fig.colorbar(CS)

    ax.set_xlabel("q")
    ax.set_ylabel("p")
    fig.show()
