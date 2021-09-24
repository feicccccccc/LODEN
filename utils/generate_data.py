"""
Function for generate different data set
TODO: Restructure into data module
"""
import pickle

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data


def SHM_1D_model(x0=1., p0=0., t0=0., t1=2 * np.pi, steps=100, k=1., m=1.):
    """
    1D Simple Harmonic Motion model
    EOM:
    H = p^2 / 2m + 1/2 k x^2
    [dq/dt, dp/dt] = [p/m, -kx]

    SOL:
    q = c1 * sin(sqrt(k/m) * t) + c2 * cos(sqrt(k/m) * t)
    p = sqrt(k/m) (c1 * cos(sqrt(k/m) * t) - c2 * sin(sqrt(k/m) * t))

    :param x0: initial position
    :param p0: initial momentum
    :param t0: initial time of the trajectory
    :param t1: final time of the trajectory
    :param steps: number of samples/steps in the trajectory
    :param k: Spring Constant
    :param m: Mass
    :return: data set [Canonical coordinate, Symplectic gradient, time stamps]
    """
    assert k > 0 or m > 0, "Spring Constant k and mass m must larger than 0"

    # Solve for c1 and c2
    period = np.sqrt(k * m)
    # Solve for ivp
    A = np.array([
        [np.sin(period * t0), -np.cos(period * t0)],
        [np.sqrt(k * m) * np.cos(period * t0), np.sqrt(k * m) * np.sin(period * t0)]
    ])
    b = np.array([x0, p0])
    c1, c2 = np.linalg.solve(A, b)

    time = torch.linspace(t0, t1, steps).reshape(-1, 1)

    # q_p: (time, [q,p]) Canonical coordinate
    q_p = torch.cat([
        c1 * torch.sin(period * time) - c2 * torch.cos(period * time),
        np.sqrt(k * m) * (c1 * torch.cos(period * time) + c2 * torch.sin(period * time))
    ], 1)

    # dq_dp: (time, [dq, dp]) Symplectic gradient
    dq_dp = torch.cat([
        period * (c1 * torch.cos(period * time) + c2 * torch.sin(period * time)),
        np.sqrt(k * m) * period * (-c1 * torch.sin(period * time) + c2 * torch.cos(period * time))
    ], 1)

    ddq = -k / m * q_p[:, [0]]  # a = -k/m x
    return q_p, dq_dp, ddq, time


def to_pickle(thing, path):
    with open(path, 'wb') as handle:
        pickle.dump(thing, handle, protocol=pickle.HIGHEST_PROTOCOL)


def from_pickle(path):
    with open(path, 'rb') as handle:
        thing = pickle.load(handle)
    return thing


if __name__ == "__main__":
    # For reproducibility
    torch.manual_seed(0)

    # Test for 1D SHM
    test_q_p, test_dq_dp, test_time = SHM_1D_model(x0=2, p0=1, t1=2 * np.pi * 0.75)
    test_train = data.TensorDataset(test_q_p, test_dq_dp, test_time)
    testLoader = data.DataLoader(test_train, batch_size=128, shuffle=True)

    print("done")
