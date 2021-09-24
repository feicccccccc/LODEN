"""
Model Based Reinforcement Learning with Analytic Mechanic
Fei, Cheung

Mainly for generating trajectory from gym gym_env
pl.LightningDataModule storing the data set for Supervised learning task
- 1D SHM
"""
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, random_split, TensorDataset

import pytorch_lightning as pl

from utils import from_pickle, to_pickle
from utils import SHM_1D_model


class SHM_1D_HNN_DataModule(pl.LightningDataModule):
    def __init__(self, data_dir=None,
                 batch_size=128,
                 init_conditions=None,
                 k=1., m=1.,
                 ):
        """
        If data_dir is None, we will generate the data set
        init_conditions should be list of tuple, i.e.
        [(x0, p0, t0, t1, steps), ...]
        TODO: Is it better to use Dict?
        Data as form: (bs, q_p, dq_dp, 1)  (no time order)
        """
        super().__init__()
        self.data_dir = data_dir
        self.file_name = "shm_1d_hnn_k{}_m{}_dataset.pkl".format(k, m)
        self.batch_size = batch_size
        self.init_conditions = init_conditions
        self.k = k
        self.m = m

        self.dataset = None

        self.train_set = None
        self.val_set = None
        self.test_set = None

    def setup(self, **kwargs):
        """
        Generate or get the dataset here
        :return: torch DataLoader
        """
        if self.data_dir is not None:
            path = os.path.join(self.data_dir, self.file_name)
            self.dataset = from_pickle(path)
        else:
            # Generate data set here
            X = []
            y = []
            t = []
            for init_cond in self.init_conditions:
                x0, p0, t0, t1, steps = init_cond
                q_p, dq_dp, _, ts = SHM_1D_model(x0=x0, p0=p0, t0=t0, t1=t1, steps=steps)
                X.append(q_p)
                y.append(dq_dp)
                t.append(ts)
            X = torch.cat(X, dim=0)
            y = torch.cat(y, dim=0)
            t = torch.cat(t, dim=0)
            self.dataset = TensorDataset(X, y, t)

            # save the dataset
            path = os.path.join("../../data/dataset", self.file_name)  # TODO: ugly af path, fix this sometime
            to_pickle(self.dataset, path)

        # Split the data set
        portion = [0.9, 0.1, 0.0]
        splits = [int(x * len(self.dataset)) for x in portion]
        if sum(splits) != len(self.dataset):
            splits[1] += len(self.dataset) - sum(splits)
        self.train_set, self.val_set, self.test_set = random_split(self.dataset, splits)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)

    def plot_data(self, train=False, val=False, test=False):
        all_indices = []
        if train:
            all_indices += self.train_set.indices
        if val:
            all_indices += self.val_set.indices
        if test:
            all_indices += self.test_set.indices

        if train or val or test is not False:
            X, y, t = self.dataset.tensors
        else:
            raise Exception('At least one data set to be true')

        if train and val and test:
            # for checking the whole trajectory
            fig = plt.figure(figsize=(12, 5))
            ax1 = fig.add_subplot(121)

            cur_idx = 0
            for traj in range(len(self.init_conditions)):
                # plot position
                _, _, _, _, step = self.init_conditions[traj]
                ax1.plot(t[cur_idx:step + cur_idx], X[cur_idx:step + cur_idx, 0], label='position_{}'.format(traj))
                # plot momentum
                ax1.plot(t[cur_idx:step + cur_idx], X[cur_idx:step + cur_idx, 1], label='momentum_{}'.format(traj))
                cur_idx += step

            ax1.set_xlabel('t')
            ax1.set_ylabel('q/p')
            ax1.legend()
            ax1.set_title("q(t) and p(t) graph of 1D SHM")
            #
            # fig.show()
        else:
            X = X[all_indices]
            # y = y[all_indices]
            # t = t[all_indices]

        # === Plot Phase diagram ===
        ax2 = fig.add_subplot(122)
        ax2.scatter(X[:, 0], X[:, 1], color='blue', label='phase')

        ax2.set_xlabel('q')
        ax2.set_ylabel('p')
        ax2.set_title("Phase Diagram of 1D SHM")
        # fig.show()

        # TODO: Plot ground truth vector field


class SHM_1D_ODE_DataModule(pl.LightningDataModule):
    def __init__(self, data_dir=None,
                 batch_size=128,
                 init_conditions=None,
                 k=1., m=1.,
                 time_horizon=100,  # Use for generating the samples
                 get_velocity=False  # should we output momentum or velocity
                 ):
        """
        If data_dir is None, we will generate the data set
        init_conditions should be list of tuple, i.e.
        [(x0, p0, t0, t1, steps), ...]

        data set as tuple: (data, t)
        t: (num of traj, ts)
        data: (num of traj, q_p)  TODO: data shape not clear at this moment
        """
        super().__init__()
        self.data_dir = data_dir
        self.file_name = "shm_1d_ode_k{}_m{}_dataset.pkl".format(k, m)
        self.batch_size = batch_size
        self.init_conditions = init_conditions
        self.k = k
        self.m = m
        self.get_velocity = get_velocity

        self.dataset = None
        self.time_horizon = time_horizon

        self.train_set = None
        self.val_set = None
        self.test_set = None

    def setup(self, **kwargs):
        """
        Generate or get the dataset here
        :return: torch DataLoader
        """
        if self.data_dir is not None:
            path = os.path.join(self.data_dir, self.file_name)
            self.dataset = from_pickle(path)
        else:
            # Generate data set here
            X = []
            y = []
            z = []
            t = []
            for init_cond in self.init_conditions:
                x0, p0, t0, t1, steps = init_cond
                q_p, dq_dp, ddq, ts = SHM_1D_model(x0=x0, p0=p0, t0=t0, t1=t1, steps=steps, k=self.k, m=self.m)

                # === Split the trajectory here ===
                assert len(q_p) >= self.time_horizon, \
                    "Trajectory not long enough to split accordingly, " \
                    "try reducing the time horizon parameters"

                num_of_trajs = len(q_p) - self.time_horizon + 1
                for i in range(num_of_trajs):
                    X.append(q_p[i:i + self.time_horizon, :])
                    y.append(dq_dp[i:i + self.time_horizon, :])
                    z.append(ddq[i:i + self.time_horizon, :])
                    t.append(ts[i:i + self.time_horizon, :])

            X = torch.stack(X, dim=0)  # (bs, time, dim)
            y = torch.stack(y, dim=0)
            z = torch.stack(z, dim=0)
            t = torch.stack(t, dim=0)

            if self.get_velocity:
                X[:, :, 1] = X[:, :, 1] / self.m

            self.dataset = TensorDataset(X, y, z, t)

            # save the dataset
            path = os.path.join("../../data/dataset", self.file_name)  # TODO: ugly af path, fix this sometime
            to_pickle(self.dataset, path)

        # Split the data set
        portion = [0.9, 0.1, 0.0]
        splits = [int(x * len(self.dataset)) for x in portion]
        # Make sure the split sum to 1, and put all of them into val set
        if sum(splits) != len(self.dataset):
            splits[1] += len(self.dataset) - sum(splits)
        self.train_set, self.val_set, self.test_set = random_split(self.dataset, splits)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True)

    def plot_data(self, train=False, val=False, test=False):
        all_indices = []
        if train:
            all_indices += self.train_set.indices
        if val:
            all_indices += self.val_set.indices
        if test:
            all_indices += self.test_set.indices

        if train or val or test is not False:
            X, y, _, t = self.dataset.tensors
        else:
            raise Exception('At least one data set to be true')

        if train and val and test:
            # for checking the whole trajectory
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111)

            for traj in range(len(self.init_conditions)):
                # plot position
                _, _, _, _, step = self.init_conditions[traj]
                ax.plot(self.dataset.tensors[2][traj, :, 0],
                        self.dataset.tensors[0][traj, :, 0],
                        label='position_{}'.format(traj))
                # plot momentum
                ax.plot(self.dataset.tensors[2][traj, :, 0],
                        self.dataset.tensors[0][traj, :, 1],
                        label='momentum_{}'.format(traj))

            ax.legend()
            ax.set_title("q(t) and p(t) graph of 1D SHM")
            # fig.show()
        else:
            X = X[all_indices]
            # y = y[all_indices]
            # t = t[all_indices]

        # === Plot Phase diagram ===
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        ax.scatter(X[:, :, 0], X[:, :, 1], color='blue', label='phase')

        ax.set_xlabel(r"$q$")
        ax.set_ylabel(r"$p$")
        ax.set_title("Phase Diagram of 1D SHM")
        # fig.show()

        # TODO: Plot ground truth vector field


class SHM_1D_LNN_DataModule(pl.LightningDataModule):
    def __init__(self, data_dir=None,
                 batch_size=128,
                 init_conditions=None,
                 k=1., m=1.,
                 ):
        """
        If data_dir is None, we will generate the data set
        init_conditions should be list of tuple, i.e.
        [(x0, p0, t0, t1, steps), ...]
        TODO: Is it better to use Dict?
        """
        super().__init__()
        self.data_dir = data_dir
        self.file_name = "shm_1d_lnn_k{}_m{}_dataset.pkl".format(k, m)
        self.batch_size = batch_size
        self.init_conditions = init_conditions
        self.k = k
        self.m = m

        self.dataset = None

        self.train_set = None
        self.val_set = None
        self.test_set = None

    def setup(self, **kwargs):
        """
        Generate or get the dataset here
        :return: torch DataLoader
        """
        if self.data_dir is not None:
            path = os.path.join(self.data_dir, self.file_name)
            self.dataset = from_pickle(path)
        else:
            # Generate data set here
            X = []
            y = []
            t = []
            for init_cond in self.init_conditions:
                x0, p0, t0, t1, steps = init_cond
                q_p, _, _, ts = SHM_1D_model(x0=x0, p0=p0, t0=t0, t1=t1, steps=steps)  # TODO: merge acceleration
                ddq = -1. / 1. * q_p[:, 0]

                X.append(q_p)  # for k=1, m=1 -> p = dq
                y.append(ddq)
                t.append(ts)

            X = torch.cat(X, dim=0)
            y = torch.cat(y, dim=0)
            t = torch.cat(t, dim=0)
            self.dataset = TensorDataset(X, y, t)

            # save the dataset
            path = os.path.join("../../data/dataset", self.file_name)  # TODO: ugly af path, fix this sometime
            to_pickle(self.dataset, path)

        # Split the data set
        portion = [0.9, 0.1, 0.0]
        splits = [int(x * len(self.dataset)) for x in portion]
        if sum(splits) != len(self.dataset):
            splits[1] += len(self.dataset) - sum(splits)
        self.train_set, self.val_set, self.test_set = random_split(self.dataset, splits)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True)


if __name__ == "__main__":
    # (x0, p0, t0, t1, steps)
    test_init_conditions = [
        (1, 0, 0, 2. * np.pi, 100),
        (2, 0, 0, 2. * np.pi, 100),
        (0.5, 0, 0, 2. * np.pi, 100),
        (1.5, 0, 0, 2. * np.pi, 100)
    ]
    test_DataModule = SHM_1D_ODE_DataModule(init_conditions=test_init_conditions, k=1., m=1.)
    test_DataModule.setup()

    # test_DataModule = SHM_1D_DataModule(data_dir='./dataset')
    # test_DataModule.setup()
    test_DataModule.plot_data(train=True, val=True, test=True)

    print("done")
