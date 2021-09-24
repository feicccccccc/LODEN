"""
Model Based Reinforcement Learning with Analytic Mechanic
Fei, Cheung

Generate the trajectory from Gym Env
Should output a dataModule for Pytorch lightning
"""
import os
import warnings
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
from numpy import ndarray
from torch.utils.data import DataLoader, random_split, TensorDataset

import pytorch_lightning as pl

import gym
from gym_env import CartPoleCustomEnv, CartPoleCustomEnv


class Gym_DataModule(pl.LightningDataModule):
    def __init__(self,
                 env: gym.Env,
                 time_step,  # for gym
                 num_traj,
                 controls,
                 time_horizon=10,  # for ODE
                 data_dir=None,
                 batch_size=128,
                 precision=32
                 ):
        """
        Generate data set from Gym Env for Supervised Learning.
        Control should be list of constant as probe to the control dynamic.
        Check this file for usage.
        """
        super().__init__()
        self.env = env
        self.controls = controls  # Should be list of list base on gym_env

        self.data_dir = data_dir
        self.file_name = "{}_dataset.pkl".format(type(self.env).__name__)

        self.batch_size = batch_size
        self.time_horizon = time_horizon

        self.num_traj = num_traj
        self.time_step = time_step

        self.dataset = None

        self.train_set = None
        self.val_set = None
        self.test_set = None

        self._precision = precision

    def setup(self, **kwargs):
        """
        Generate or get the dataset here
        :return: torch DataLoader
        """
        env = self.env
        env.seed(0)  # TODO: Record Seed and generator for reproducibility
        dt = env.dt  # time step for each steps.

        if self.data_dir is not None:
            path = os.path.join(self.data_dir, self.file_name)
            self.dataset = from_pickle(path)
        else:
            # Generate data set here
            observations = []
            infos = []

            # Generate from trajectory
            for control in self.controls:
                trajs = []
                extras = []  # for extra info

                for traj in range(self.num_traj):
                    env.reset()
                    traj = []
                    extra = []

                    # Data should be (cur obs, cur u), Gym data is (next obs, cur action)
                    prev_obs, _, _, _ = env.step(control)
                    for step in range(self.time_step):
                        traj.append(prev_obs)

                        next_obs, reward, done, info = env.step(control)

                        if isinstance(self.env, CartPoleCustomEnv):
                            extra.append(np.array([info['th'], info['dth'], info['ddth']]))
                        elif isinstance(self.env, CartPoleCustomEnv):
                            extra.append(np.array([info['ddx'], info['ddth']]))
                        else:
                            extra.append(np.array([0]))
                        if done:
                            raise ValueError('Trajectory end in the middle of target time step')

                        prev_obs = next_obs
                    trajs.append(traj)
                    extras.append(extra)

                observations.append(trajs)
                infos.append(extras)

            # observation data
            observations = torch.Tensor(observations)  # (control, traj, time, dim)
            infos = torch.Tensor(infos)  # (control, traj, time, dim)

            # Control data
            num_control, control_dim = self.controls.shape
            control = torch.Tensor(self.controls).reshape([num_control, 1, 1, control_dim])
            control = control.expand(num_control, self.num_traj, self.time_step, control_dim)

            # Time stamp
            t_span = torch.linspace(dt, self.time_step * dt, self.time_step)
            t_span = t_span.reshape(1, 1, t_span.shape[0], 1)
            t_span = t_span.expand(num_control, self.num_traj, self.time_step, 1)

            # === Split the trajectory here ===
            X = []
            U = []
            T = []
            I = []

            num_of_trajs = self.time_step - self.time_horizon + 1
            assert num_of_trajs >= 1, \
                "Trajectory not long enough to split accordingly, " \
                "try reduce the time horizon parameters"

            num_control, num_traj, time_step, observation_dim = observations.shape
            for control_idx in range(num_control):
                for traj_idx in range(num_traj):
                    for i in range(num_of_trajs):
                        X.append(observations[control_idx, traj_idx, i:i + self.time_horizon, :])
                        U.append(control[control_idx, traj_idx, i:i + self.time_horizon, :])
                        T.append(t_span[control_idx, traj_idx, i:i + self.time_horizon, :])
                        I.append(infos[control_idx, traj_idx, i:i + self.time_horizon, :])

            if self._precision == 64:
                X = torch.stack(X, dim=0)  # obs
                U = torch.stack(U, dim=0)  # control
                T = torch.stack(T, dim=0)  # time
                I = torch.stack(I, dim=0)  # info
            elif self._precision == 32:
                X = torch.stack(X, dim=0).float()  # obs
                U = torch.stack(U, dim=0).float()  # control
                T = torch.stack(T, dim=0).float()  # time
                I = torch.stack(I, dim=0).float()  # info
            else:
                raise ValueError("Precision does not match float / Double")

            self.dataset = TensorDataset(X, U, T, I)

            # save the dataset
            path = os.path.join("../../data/dataset", self.file_name)  # TODO: ugly af path, fix this sometime
            to_pickle(self.dataset, path)

        # Split the data set
        portion = [0.9, 0.1, 0.0]
        splits = [int(x * len(self.dataset)) for x in portion]
        self.train_set, self.val_set, self.test_set = random_split(self.dataset, splits)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)


# Generate all possible trajectory with random action
class Gym_DataModule_rng(pl.LightningDataModule):
    def __init__(self,
                 env: gym.Env,
                 num_data,
                 low: ndarray,  # for Control Distribution (Uniform)
                 high: ndarray,
                 no_act_portion=0.5,  # no control vs control portion, float
                 time_horizon=10,  # for ODE
                 data_dir=None,
                 batch_size=128,
                 use_img=False
                 ):
        """
        Generate data set from Gym Env for Supervised Learning.
        Control should be list of constant as probe to the control dynamic.
        Check this file for usage.
        """
        super().__init__()
        self.env = env

        self.low = low
        self.high = high

        self.data_dir = data_dir
        if use_img:
            self.file_name = "{}_img_rng_dataset.pkl".format(type(self.env).__name__)
        else:
            self.file_name = "{}_rng_dataset.pkl".format(type(self.env).__name__)

        self.batch_size = batch_size
        self.time_horizon = time_horizon

        self.num_data = num_data
        self.portion = no_act_portion

        self.dataset = None

        self.train_set = None
        self.val_set = None
        self.test_set = None

        self._use_img = use_img
        if self._use_img:
            self.time_horizon += 1  # For one more frame to encode the difference

    def setup(self, **kwargs):
        """
        Generate or get the dataset here
        :return: torch DataLoader
        """
        env = self.env
        env.seed(0)  # TODO: Record Seed and generator for reproducibility
        dt = env.dt  # time step for each steps.

        if self.data_dir is not None:
            path = os.path.join(self.data_dir, self.file_name)
            self.dataset = from_pickle(path)
        else:
            # Generate data set here
            trajs = []
            extras = []  # for extra info
            controls = []
            trajs_img = []
            data_count = 0

            while data_count < self.num_data:
                traj = []
                extra = []
                traj_img = []

                # Decide current control
                if np.random.uniform() <= self.portion:
                    control = np.zeros_like(self.high)
                else:
                    control = np.random.uniform(self.low, self.high)

                prev_obs, _ = env.reset()  # Assume reset will generate uniform distribution in state space
                if self._use_img:
                    prev_img = self.env.render('rgb_array')

                success_flag = True
                for t in range(self.time_horizon):
                    traj.append(prev_obs)

                    if isinstance(self.env, CartPoleCustomEnv) and self._use_img:
                        print(data_count)
                        traj_img.append(preproc_pend(prev_img))

                    next_obs, reward, done, info = env.step(control)
                    if self._use_img:
                        next_img = env.render('rgb_array')

                    if isinstance(self.env, CartPoleCustomEnv):
                        extra.append(np.array([info['th'], info['dth'], info['ddth']]))

                    elif isinstance(self.env, CartPoleCustomEnv):
                        extra.append(np.array([info['ddx'], info['ddth']]))
                    else:
                        extra.append(np.array([0]))
                    if done:
                        warnings.warn('Trajectory end in the middle of target time step')
                        success_flag = False
                        break

                    prev_obs = next_obs
                    if self._use_img:
                        prev_img = next_img

                if success_flag:
                    trajs.append(traj)
                    extras.append(extra)
                    controls.append(control)
                    trajs_img.append(traj_img)
                    data_count += 1

            # Constant Control data
            controls = torch.tensor(controls)[:, None, :]
            controls = controls.repeat(1, self.time_horizon, 1)

            # Time stamp
            t_span = torch.linspace(dt, self.time_horizon * dt, self.time_horizon)
            t_span = t_span.repeat(self.num_data, 1).view(self.num_data, self.time_horizon, 1)

            X = torch.Tensor(trajs).float()  # obs
            U = controls.float()
            T = t_span.float()
            I = torch.Tensor(extras).float()

            if self._use_img:
                X = torch.Tensor(trajs).float()[:, 1:self.time_horizon, :]  # obs
                U = controls.float()[:, 1:self.time_horizon, :]
                T = t_span.float()[:, 1:self.time_horizon, :]
                I = torch.Tensor(extras).float()[:, 1:self.time_horizon, :]

                Images = torch.Tensor(trajs_img).float()
                P = Images[:, 1:self.time_horizon, :, :]
                dP = Images[:, 1:self.time_horizon, :, :] - Images[:, 0:self.time_horizon-1, :, :]
                P_dP = torch.stack([P, dP], dim=4)
                self.dataset = TensorDataset(X, U, T, I, P_dP)
            else:
                self.dataset = TensorDataset(X, U, T, I)

            # save the dataset
            path = os.path.join("../../data/dataset", self.file_name)  # TODO: ugly af path, fix this sometime
            to_pickle(self.dataset, path)

        # Split the data set
        percent = [0.9, 0.1, 0.0]
        splits = [int(x * len(self.dataset)) for x in percent]
        self.train_set, self.val_set, self.test_set = random_split(self.dataset, splits)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True)


def preproc_pend(img):
    gray = cv2.cvtColor(img[220: 780, 220: 780, :], cv2.COLOR_BGR2GRAY)  # Grey Scale
    img = cv2.resize(gray, [28, 28])  # Rescale image
    img[img == 255] = 0  # Black background (0 seems better for training)
    output = img / 255
    return output

if __name__ == "__main__":
    test_env = CartPoleCustomEnv()
    # test_env = PendulumCustomEnv()
    test_controls = np.array([[-1.], [0.], [1.], [2.]])  # (different control, control dim)

    # low = np.array([-2])
    # high = np.array([2])
    # test_DataModule = Gym_DataModule_rng(test_env, 100, low, high, 0.7, 10)
    # test_DataModule = Gym_DataModule(test_env, 45, 50, test_controls, time_horizon=5)
    # test_DataModule.setup()

    # test_DataModule.plot_data(train=True, val=True, test=True)

    print("done")
