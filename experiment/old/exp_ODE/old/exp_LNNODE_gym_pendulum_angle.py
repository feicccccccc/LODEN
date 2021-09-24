"""
Model Based Reinforcement Learning with Analytic Mechanic
Fei, Cheung

Learning Lagrangian from Gym Pendulum
Train by Pytorch Lighting

TODO: Use configuration file to config the parameters
"""
import numpy as np
import matplotlib

import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from torchdyn.models import NeuralODE

from module import LNN
from nn import MLP

from gym_env import CartPoleCustomEnv
from data import Gym_DataModule
from utils import plot_LNN_1D_traj, plot_LNN_1D_vector_field, plot_LNN_1D_surface

# matplotlib.use('macosx')  # Disable PyCharm SciView, Can turn on for interactive mode

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

if __name__ == "__main__":
    env = CartPoleCustomEnv()
    controls = np.array([[0.]])  # (bs, dim)

    # env, time step, traj, control
    # TODO: again time_horizon can be way off for learning
    dataModule = Gym_DataModule(env, 100, 100, controls, time_horizon=3)  # Can't train coz loss explode
    dataModule.setup()

    # Network for estimate the Hamiltonian
    lagrangian = MLP([2, 64, 64, 1], activation='softplus')
    LagFunc = LNN(lagrangian, dim=1)
    # TODO: Finish the HNN->LNN with different coordinate
    LagODE = NeuralODE(LagFunc).to(device)

    def pendulum_hamiltonian(q, p, l=1., m=1.):
        return p ** 2. / (2. * m * l) - m * 10. * l * (1-np.cos(q))

    class Learner(pl.LightningModule):
        def __init__(self, model: torch.nn.Module):
            super().__init__()
            self.model = model
            self.loss_func = F.mse_loss
            self.n = self.model.defunc.m.n

        def forward(self, x):
            """
            :param x: (q, p)
            :return: (dHdp, -dHdq)
            """
            return self.model.defunc(0, x)

        def training_step(self, batch, batch_idx):
            _, _, t, x = batch

            xs0 = x[:, 0, :self.n]
            t_span = t[0, :, 0]

            dxs0 = (x[:, 1, :self.n] - x[:, 0, :self.n]) / (t[:, 1, :] - t[:, 0, :])  # finite difference

            x_dx0 = torch.cat([xs0, dxs0], dim=1)
            cur_x_hat = self.model.trajectory(x_dx0, t_span)

            loss = self.loss_func(x, cur_x_hat.transpose(0, 1))

            self.logger.experiment.add_scalars("loss", {"train": loss}, self.current_epoch)
            return loss

        def validation_step(self, batch, batch_idx):
            _, _, t, x = batch

            xs0 = x[:, 0, :self.n]
            t_span = t[0, :, 0]

            dxs0 = (x[:, 1, :self.n] - x[:, 0, :self.n]) / (t[:, 1, :] - t[:, 0, :])  # finite difference

            x_dx0 = torch.cat([xs0, dxs0], dim=1)
            cur_x_hat = self.model.trajectory(x_dx0, t_span)

            loss = self.loss_func(x, cur_x_hat.transpose(0, 1))

            self.log('val_loss', loss)
            self.logger.experiment.add_scalars("loss", {"val": loss}, self.current_epoch)
            return loss

        def test_step(self, batch, batch_idx):
            _, _, t, x = batch

            xs0 = x[:, 0, :self.n]
            t_span = t[0, :, 0]

            dxs0 = (x[:, 1, :self.n] - x[:, 0, :self.n]) / (t[:, 1, :] - t[:, 0, :])  # finite difference

            x_dx0 = torch.cat([xs0, dxs0], dim=1)
            cur_x_hat = self.model.trajectory(x_dx0, t_span)

            loss = self.loss_func(x, cur_x_hat.transpose(0, 1))

            self.logger.experiment.add_scalars("loss", {"test": loss}, self.current_epoch)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.model.parameters(), lr=1e-4)


    # Plot the dynamic before we learn the model
    # plot_LNN_1D_traj(HamODE)
    # plot_LNN_1D_vector_field(HamODE)
    # plot_LNN_1D_surface(HamODE, ground_truth=SHM_hamiltonian)

    # Use pl for training
    learn = Learner(LagODE)
    early_stopping = EarlyStopping('val_loss', verbose=True, min_delta=1e-4, patience=20)
    trainer = pl.Trainer(callbacks=[early_stopping], min_epochs=1000, max_epochs=5000)  # tqdm bug mess with PyCharm
    trainer.fit(learn, dataModule)

    # TODO: Add animation on how the learned hamiltonian evolve
    # Test for the Learned model
    # plot_LNN_1D_traj(HamODE)
    plot_LNN_1D_vector_field(LagODE)
    plot_LNN_1D_surface(LagODE, ground_truth=pendulum_hamiltonian)

    # log can be view using tensorboard
    # tensorboard --logdir ./experiment/exp_grad/lightning_logs
    print("done")
