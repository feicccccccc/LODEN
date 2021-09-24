"""
Model Based Reinforcement Learning with Analytic Mechanic
Fei, Cheung

Learning Lagrangian from Gym Pendulum from observation
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

from module import StructuredSO2LNN
from nn import MLP, PSD

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
    dataModule = Gym_DataModule(env, 100, 100, controls, time_horizon=1, batch_size=512)
    dataModule.setup()

    M = PSD([2, 64, 64, 1], activation='softplus')  # Mass matrix, M(cos q, sin q)
    V = MLP([2, 64, 64, 1], activation='softplus')  # Potential V(cos q, sin q)
    LagFunc = StructuredSO2LNN(massMatrixNet=M,
                               potentialNet=V,
                               dim=1)

    LagODE = NeuralODE(LagFunc, sensitivity='adjoint', solver='rk4').to(device)

    class Learner(pl.LightningModule):
        def __init__(self, model: torch.nn.Module):
            super().__init__()
            self.model = model
            self.loss_func = F.mse_loss
            self.n = self.model.defunc.m.n

        def forward(self, x_dx0, t_span):
            return self.model.trajectory(x_dx0, t_span)

        def training_step(self, batch, batch_idx):
            x, _, t, info = batch
            t_span = t[0, :, 0]

            # # x.shape: (bs, time, [cos q, sin q, dq])
            # cur_x_hat = self.model.trajectory(x[:, 0, :], t_span[:])
            #
            # loss = self.loss_func(x, cur_x_hat.transpose(0, 1))

            # For checking the loss
            dx1x2dq = self.model.defunc(0, x[:, 0, :])
            dx1 = torch.multiply(-torch.sin(info[:, 0, 0]), info[:, 0, 1])
            dx2 = torch.multiply(-torch.cos(info[:, 0, 0]), info[:, 0, 1])
            target = torch.stack([dx1, dx2, info[:, 0, 2]], dim=1)
            loss = self.loss_func(dx1x2dq, target)

            self.logger.experiment.add_scalars("loss", {"train": loss}, self.current_epoch)
            return loss

        def validation_step(self, batch, batch_idx):
            x, _, t, info = batch
            t_span = t[0, :, 0]

            # # x.shape: (bs, time, [cos q, sin q, dq])
            # cur_x_hat = self.model.trajectory(x[:, 0, :], t_span[:])
            #
            # loss = self.loss_func(x, cur_x_hat.transpose(0, 1))

            dx1x2dq = self.model.defunc(0, x[:, 0, :])
            dx1 = torch.multiply(-torch.sin(info[:, 0, 0]), info[:, 0, 1])
            dx2 = torch.multiply(-torch.cos(info[:, 0, 0]), info[:, 0, 1])
            target = torch.stack([dx1, dx2, info[:, 0, 2]], dim=1)
            loss = self.loss_func(dx1x2dq, target)

            self.log('val_loss', loss)
            self.logger.experiment.add_scalars("loss", {"val": loss}, self.current_epoch)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.model.parameters(), lr=1e-4)


    # Use pl for training
    learn = Learner(LagODE)
    early_stopping = EarlyStopping('val_loss', verbose=True, min_delta=1e-4, patience=20)
    trainer = pl.Trainer(callbacks=[early_stopping], min_epochs=1000, max_epochs=5000)  # tqdm bug mess with PyCharm
    trainer.fit(learn, dataModule)

    # Test for the Learned model
    def pendulum_lagrangian(q, p, l=1., m=1.):
        return p ** 2. / (2. * m * l) - m * 10. * l * (1-np.cos(q))

    # log can be view using tensorboard
    # tensorboard --logdir ./experiment/exp_grad/lightning_logs
    print("done")
