"""
Model Based Reinforcement Learning with Analytic Mechanic
Fei, Cheung

Learning Lagrangian of Simple Harmonic Motion (SHM) from Finite Difference gradient
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
from data import SHM_1D_ODE_DataModule
from utils import plot_LNN_1D_traj, plot_LNN_1D_vector_field, plot_LNN_1D_surface

# matplotlib.use('macosx')  # Disable PyCharm SciView, Can turn on for interactive mode

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

if __name__ == "__main__":
    # Generate data for SHM motion
    init_conditions = [
        (1, 0, 0, 2. * np.pi, 500),
        (2, 0, 0, 2. * np.pi, 500),
        (0.5, 0, 0, 2. * np.pi, 500),
        (1.5, 0, 0, 2. * np.pi, 500)
    ]
    dataModule = SHM_1D_ODE_DataModule(init_conditions=init_conditions, time_horizon=2, get_velocity=True)
    dataModule.setup()

    # lagrangian, later for generate the ground truth
    def SHM_lagrangian(q, dq, k=1., m=1.):
        return m * dq ** 2. / 2. - k * q ** 2. / 2.

    # Network for estimate the Hamiltonian
    # TODO: Interesting that 1 layer almost never work
    # TODO: Increasing complexity increase sample efficiency
    LagFunc = LNN(0, MLP([2, 64, 64, 1], activation='softplus'))
    LagODE = NeuralODE(LagFunc).to(device)

    class Learner(pl.LightningModule):
        def __init__(self, model: torch.nn.Module):
            super().__init__()
            self.model = model
            self.loss_func = F.mse_loss
            # TODO: Maybe we can try different loss function. Gaussian noise assumption seems good here.

        def forward(self, x):
            return self.model.defunc(0, x)

        def training_step(self, batch, batch_idx):
            q_dq, _, ddq, t = batch  # q_dq: (bs, time, dim)
            # x: q_dq, target: ddq

            ddq_fd = (q_dq[:, 1, 1] - q_dq[:, 0, 1]) / (t[:, 1, 0] - t[:, 0, 0])  # pick the speed dimension and do FD
            dq_ddq_fd = torch.stack([q_dq[:, 0, 1], ddq_fd], dim=1)
            dq_ddq_hat = self.model.defunc(0, q_dq[:, 0, :])  # from LNN output

            loss = self.loss_func(dq_ddq_fd, dq_ddq_hat)
            self.logger.experiment.add_scalars("loss", {"train": loss}, self.current_epoch)
            return loss

        def validation_step(self, batch, batch_idx):
            q_dq, _, ddq, t = batch
            ddq_fd = (q_dq[:, 1, 1] - q_dq[:, 0, 1]) / (t[:, 1, :] - t[:, 0, :])
            dq_ddq_hat = self.model.defunc(0, q_dq[:, 0, :])

            loss = self.loss_func(ddq_fd, dq_ddq_hat[:, 1])
            self.log('val_loss', loss)
            self.logger.experiment.add_scalars("loss", {"val": loss}, self.current_epoch)
            return loss

        def test_step(self, batch, batch_idx):
            q_dq, _, ddq, t = batch
            ddq_fd = (q_dq[:, 1, 1] - q_dq[:, 0, 1]) / (t[:, 1, :] - t[:, 0, :])
            dq_ddq_hat = self.model.defunc(0, q_dq[:, 0, :])

            loss = self.loss_func(ddq_fd, dq_ddq_hat[:, 1])
            self.logger.experiment.add_scalars("loss", {"test": loss}, self.current_epoch)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.model.parameters(), lr=1e-4)


    # Plot the dynamic before we learn the model
    # plot_LNN_1D_traj(LagODE)
    # plot_LNN_1D_vector_field(LagODE)
    # plot_LNN_1D_surface(LagODE)

    # Use pl for training
    learn = Learner(LagODE)
    early_stopping = EarlyStopping('val_loss', verbose=True, min_delta=1e-4, patience=20)
    trainer = pl.Trainer(callbacks=[early_stopping], min_epochs=500, max_epochs=5000)
    trainer.fit(learn, dataModule)

    # TODO: Add animation on how the learned hamiltonian evolve
    # Test for the Learned model
    # plot_LNN_1D_traj(LagODE)
    plot_LNN_1D_vector_field(LagODE)
    plot_LNN_1D_surface(LagODE)

    # log can be view using tensorboard
    # tensorboard --logdir ./experiment/exp_grad/lightning_logs
    print("done")
