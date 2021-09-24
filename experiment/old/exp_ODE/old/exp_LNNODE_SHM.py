"""
Model Based Reinforcement Learning with Analytic Mechanic
Fei, Cheung

Learning Lagrangian directly from trajectory
The loss function will depends on the time horizon
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
from utils import plot_LNN_1D_vector_field

# matplotlib.use('macosx')  # Disable PyCharm SciView, Can turn on for interactive mode

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

if __name__ == "__main__":
    # Generate data for SHM motion
    init_conditions = [
        (1, 0, 0, 2. * np.pi, 100),
        (2, 0, 0, 2. * np.pi, 100),
        (0.5, 0, 0, 2. * np.pi, 100),
        (1.5, 0, 0, 2. * np.pi, 100)
    ]

    # TODO: if time horizon is too large, it is easy to get diverge of the traj and hence loss overflow
    # TODO: adaptive time_horizon training to boost training speed?
    dataModule = SHM_1D_ODE_DataModule(init_conditions=init_conditions, time_horizon=10)
    dataModule.setup()
    # dataModule.plot_data(train=True)

    # Network for estimate the ODE
    LagFunc = LNN(MLP([2, 64, 64, 1], activation='softplus'), dim=1)

    # Make the model a NeuralODE class
    # implicit method for backward grad/ integrator methods
    LagODE = NeuralODE(LagFunc).to(device)

    # Use PyTorch Lightning for training
    # Brief Summary of using PyTorch Lightning
    # define the network
    # define the data loader
    # pl will handle training script

    class Learner(pl.LightningModule):
        def __init__(self, model: torch.nn.Module):
            super().__init__()
            self.model = model
            self.loss_func = F.mse_loss
            self.n = self.model.defunc.m.n
            # TODO: Try weighted error. Larger weight for point near x0

        def forward(self, x):
            return self.model.defunc(0, x)

        def training_step(self, batch, batch_idx):
            # bs, time, dim
            x, _, _, t = batch

            # x: (bs, t, [q, dq])
            xs0 = x[:, 0, :self.n]
            # t: (bs, t, 0)
            t_span = t[0, :, 0]  # Assume all trajectory have the same time stamps, which only work for model.trajectory

            # generate finite difference for initial dq
            dxs0 = (x[:, 1, :self.n] - x[:, 0, :self.n]) / (t[:, 1, :] - t[:, 0, :])  # finite difference

            x_dx0 = torch.cat([xs0, dxs0], dim=1)

            # take X (bs, dim), and t (number of time steps)
            # and output the trajectory in system of 1st order ODE
            cur_x_hat = self.model.trajectory(x_dx0, t_span)
            # return shape (time, batch, dim)

            loss = self.loss_func(x, cur_x_hat.transpose(0, 1))

            # TODO: Second order ODE loss is not clear here
            self.logger.experiment.add_scalars("loss", {"train": loss}, self.current_epoch)
            return loss

        def validation_step(self, batch, batch_idx):
            x, _, _, t = batch
            xs0 = x[:, 0, :self.n]
            dxs0 = (x[:, 1, :self.n] - x[:, 0, :self.n]) / (t[:, 1, :] - t[:, 0, :])

            x_dx0 = torch.cat([xs0, dxs0], dim=1)
            t_span = t[0, :, 0]

            cur_x_hat = self.model.trajectory(x_dx0, t_span)
            loss = self.loss_func(x, cur_x_hat.transpose(0, 1))

            self.log('val_loss', loss)
            self.logger.experiment.add_scalars("loss", {"val": loss}, self.current_epoch)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.model.parameters(), lr=1e-4)

    # Use pl for training
    learn = Learner(LagODE)

    # need longer patience for trajectory loss
    early_stopping = EarlyStopping('val_loss', verbose=True, min_delta=1e-5, patience=50)
    trainer = pl.Trainer(callbacks=[early_stopping], min_epochs=1000, max_epochs=5000)  # tqdm bug mess with PyCharm
    trainer.fit(learn, dataModule)

    # Test for the Learned model
    plot_LNN_1D_vector_field(LagODE)

    # log can be view using tensorboard
    # tensorboard --logdir ./experiment/exp_grad/lightning_logs
    print("done")
