"""
Model Based Reinforcement Learning with Analytic Mechanic
Fei, Cheung

Learning gradient directly from Neural ODE
The loss function will depends on the time horizon
"""
import numpy as np
import matplotlib

import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from torchdyn.models import NeuralODE

from nn import MLP
from data import SHM_1D_ODE_DataModule
from utils import plot_2D_vector_field

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

    # TODO: try different data set
    dataModule = SHM_1D_ODE_DataModule(init_conditions=init_conditions, time_horizon=10)
    dataModule.setup()
    dataModule.plot_data(train=True)

    # Network for estimate the ODE
    df = MLP([2, 64, 2], activation='relu')  # TODO: check with toy data set on the trajectory

    # Make the model a NeuralODE class
    # implicit method for backward grad/ integrator methods
    ODE = NeuralODE(df).to(device)

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
            # TODO: Maybe we can try different loss function. Gaussian noise assumption seems good here.

        def forward(self, x):
            """
            :param x: (q, p)
            :return: (dHdp, -dHdq)
            """
            return self.model.defunc(0, x)

        def training_step(self, batch, batch_idx):
            # x: (3, 100, 2)  batch, time, dim
            # y: (3, 100, 2)
            # t: (3, 2)
            x, _, _, t = batch

            xs0 = x[:, 0, :]
            t_span = t[0, :, 0]  # Assume all trajectory have the same time stamps
            # TODO: unify shape across module

            # take X (bs, dim), and t (number of time steps)
            cur_x_hat = self.model.trajectory(xs0, t_span)
            # return shape (time, batch, dim)
            loss = self.loss_func(x, cur_x_hat.transpose(0, 1))
            self.logger.experiment.add_scalars("loss", {"train": loss}, self.current_epoch)
            return loss

        def validation_step(self, batch, batch_idx):
            x, _, _, t = batch

            xs0 = x[:, 0, :]
            t_span = t[0, :, 0]

            cur_x_hat = self.model.trajectory(xs0, t_span)
            loss = self.loss_func(x, cur_x_hat.transpose(0, 1))
            self.log('val_loss', loss)
            self.logger.experiment.add_scalars("loss", {"val": loss}, self.current_epoch)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.model.parameters(), lr=1e-4)


    # Plot the dynamic before we learn the model
    plot_2D_vector_field(ODE)

    # Use pl for training
    learn = Learner(ODE)
    early_stopping = EarlyStopping('val_loss', verbose=True, min_delta=1e-4, patience=20)
    trainer = pl.Trainer(callbacks=[early_stopping], min_epochs=1000, max_epochs=5000)  # tqdm bug mess with PyCharm
    trainer.fit(learn, dataModule)

    # Test for the Learned model
    plot_2D_vector_field(ODE)

    # log can be view using tensorboard
    # tensorboard --logdir ./experiment/exp_grad/lightning_logs
    print("done")
