"""
Model Based Reinforcement Learning with Analytic Mechanic
Fei, Cheung

Learning Hamiltonian directly from Finite Difference gradient
The loss function will depends on the time horizon
"""
import numpy as np
import matplotlib

import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from torchdyn.models import NeuralODE

from module import HNN
from nn import MLP
from data import SHM_1D_ODE_DataModule
from utils import plot_HNN_1D_vector_field

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
    dataModule = SHM_1D_ODE_DataModule(init_conditions=init_conditions, time_horizon=2)  # 2 for finite difference
    dataModule.setup()
    # dataModule.plot_data(train=True)

    # Network for estimate the ODE
    HamFunc = HNN(MLP([2, 64, 1], activation='relu'), dim=1)

    # Make the model a NeuralODE class
    # implicit method for backward grad/ integrator methods
    HamODE = NeuralODE(HamFunc).to(device)

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
            x, _, t = batch
            # x.shape: (bs, time, dim)
            # t.shape: (bs, time, dim)

            dx_fd = (x[:, 1, :] - x[:, 0, :]) / (t[:, 1, :] - t[:, 0, :])  # finite difference
            dx_hat = self.model.defunc(0, x[:, 0, :])  # NN approx of dxdt

            loss = self.loss_func(dx_fd, dx_hat)
            self.logger.experiment.add_scalars("loss", {"train": loss}, self.current_epoch)
            return loss

        def validation_step(self, batch, batch_idx):
            x, _, t = batch

            dx_fd = (x[:, 1, :] - x[:, 0, :]) / (t[:, 1, :] - t[:, 0, :])  # finite difference
            dx_hat = self.model.defunc(0, x[:, 0, :])  # NN approx of dxdt

            loss = self.loss_func(dx_fd, dx_hat)
            self.log("val_loss", loss)
            self.logger.experiment.add_scalars("loss", {"train": loss}, self.current_epoch)

        def configure_optimizers(self):
            return torch.optim.Adam(self.model.parameters(), lr=1e-4)


    # Plot the dynamic before we learn the model
    # plot_HNN_1D_vector_field(HamODE)

    # Use pl for training
    learn = Learner(HamODE)
    early_stopping = EarlyStopping('val_loss', verbose=True, min_delta=1e-4, patience=20)
    trainer = pl.Trainer(callbacks=[early_stopping], min_epochs=1000, max_epochs=5000)  # tqdm bug mess with PyCharm
    trainer.fit(learn, dataModule)

    # Test for the Learned model
    # plot_HNN_1D_vector_field(HamODE)

    # log can be view using tensorboard
    # tensorboard --logdir ./experiment/exp_grad/lightning_logs
    print("done")
