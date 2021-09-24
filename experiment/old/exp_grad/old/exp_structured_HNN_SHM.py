"""
Model Based Reinforcement Learning with Analytic Mechanic
Fei, Cheung

Learning Hamiltonian of Simple Harmonic Motion (SHM)
Hamiltonian is model by:
H = 1/2 p.T M(q) p + V(q)
Where we model M(q), V(q) by neural network

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

from module import StructuredHNN
from nn import MLP, PSD
from data import SHM_1D_HNN_DataModule
from utils import plot_HNN_1D_traj, plot_HNN_1D_vector_field, plot_HNN_1D_surface

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
    dataModule = SHM_1D_HNN_DataModule(init_conditions=init_conditions, batch_size=128)  # TODO: try different data set
    dataModule.setup()
    # dataModule.plot_data(train=True)

    # Total energy, later for generate the ground truth
    def SHM_hamiltonian(p, q, k=1., m=1.):
        return p ** 2. / (2. * m) + k * q ** 2. / 2.

    # Network for estimate the Hamiltonian
    M = PSD([1, 64, 1], activation='sigmoid')  # Mass matrix, M(q)
    V = MLP([1, 64, 1], activation='sigmoid')  # Potential V(q)
    HamFunc = StructuredHNN(massMatrixNet=M,
                            potentialNet=V,
                            dim=1)

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
            x, y, _ = batch
            y_hat = self.model.defunc(0, x)
            loss = self.loss_func(y, y_hat)
            self.logger.experiment.add_scalars("loss", {"train": loss}, self.current_epoch)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y, _ = batch
            y_hat = self.model.defunc(0, x)
            loss = self.loss_func(y, y_hat)
            self.log('val_loss', loss)  # For early stopping
            self.logger.experiment.add_scalars("loss", {"val": loss}, self.current_epoch)
            return loss

        def test_step(self, batch, batch_idx):
            x, y, _ = batch
            y_hat = self.model.defunc(0, x)
            loss = self.loss_func(y, y_hat)
            self.log('test_loss', loss)
            self.logger.experiment.add_scalars("loss", {"test": loss}, self.current_epoch)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.model.parameters(), lr=1e-4)


    # Plot the dynamic before we learn the model
    # plot_HNN_1D_traj(HamODE)
    # plot_HNN_1D_vector_field(HamODE)
    # plot_HNN_1D_surface(HamODE, ground_truth=SHM_hamiltonian)

    # Use pl for training
    learn = Learner(HamODE)
    early_stopping = EarlyStopping('val_loss', verbose=True, min_delta=1e-4, patience=20)
    trainer = pl.Trainer(callbacks=[early_stopping], min_epochs=1000, max_epochs=10000)
    trainer.fit(learn, dataModule)

    # TODO: Add animation on how the learned hamiltonian evolve
    # Test for the Learned model
    # plot_HNN_1D_traj(HamODE)
    plot_HNN_1D_vector_field(HamODE)
    plot_HNN_1D_surface(HamODE, ground_truth=SHM_hamiltonian)

    # TODO: Investigate the meaning of the learned M(q) and V(q)
    # log can be view using tensorboard
    # tensorboard --logdir ./experiment/exp_grad/lightning_logs
    print("done")
