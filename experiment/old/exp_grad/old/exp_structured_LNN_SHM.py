"""
Model Based Reinforcement Learning with Analytic Mechanic
Fei, Cheung

Learning Lagrangian of Simple Harmonic Motion (SHM)
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

from module import StructuredLNN
from nn import MLP, PSD
from data import SHM_1D_LNN_DataModule
from utils import plot_LNN_1D_traj, plot_LNN_1D_vector_field, plot_LNN_1D_surface

# matplotlib.use('macosx')  # Disable PyCharm SciView, Can turn on for interactive mode

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

if __name__ == "__main__":
    # Generate data for SHM motion
    init_conditions = [
        (1, 0, 0, 2. * np.pi, 100),  # TODO: LNN need more trajectory
        (2, 0, 0, 2. * np.pi, 100),
        (0.5, 0, 0, 2. * np.pi, 100),
        (1.5, 0, 0, 2. * np.pi, 100)
    ]
    dataModule = SHM_1D_LNN_DataModule(init_conditions=init_conditions, batch_size=100)  # TODO: try different data set
    dataModule.setup()

    # lagrangian, later for generate the ground truth
    def SHM_lagrangian(q, dq, k=1., m=1.):
        return m * dq ** 2. / 2. - k * q ** 2. / 2.

    # Network for estimate the Hamiltonian
    # TODO: Interesting that 1 layer almost never work
    # TODO: Increasing complexity increase sample efficiency
    M = PSD([1, 64, 1], activation='sigmoid')  # Mass matrix, M(q)
    V = MLP([1, 64, 1], activation='sigmoid')  # Potential V(q)
    LagFunc = StructuredLNN(massMatrixNet=M,
                            potentialNet=V,
                            dim=1)
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
            # TODO: Investigate different behaviour of getting dqdt. For example Finite difference, our from observation
            x, y, _ = batch
            y_hat = self.model.defunc(0, x)
            loss = self.loss_func(y, y_hat[:, 1])
            self.logger.experiment.add_scalars("loss", {"train": loss}, self.current_epoch)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y, _ = batch
            y_hat = self.model.defunc(0, x)
            loss = self.loss_func(y, y_hat[:, 1])
            self.log('val_loss', loss)
            self.logger.experiment.add_scalars("loss", {"val": loss}, self.current_epoch)
            return loss

        def test_step(self, batch, batch_idx):
            x, y, _ = batch
            y_hat = self.model.defunc(0, x)
            loss = self.loss_func(y, y_hat[:, 1])
            self.logger.experiment.add_scalars("loss", {"test": loss}, self.current_epoch)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.model.parameters(), lr=1e-4)


    # Plot the dynamic before we learn the model
    # plot_LNN_1D_traj(LagODE)
    # plot_LNN_1D_vector_field(LagODE)
    # plot_LNN_1D_surface(LagODE, ground_truth=SHM_lagrangian)

    # Use pl for training
    learn = Learner(LagODE)
    early_stopping = EarlyStopping('val_loss', verbose=True, min_delta=1e-4, patience=20)
    trainer = pl.Trainer(callbacks=[early_stopping], min_epochs=500, max_epochs=5000)
    trainer.fit(learn, dataModule)

    # TODO: Add animation on how the learned hamiltonian evolve
    # Test for the Learned model
    # plot_LNN_1D_traj(LagODE)
    plot_LNN_1D_vector_field(LagODE)
    plot_LNN_1D_surface(LagODE, ground_truth=SHM_lagrangian)

    # log can be view using tensorboard
    # tensorboard --logdir ./experiment/exp_grad/lightning_logs
    print("done")
