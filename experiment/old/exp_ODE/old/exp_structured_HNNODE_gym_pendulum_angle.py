"""
Model Based Reinforcement Learning with Analytic Mechanic
Fei, Cheung

Learning Structured Hamiltonian directly from trajectory
The loss function will depends on the time horizon
TODO: Training is super slow
TODO: https://www.quora.com/Is-it-possible-using-Lagrangian-mechanics-in-cartesian-coordinates-to-find-the-kinematic-equations-for-a-pendulum-simply-by-observing-that-if-the-mass-is-attached-to-a-rigid-rod-that-y-sqrt-r-2-x-2
Check this for using different data
if potential is not velocity depends, the multiplier depends only on ddq, q.
We can actually try extract the information from it
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

from gym_env import CartPoleCustomEnv
from data import Gym_DataModule
from utils import plot_HNN_1D_vector_field, plot_HNN_1D_surface

# matplotlib.use('macosx')  # Disable PyCharm SciView, Can turn on for interactive mode

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

if __name__ == "__main__":
    env = CartPoleCustomEnv()
    controls = np.array([[0.]])  # (bs, dim)

    # env, time step, traj, control
    dataModule = Gym_DataModule(env, 100, 100, controls, time_horizon=3)
    dataModule.setup()

    # Network for estimate the ODE
    M = PSD([1, 64, 1], activation='relu')  # Mass matrix, M(q)
    V = MLP([1, 64, 1], activation='relu')  # Potential V(q)
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
            # TODO: Try weighted error. Larger weight for point near x0

        def forward(self, x):
            return self.model.defunc(0, x)

        def training_step(self, batch, batch_idx):
            _, _, t, x = batch  # obs from env, control, time, extra Info

            xs0 = x[:, 0, :]
            t_span = t[0, :, 0]  # Assume all trajectory have the same time stamps, which only work for model.trajectory

            # take X (bs, dim), and t (number of time steps)
            cur_x_hat = self.model.trajectory(xs0, t_span)
            # return shape (time, batch, dim)
            loss = self.loss_func(x, cur_x_hat.transpose(0, 1))
            self.logger.experiment.add_scalars("loss", {"train": loss}, self.current_epoch)
            return loss

        def validation_step(self, batch, batch_idx):
            _, _, t, x = batch

            xs0 = x[:, 0, :]
            t_span = t[0, :, 0]

            cur_x_hat = self.model.trajectory(xs0, t_span)
            loss = self.loss_func(x, cur_x_hat.transpose(0, 1))
            self.log('val_loss', loss)
            self.logger.experiment.add_scalars("loss", {"val": loss}, self.current_epoch)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def SHM_hamiltonian(q, p, k=1., m=1.):
        return p ** 2. / (2. * m) + k * q ** 2. / 2.

    def pendulum_hamiltonian(q, p, l=1., m=1.):
        return p ** 2. / (2. * m * l) + m * 10. * l * (1 - np.cos(q))

    # Use pl for training
    learn = Learner(HamODE)
    # need longer patience for trajectory loss
    early_stopping = EarlyStopping('val_loss', verbose=True, min_delta=1e-5, patience=50)
    trainer = pl.Trainer(callbacks=[early_stopping], min_epochs=1000, max_epochs=5000)  # tqdm bug mess with PyCharm
    trainer.fit(learn, dataModule)

    # Test for the Learned model
    plot_HNN_1D_vector_field(HamODE)
    plot_HNN_1D_surface(HamODE, ground_truth=SHM_hamiltonian)

    # log can be view using tensorboard
    # tensorboard --logdir ./experiment/exp_grad/lightning_logs
    print("done")
