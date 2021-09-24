"""
Model Based Reinforcement Learning with Analytic Mechanic
Fei, Cheung

Learning Control response and Hamiltonian from trajectory as ODE solving
SymODE -> HNN + Control + ODE learning
The loss function will depends on the time horizon
"""
import numpy as np
import matplotlib

import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from torchdyn.models import NeuralODE

from module import SymODE
from nn import MLP

from gym_env import CartPoleCustomEnv
from data import Gym_DataModule
from utils import plot_2D_vector_field

# matplotlib.use('macosx')  # Disable PyCharm SciView, Can turn on for interactive mode

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

if __name__ == "__main__":
    env = CartPoleCustomEnv(full_phase=True)
    controls = np.array([[-2.], [-1.], [0.], [1.], [2.]])
    # env, time step, traj, control
    dataModule = Gym_DataModule(env, 45, 50, controls, time_horizon=5)  # Same setting as SymODE paper
    dataModule.setup()

    # Network for estimate the ODE
    # input dim: (q, qdot)
    hamiltonian = MLP([2, 64, 1], activation='relu')
    control = MLP([1, 64, 1], activation='relu')
    SymFunc = SymODE(hamiltonian, control, dim=1)

    SymODEModel = NeuralODE(SymFunc, sensitivity='adjoint', solver='rk4').to(device)

    class Learner(pl.LightningModule):
        def __init__(self, model: torch.nn.Module):
            super().__init__()
            self.model = model
            self.loss_func = F.mse_loss

        def forward(self, x0, t_span):
            # x0 should be in (bs, [q, dq, u])
            return self.model.trajectory(x0, t_span)

        def training_step(self, batch, batch_idx):
            # (x1x2dq, control, time, info)
            # x.shape: (bs, time, dim)
            # t.shape: (bs, time, dim)

            _, u, t, x = batch
            t_span = t[0, :, 0]
            q_dq_u = torch.cat([x[:, :, 0:2], u], dim=2)  # Pick (q, dq, u)

            est = self.model.trajectory(q_dq_u[:, 0, :], t_span)

            l2_reg_G_net = sum(p.pow(2.0).sum() for p in self.model.defunc.m.G.parameters())
            loss = self.loss_func(q_dq_u, est.transpose(0, 1)) + l2_reg_G_net

            self.logger.experiment.add_scalars("loss", {"train": loss}, self.current_epoch)
            return loss

        def validation_step(self, batch, batch_idx):
            _, u, t, x = batch
            t_span = t[0, :, 0]
            q_dq_u = torch.cat([x[:, :, 0:2], u], dim=2)  # Pick (q, dq, u)

            est = self.model.trajectory(q_dq_u[:, 0, :], t_span)

            l2_reg_G_net = sum(p.pow(2.0).sum() for p in self.model.defunc.m.G.parameters())
            loss = self.loss_func(q_dq_u, est.transpose(0, 1)) + l2_reg_G_net

            self.log("val_loss", loss)
            self.logger.experiment.add_scalars("loss", {"train": loss}, self.current_epoch)

        def configure_optimizers(self):
            return torch.optim.Adam(self.model.parameters(), lr=1e-4)


    # Plot the dynamic before we learn the model
    # plot_2D_vector_field(HamODE)

    # Use pl for training
    learn = Learner(SymODEModel)
    early_stopping = EarlyStopping('val_loss', verbose=True, min_delta=1e-4, patience=20)
    trainer = pl.Trainer(callbacks=[early_stopping], min_epochs=1000, max_epochs=5000)  # tqdm bug mess with PyCharm
    trainer.fit(learn, dataModule)

    # Test for the Learned model
    # plot_2D_vector_field(HamODE)

    # log can be view using tensorboard
    # tensorboard --logdir ./experiment/exp_grad/lightning_logs
    print("done")
