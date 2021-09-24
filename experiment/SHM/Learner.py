"""
Model Based Reinforcement Learning with Analytic Mechanic
Fei, Cheung

Learner Class for Different Method (ODE, grad, FD)
"""
import torch
import torch.nn.functional as F
import pytorch_lightning as pl


class HNNLearner(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, learning_rate=3e-4):
        super().__init__()
        self.model = model
        self.loss_func = F.mse_loss
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model.defunc(0, x)

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.logger.experiment.add_scalars("loss", {"train": loss}, self.current_epoch)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('val_loss', loss)
        self.logger.experiment.add_scalars("loss", {"val": loss}, self.current_epoch)
        return loss

    def step(self, batch, batch_idx):
        q_p, dq_dp, ddq, t = [x.reshape(-1, x.shape[2]) for x in batch]
        est_dq_dp = self.model.defunc(0, q_p)
        loss = self.loss_func(est_dq_dp, dq_dp)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)


class HNNFDLearner(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, learning_rate=3e-4):
        super().__init__()
        self.model = model
        self.loss_func = F.mse_loss
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model.defunc(0, x)

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.logger.experiment.add_scalars("loss", {"train": loss}, self.current_epoch)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('val_loss', loss)
        self.logger.experiment.add_scalars("loss", {"val": loss}, self.current_epoch)
        return loss

    def step(self, batch, batch_idx):
        q_p, dq_dp, ddq, t = batch
        dim = q_p.shape[2]
        dq_dp_FD = (q_p[:, 1:, :] - q_p[:, :-1, :]) / (t[:, 1:, :] - t[:, :-1, :])
        est_dq_dp = self.model.defunc(0, dq_dp_FD.reshape(-1, dim))
        loss = self.loss_func(est_dq_dp, dq_dp[:, :-1, :].reshape(-1, dim))
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)


class LNNLearner(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, learning_rate=3e-4):
        super().__init__()
        self.model = model
        self.loss_func = F.mse_loss
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model.defunc(0, x)

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.logger.experiment.add_scalars("loss", {"train": loss}, self.current_epoch)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('val_loss', loss)
        self.logger.experiment.add_scalars("loss", {"val": loss}, self.current_epoch)
        return loss

    def step(self, batch, batch_idx):
        q_dq, dq_dp, ddq, t = [x.reshape(-1, x.shape[2]) for x in batch]
        dq_ddq = torch.stack([q_dq[:, 1], ddq[:, 0]], dim=1)
        est_dq_ddp = self.model.defunc(0, q_dq)
        loss = self.loss_func(est_dq_ddp, dq_ddq)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)


class LNNFDLearner(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, learning_rate=3e-4):
        super().__init__()
        self.model = model
        self.loss_func = F.mse_loss
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model.defunc(0, x)

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.logger.experiment.add_scalars("loss", {"train": loss}, self.current_epoch)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('val_loss', loss)
        self.logger.experiment.add_scalars("loss", {"val": loss}, self.current_epoch)
        return loss

    def step(self, batch, batch_idx):
        q_dq, dq_dp, ddq, t = batch
        dim = q_dq.shape[2]
        # Forward Finite Difference
        ddq_FD = (q_dq[:, 1:, 1] - q_dq[:, :-1, 1]) / (t[:, 1:, 0] - t[:, :-1, 0])
        dq_ddq_FD = torch.stack([q_dq[:, :-1, 1], ddq_FD], dim=2).reshape(-1, dim)
        est_dq_ddp = self.model.defunc(0, q_dq[:, :-1, :].reshape(-1, dim))
        loss = self.loss_func(est_dq_ddp, dq_ddq_FD)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)


class ODELearner(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, time_horizon: int, learning_rate=3e-4):
        super().__init__()
        self.model = model
        self.loss_func = F.mse_loss
        self.learning_rate = learning_rate
        self.time_horizon = time_horizon

    def forward(self, x):
        return self.model.defunc(0, x)

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.logger.experiment.add_scalars("loss", {"train": loss}, self.current_epoch)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('val_loss', loss)
        self.logger.experiment.add_scalars("loss", {"val": loss}, self.current_epoch)
        return loss

    def step(self, batch, batch_idx):
        q_dq, _, _, t = batch
        t_span = t[0, :self.time_horizon, 0]

        # Generate sub data set
        assert q_dq.shape[1] >= self.time_horizon, \
            "Trajectory not long enough to split accordingly, " \
            "try reducing the time horizon parameters"

        num_of_trajs = q_dq.shape[1] - self.time_horizon + 1
        X = []
        for i in range(num_of_trajs):
            X.append(q_dq[:, i:i + self.time_horizon, :])
        X = torch.cat(X, dim=0)

        est_X = self.model.trajectory(X[:, 0, :], t_span).transpose(0, 1)
        loss = self.loss_func(est_X, X)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)


class BaselineLearner(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, learning_rate=3e-4):
        super().__init__()
        self.model = model
        self.loss_func = F.mse_loss
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model.defunc(0, x)

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.logger.experiment.add_scalars("loss", {"train": loss}, self.current_epoch)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('val_loss', loss)
        self.logger.experiment.add_scalars("loss", {"val": loss}, self.current_epoch)
        return loss

    def step(self, batch, batch_idx):
        q_dq, _, _, t = batch

        X = []
        X.append(q_dq[:, 0, :])
        for i in range(1, t.shape[1]):
            next_q_dq = self.model.forward(X[-1])
            X.append(next_q_dq)

        est_q_dq = torch.stack(X, dim=1)
        loss = self.loss_func(est_q_dq, q_dq)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)


if __name__ == "__main__":
    # Test script
    from torchdyn.models import NeuralODE

    from module import LNN, HNN, LNNODE, Baseline
    from nn import MLP, PSD

    import numpy as np

    device = 'cpu'

    baseline = MLP([2, 64, 2], activation='tanh')

    baselinefunc = MLP([2, 64, 2], activation='tanh')
    baselineODE = NeuralODE(baselinefunc, sensitivity='adjoint', solver='rk4').to(device)

    lagFunc = LNN(num_angle=0, lagrangianNet=MLP([2, 80, 1], activation='softplus'))
    lagODE = NeuralODE(lagFunc, sensitivity='adjoint', solver='rk4').to(device)

    hamFunc = HNN(num_angle=0, hamiltonianNet=MLP([2, 80, 1], activation='tanh'))
    hamODE = NeuralODE(hamFunc, sensitivity='adjoint', solver='rk4').to(device)

    massMatrixNet = PSD([1, 64, 1], activation='tanh')
    potentialNet = MLP([1, 64, 1], activation='tanh')

    symFunc = LNNODE(num_angle=0, massMatrixNet=massMatrixNet, potentialNet=potentialNet)
    symODE = NeuralODE(lagFunc, sensitivity='adjoint', solver='rk4').to(device)

    test_models = [baseline, baselineODE, lagODE, hamODE, symODE]

    from torch.utils.data import DataLoader, random_split, TensorDataset
    import pytorch_lightning as pl


    class DataModule(pl.LightningDataModule):
        def __init__(self, dataset: torch.Tensor, num_sample: int, batch_size=128):
            super().__init__()
            self.num_sample = num_sample
            self.dataset = dataset[:num_sample, :, :]
            self.batch_size = 128

            self.train_set = None
            self.val_set = None

        def setup(self):
            q_p, dq_dp, ddq, t = torch.split(self.dataset, [2, 2, 1, 1], dim=2)
            dataset = TensorDataset(q_p, dq_dp, ddq, t)

            num_val = int(np.ceil(self.num_sample * 0.2))
            num_train = self.num_sample - num_val
            self.train_set, self.val_set = random_split(dataset, [num_train, num_val])

        def train_dataloader(self):
            return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

        def val_dataloader(self):
            return DataLoader(self.val_set, batch_size=self.batch_size)


    training_set = torch.load('data/SHM_training_set.pt')
    dataModule = DataModule(training_set, 100)
    dataModule.setup()

    baseline = Baseline(num_angle=0, num_raw=1, ODENet=MLP([2, 32, 33, 2], activation='softplus'))
    baseline_test = NeuralODE(baseline, sensitivity='adjoint', solver='rk4').to(device)

    # learn = HNNFDLearner(hamODE)
    # learn = LNNFDLearner(lagODE)
    # learn = ODELearner(lagODE, 7)
    learn = LNNLearner(baseline_test)
    trainer = pl.Trainer(max_epochs=20)
    trainer.fit(learn, dataModule)
