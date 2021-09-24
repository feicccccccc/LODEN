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
        q_p, u, t, info = batch

        q_p_u = torch.cat([q_p, u], dim=2)
        # (r, cos th, sin th, dr, dth, u)
        est_dq_dp = self.model.defunc(0, q_p_u.view(-1, 6))

        # Get true gradient
        dr = q_p[:, :, 3]
        dx1 = -q_p[:, :, 2] * q_p[:, :, 4]
        dx2 = q_p[:, :, 1] * q_p[:, :, 4]
        ddr = info[:, :, 1]
        ddth = info[:, :, 2]
        dq_dp = torch.stack([dr, dx1, dx2, ddr, ddth], dim=2)

        loss = self.loss_func(est_dq_dp[:, :-1], dq_dp.reshape(-1, 5))
        pole_loss = self.pole()

        total_loss = loss + 1e-3 * pole_loss
        return total_loss

    def pole(self):
        # last dimension is control
        poles = [[0., 1., 0., 0., 0., 0.],
                 [0., -1., 0., 0., 0., 0.],
                 [1., 1., 0., 0., 0., 0.],
                 [1., -1., 0., 0., 0., 0.],
                 [2., 1., 0., 0., 0., 0.],
                 [2., -1., 0., 0., 0., 0.],
                 [-1., 1., 0., 0., 0., 0.],
                 [-1., -1., 0., 0., 0., 0.],
                 [-2., 1., 0., 0., 0., 0.],
                 [-2., -1., 0., 0., 0., 0.],
                 ]
        poles = torch.tensor(poles).float()
        dq_ddq = torch.zeros_like(poles)
        est_dq_ddq = self.model.defunc(0, poles.view(-1, 6))
        pole_loss = self.loss_func(est_dq_ddq[:, :-1], dq_ddq[:, :-1])
        return pole_loss

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
        q_p, u, t, i = batch
        dim = q_p.shape[2]
        dq_dp_FD = (q_p[:, 1:, :] - q_p[:, :-1, :]) / (t[:, 1:, :] - t[:, :-1, :])
        q_p_u = torch.cat([q_p[:, :-1, :], u[:, :-1, :]], dim=2)
        est_dq_dp = self.model.defunc(0, q_p_u.reshape(-1, dim+1))

        loss = self.loss_func(est_dq_dp[:, :-1], dq_dp_FD.reshape(-1, dim))
        pole_loss = self.pole()

        total_loss = loss + 1e-3 * pole_loss
        return total_loss

    def pole(self):
        # last dimension is control
        poles = [[0., 1., 0., 0., 0., 0.],
                 [0., -1., 0., 0., 0., 0.],
                 [1., 1., 0., 0., 0., 0.],
                 [1., -1., 0., 0., 0., 0.],
                 [2., 1., 0., 0., 0., 0.],
                 [2., -1., 0., 0., 0., 0.],
                 [-1., 1., 0., 0., 0., 0.],
                 [-1., -1., 0., 0., 0., 0.],
                 [-2., 1., 0., 0., 0., 0.],
                 [-2., -1., 0., 0., 0., 0.],
                 ]
        poles = torch.tensor(poles).float()
        dq_ddq = torch.zeros_like(poles)
        est_dq_ddq = self.model.defunc(0, poles.view(-1, 6))
        pole_loss = self.loss_func(est_dq_ddq[:, :-1], dq_ddq[:, :-1])
        return pole_loss

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
        pole_loss = self.pole()

        total_loss = loss + 1e-3 * pole_loss
        return total_loss

    def pole(self):
        # last dimension is control
        poles = [[np.pi, 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.],
                 [1., np.pi, 0., 0., 0.],
                 [1., 0., 0., 0., 0.],
                 [2., np.pi, 0., 0., 0.],
                 [2., 0., 0., 0., 0.],
                 [-1., np.pi, 0., 0., 0.],
                 [-1., 0., 0., 0., 0.],
                 [-2., np.pi, 0., 0., 0.],
                 [-2., 0., 0., 0., 0.],
                 ]
        poles = torch.tensor(poles).float()
        dq_ddq = torch.zeros_like(poles)
        est_dq_ddq = self.model.defunc(0, poles.view(-1, 5))
        pole_loss = self.loss_func(est_dq_ddq[:, :-1], dq_ddq[:, :-1])
        return pole_loss

    def step(self, batch, batch_idx):
        q_dq, u, t, info = batch

        q_dq_u = torch.cat([q_dq, u], dim=2)
        # (r, cos th, sin th, dr, dth, u)
        est_dq_ddq = self.model.defunc(0, q_dq_u.view(-1, 5))

        # Get true gradient
        dr = q_dq[:, :, 3]
        dth = q_dq[:, :, 4]
        ddr = info[:, :, 1]
        ddth = info[:, :, 2]
        dq_ddq = torch.stack([dr, dth, ddr, ddth], dim=2)

        loss = self.loss_func(est_dq_ddq[:, :-1], dq_ddq.reshape(-1, 5))
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
        q_dq, u, t, i = batch
        dim = q_dq.shape[2]
        dq_ddq_FD = (q_dq[:, 1:, :] - q_dq[:, :-1, :]) / (t[:, 1:, :] - t[:, :-1, :])
        q_dq_u = torch.cat([q_dq[:, :-1, :], u[:, :-1, :]], dim=2)
        est_dq_ddq = self.model.defunc(0, q_dq_u.reshape(-1, dim + 1))

        loss = self.loss_func(est_dq_ddq[:, :-1], dq_ddq_FD.reshape(-1, dim))
        pole_loss = self.pole()

        total_loss = loss + 1e-3 * pole_loss
        return total_loss

    def pole(self):
        # last dimension is control
        poles = [[np.pi, 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.],
                 [1., np.pi, 0., 0., 0.],
                 [1., 0., 0., 0., 0.],
                 [2., np.pi, 0., 0., 0.],
                 [2., 0., 0., 0., 0.],
                 [-1., np.pi, 0., 0., 0.],
                 [-1., 0., 0., 0., 0.],
                 [-2., np.pi, 0., 0., 0.],
                 [-2., 0., 0., 0., 0.],
                 ]
        poles = torch.tensor(poles).float()
        dq_ddq = torch.zeros_like(poles)
        est_dq_ddq = self.model.defunc(0, poles.view(-1, 5))
        pole_loss = self.loss_func(est_dq_ddq[:, :-1], dq_ddq[:, :-1])
        return pole_loss

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
        q_dq, u, t, _ = batch
        t_span = t[0, :self.time_horizon, 0]

        # Generate sub data set
        assert q_dq.shape[1] >= self.time_horizon, \
            "Trajectory not long enough to split accordingly, " \
            "try reducing the time horizon parameters"

        num_of_trajs = q_dq.shape[1] - self.time_horizon + 1
        X = []
        q_dq_u = torch.cat([q_dq, u], dim=2)
        for i in range(num_of_trajs):
            X.append(q_dq_u[:, i:i + self.time_horizon, :])
        X = torch.cat(X, dim=0)

        est_X = self.model.trajectory(X[:, 0, :], t_span).transpose(0, 1)
        loss = self.loss_func(est_X[:, :, :-1], X[:, :, :-1])
        # pole_loss = self.pole()

        total_loss = loss #+ 1e-3 * pole_loss
        return total_loss

    def pole(self):
        # last dimension is control
        poles = [[np.pi, 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.],
                 [1., np.pi, 0., 0., 0.],
                 [1., 0., 0., 0., 0.],
                 [2., np.pi, 0., 0., 0.],
                 [2., 0., 0., 0., 0.],
                 [-1., np.pi, 0., 0., 0.],
                 [-1., 0., 0., 0., 0.],
                 [-2., np.pi, 0., 0., 0.],
                 [-2., 0., 0., 0., 0.],
                 ]
        poles = torch.tensor(poles).float()
        dq_ddq = torch.zeros_like(poles)
        est_dq_ddq = self.model.defunc(0, poles.view(-1, ))
        pole_loss = self.loss_func(est_dq_ddq[:, :-1], dq_ddq[:, :-1])
        return pole_loss

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
    import sys

    sys.path.append("../../")

    from torchdyn.models import NeuralODE

    from module import LNN, HNN, LNNODE, Baseline
    from nn import MLP, PSD, MatrixNet

    import copy
    import numpy as np

    device = 'cpu'
    torch.manual_seed(0)

    cNet = MatrixNet([3, 64, 64, 64, 2], control_dim=1, activation='tanh')

    # baseline = Baseline(num_angle=1, num_raw=1, num_control=1,
    #                     ODENet=MLP([5, 128, 62, 32, 4], activation='softplus'), controlNet=copy.deepcopy(cNet))
    #
    # baselineODE = NeuralODE(copy.deepcopy(baseline), sensitivity='adjoint', solver='rk4').to(device)
    # baselineODE_grad = NeuralODE(copy.deepcopy(baseline), sensitivity='adjoint', solver='rk4').to(device)
    # baselineODE_fd = NeuralODE(copy.deepcopy(baseline), sensitivity='adjoint', solver='rk4').to(device)

    # hamFunc = HNN(num_angle=1,
    #               hamiltonianNet=MLP([5, 128, 64, 32, 1], activation='softplus'), controlNet=copy.deepcopy(cNet))

    # hamODE = NeuralODE(copy.deepcopy(hamFunc), sensitivity='adjoint', solver='rk4').to(device)
    # hamODE_grad = NeuralODE(copy.deepcopy(hamFunc), sensitivity='adjoint', solver='rk4').to(device)
    # hamODE_fd = NeuralODE(copy.deepcopy(hamFunc), sensitivity='adjoint', solver='rk4').to(device)

    # lagFunc = LNN(num_angle=1,
    #               lagrangianNet=MLP([5, 512, 512, 512, 1], activation='softplus'), controlNet=copy.deepcopy(cNet))
    #
    # lagODE = NeuralODE(copy.deepcopy(lagFunc), sensitivity='adjoint', solver='rk4').to(device)
    # lagODE_grad = NeuralODE(copy.deepcopy(lagFunc), sensitivity='adjoint', solver='rk4').to(device)
    # lagODE_fd = NeuralODE(copy.deepcopy(lagFunc), sensitivity='adjoint', solver='rk4').to(device)

    mass = PSD([3, 128, 128, 128, 2], activation='sigmoid')
    potential = MLP([3, 128, 128, 128, 1], activation='softplus')
    symFunc = LNNODE(num_angle=1, massMatrixNet=mass, potentialNet=potential, controlNet=copy.deepcopy(cNet))

    symODE = NeuralODE(copy.deepcopy(symFunc), sensitivity='adjoint', solver='rk4').to(device)
    symODE_grad = NeuralODE(copy.deepcopy(symFunc), sensitivity='adjoint', solver='rk4').to(device)
    symODE_fd = NeuralODE(copy.deepcopy(symFunc), sensitivity='adjoint', solver='rk4').to(device)

    # models = [hamODE, lagODE, symODE, baselineODE,
    #           hamODE_grad, lagODE_grad, symODE_grad, baselineODE_grad,
    #           hamODE_fd, lagODE_fd, symODE_fd, baselineODE_fd]


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # for model in models:
    #     print("{} have {} parameters".format(type(model), count_parameters(model)))

    from pytorch_lightning.callbacks import EarlyStopping
    from torch.utils.data import DataLoader, random_split, TensorDataset
    import pytorch_lightning as pl

    class DataModule(pl.LightningDataModule):
        def __init__(self, dataset: TensorDataset, num_sample: int, batch_size=128):
            super().__init__()
            self.num_sample = num_sample
            self.dataset = dataset
            self.batch_size = batch_size

            self.train_set = None
            self.val_set = None

        def setup(self):
            X, U, T, I = self.dataset.tensors
            dataset = TensorDataset(X[:self.num_sample], U[:self.num_sample], T[:self.num_sample], I[:self.num_sample])
            num_val = int(np.ceil(self.num_sample * 0.2))
            num_train = self.num_sample - num_val
            self.train_set, self.val_set = random_split(dataset, [num_train, num_val])

        def train_dataloader(self):
            return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

        def val_dataloader(self):
            return DataLoader(self.val_set, batch_size=self.batch_size)


    training_set = torch.load('data/Cartpole_training_set_20k_test.pt')
    dataModule = DataModule(training_set, 10000, batch_size=128)
    dataModule.setup()

    import time
    pre_trainer = pl.Trainer(max_epochs=20)
    early_stopping = EarlyStopping('val_loss', verbose=False, min_delta=0, patience=30)
    trainer = pl.Trainer(callbacks=[early_stopping], min_epochs=100, max_epochs=10000)

    test_model = symODE
    # pre_learner = LNNFDLearner(test_model)
    # pre_learner = ODELearner(lagODE, 2)
    test_learner = ODELearner(test_model, 7)

    start_time = time.time()
    # pre_trainer.fit(pre_learner, dataModule)
    trainer.fit(test_learner, dataModule)
    print("--- %s seconds ---" % (time.time() - start_time))

    torch.save(test_model, "model/symODE3_test_10k.pt")