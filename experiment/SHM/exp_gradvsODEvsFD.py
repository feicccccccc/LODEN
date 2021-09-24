import numpy as np
import torch
import copy

import matplotlib.pyplot as plt

import sys

sys.path.append("../../")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

min_delta = 0
patience = 20

training_set = torch.load('data/SHM_training_set.pt')

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


from torchdyn.models import NeuralODE

from module import LNN, HNN, LNNODE, Baseline
from nn import MLP, PSD

baseline = Baseline(num_angle=0, num_raw=1, ODENet=MLP([2, 32, 32, 2], activation='softplus'))

baselineODE = NeuralODE(copy.deepcopy(baseline), sensitivity='adjoint', solver='rk4').to(device)
baselineODE_grad = NeuralODE(copy.deepcopy(baseline), sensitivity='adjoint', solver='rk4').to(device)
baselineODE_fd = NeuralODE(copy.deepcopy(baseline), sensitivity='adjoint', solver='rk4').to(device)

hamFunc = HNN(num_angle=0, hamiltonianNet=MLP([2, 32, 33, 1], activation='softplus'))
hamODE = NeuralODE(copy.deepcopy(hamFunc), sensitivity='adjoint', solver='rk4').to(device)
hamODE_grad = NeuralODE(copy.deepcopy(hamFunc), sensitivity='adjoint', solver='rk4').to(device)
hamODE_fd = NeuralODE(copy.deepcopy(hamFunc), sensitivity='adjoint', solver='rk4').to(device)

lagFunc = LNN(num_angle=0, lagrangianNet=MLP([2, 32, 33, 1], activation='softplus'))
lagODE = NeuralODE(copy.deepcopy(lagFunc), sensitivity='adjoint', solver='rk4').to(device)
lagODE_grad = NeuralODE(copy.deepcopy(lagFunc), sensitivity='adjoint', solver='rk4').to(device)
lagODE_fd = NeuralODE(copy.deepcopy(lagFunc), sensitivity='adjoint', solver='rk4').to(device)

mass = PSD([1, 32, 16, 1], activation='softplus')
potential = MLP([1, 32, 16, 1], activation='softplus')
symFunc = LNNODE(num_angle=0, massMatrixNet=mass, potentialNet=potential)

symODE = NeuralODE(copy.deepcopy(symFunc), sensitivity='adjoint', solver='rk4').to(device)
symODE_grad = NeuralODE(copy.deepcopy(symFunc), sensitivity='adjoint', solver='rk4').to(device)
symODE_fd = NeuralODE(copy.deepcopy(symFunc), sensitivity='adjoint', solver='rk4').to(device)

models = [hamODE, lagODE, symODE, baselineODE,
          hamODE_grad, lagODE_grad, symODE_grad, baselineODE_grad,
          hamODE_fd, lagODE_fd, symODE_fd, baselineODE_fd]

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from Learner import ODELearner, HNNLearner, LNNLearner, LNNFDLearner, HNNFDLearner

dataModule = DataModule(training_set, 100)
dataModule.setup()

hamODELearner = ODELearner(models[0], 7)
lagODELearner = ODELearner(models[1], 7)
symODELearner = ODELearner(models[2], 7)
baselineODELearner = ODELearner(models[3], 7)

hamgradLearner = HNNLearner(models[4])
laggradLearner = LNNLearner(models[5])
symgradLearner = LNNLearner(models[6])
baselinegradLearner = LNNLearner(models[7])

hamfdLearner = HNNFDLearner(models[8])
lagfdLearner = LNNFDLearner(models[9])
symfdLearner = LNNFDLearner(models[10])
baselinefdLearner = LNNFDLearner(models[11])

learners = [hamODELearner, lagODELearner, symODELearner, baselineODELearner,
            hamgradLearner, laggradLearner, symgradLearner, baselinegradLearner,
            hamfdLearner, lagfdLearner, symfdLearner, baselinefdLearner]

import time

for learner in learners:
    early_stopping = EarlyStopping('val_loss', verbose=False, min_delta=min_delta, patience=patience)
    trainer = pl.Trainer(callbacks=[early_stopping], min_epochs=100, max_epochs=10000)

    start_time = time.time()
    trainer.fit(learner, dataModule)
    print("--- %s seconds ---" % (time.time() - start_time))

torch.save(models, "model/gradvsODEvsFD.pt")