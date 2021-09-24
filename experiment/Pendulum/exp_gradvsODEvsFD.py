import numpy as np
import torch
import copy

import matplotlib.pyplot as plt

import sys

sys.path.append("../../")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

min_delta = 0
patience = 20

training_set = torch.load('data/Pendulum_training_set_new.pt')

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


from torchdyn.models import NeuralODE

from module import LNN, HNN, LNNODE, Baseline
from nn import MLP, PSD, MatrixNet

cNet = MatrixNet([2, 32, 32, 1], control_dim=1, activation='relu')

baseline = Baseline(num_angle=1, num_raw=0, num_control=1,
                    ODENet=MLP([3, 64, 63, 2], activation='softplus'), controlNet=copy.deepcopy(cNet))

baselineODE = NeuralODE(copy.deepcopy(baseline), sensitivity='adjoint', solver='rk4').to(device)
baselineODE_grad = NeuralODE(copy.deepcopy(baseline), sensitivity='adjoint', solver='rk4').to(device)
baselineODE_fd = NeuralODE(copy.deepcopy(baseline), sensitivity='adjoint', solver='rk4').to(device)

hamFunc = HNN(num_angle=1,
              hamiltonianNet=MLP([3, 64, 64, 1], activation='softplus'), controlNet=copy.deepcopy(cNet))

hamODE = NeuralODE(copy.deepcopy(hamFunc), sensitivity='adjoint', solver='rk4').to(device)
hamODE_grad = NeuralODE(copy.deepcopy(hamFunc), sensitivity='adjoint', solver='rk4').to(device)
hamODE_fd = NeuralODE(copy.deepcopy(hamFunc), sensitivity='adjoint', solver='rk4').to(device)

lagFunc = LNN(num_angle=1,
              lagrangianNet=MLP([3, 64, 64, 1], activation='softplus'), controlNet=copy.deepcopy(cNet))

lagODE = NeuralODE(copy.deepcopy(lagFunc), sensitivity='adjoint', solver='rk4').to(device)
lagODE_grad = NeuralODE(copy.deepcopy(lagFunc), sensitivity='adjoint', solver='rk4').to(device)
lagODE_fd = NeuralODE(copy.deepcopy(lagFunc), sensitivity='adjoint', solver='rk4').to(device)

mass = PSD([2, 62, 32, 1], activation='softplus')
potential = MLP([2, 62, 32, 1], activation='softplus')
symFunc = LNNODE(num_angle=1, massMatrixNet=mass, potentialNet=potential, controlNet=copy.deepcopy(cNet))

symODE = NeuralODE(copy.deepcopy(symFunc), sensitivity='adjoint', solver='rk4').to(device)
symODE_grad = NeuralODE(copy.deepcopy(symFunc), sensitivity='adjoint', solver='rk4').to(device)
symODE_fd = NeuralODE(copy.deepcopy(symFunc), sensitivity='adjoint', solver='rk4').to(device)

models = [hamODE, symODE, baselineODE,
          hamODE_grad, lagODE_grad, symODE_grad, baselineODE_grad,
          hamODE_fd, lagODE_fd, symODE_fd, baselineODE_fd]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

for model in models:
    print("{} have {} parameters".format(type(model), count_parameters(model)))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from Learner import ODELearner, HNNLearner, LNNLearner, LNNFDLearner, HNNFDLearner

dataModule = DataModule(training_set, 300)
dataModule.setup()

hamODELearner = ODELearner(models[0], 7)
# lagODELearner = ODELearner(models[1], 7)
symODELearner = ODELearner(models[1], 7)
baselineODELearner = ODELearner(models[2], 7)

hamgradLearner = HNNLearner(models[3])
laggradLearner = LNNLearner(models[4])
symgradLearner = LNNLearner(models[5])
baselinegradLearner = LNNLearner(models[6])

hamfdLearner = HNNFDLearner(models[7])
lagfdLearner = LNNFDLearner(models[8])
symfdLearner = LNNFDLearner(models[9])
baselinefdLearner = LNNFDLearner(models[10])

learners = [hamODELearner, symODELearner, baselineODELearner,
            hamgradLearner, laggradLearner, symgradLearner, baselinegradLearner,
            hamfdLearner, lagfdLearner, symfdLearner, baselinefdLearner]

import time

early_stopping = EarlyStopping('val_loss', verbose=False, min_delta=min_delta, patience=patience)
pre_trainer = pl.Trainer(max_epochs=20)
trainer = pl.Trainer(callbacks=[early_stopping], min_epochs=100, max_epochs=10000)

start_time = time.time()
pre_learner = ODELearner(lagODE, 2)
pre_trainer.fit(pre_learner, dataModule)

learner = ODELearner(lagODE, 7)
trainer.fit(learner, dataModule)
print("--- %s seconds ---" % (time.time() - start_time))

torch.save(lagODE, "model/gradvsODEvsFD_lagODE_new.pt")

for learner in learners:
    early_stopping = EarlyStopping('val_loss', verbose=False, min_delta=min_delta, patience=patience)
    trainer = pl.Trainer(callbacks=[early_stopping], min_epochs=100, max_epochs=10000)

    start_time = time.time()
    trainer.fit(learner, dataModule)
    print("--- %s seconds ---" % (time.time() - start_time))

torch.save(models, "model/gradvsODEvsFD_new.pt")