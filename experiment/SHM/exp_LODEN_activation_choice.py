import numpy as np
import torch

import matplotlib.pyplot as plt

import sys
sys.path.append("../../")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

min_delta = 0
patience = 30

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

from module import LNN, HNN, LNNODE
from nn import MLP, PSD

models = []
activations = ['tanh', 'sigmoid', 'relu', 'softplus']
for activation in activations:
    mass = PSD([1, 32, 16, 1], activation=activation)
    potential = MLP([1, 32, 16, 1], activation=activation)

    lodenFunc = LNNODE(num_angle=0, massMatrixNet=mass, potentialNet=potential)
    loden = NeuralODE(lodenFunc, sensitivity='adjoint', solver='rk4').to(device)

    models.append(loden)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from Learner import ODELearner

dataModule = DataModule(training_set, 300)
dataModule.setup()

for model in models:
    learner = ODELearner(model, 7)
    early_stopping = EarlyStopping('val_loss', verbose=False, min_delta=min_delta, patience=patience)
    trainer = pl.Trainer(callbacks=[early_stopping], min_epochs=100, max_epochs=10000)
    trainer.fit(learner, dataModule)

torch.save(models, "model/loden_activation.pt")