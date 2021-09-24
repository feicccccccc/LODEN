import numpy as np
import torch

import matplotlib.pyplot as plt

import sys
sys.path.append("../../")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(10)

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

from module import LNN, HNN, LNNODE
from nn import MLP, PSD


def create_models():
    baseline = MLP([2, 32, 32, 2], activation='tanh')

    baselinefunc = MLP([2, 32, 32, 2], activation='tanh')
    baselineODE = NeuralODE(baselinefunc, sensitivity='adjoint', solver='rk4').to(device)

    hamFunc2 = HNN(num_angle=0, hamiltonianNet=MLP([2, 32, 33, 1], activation='tanh'))
    hamGrad = NeuralODE(hamFunc2, sensitivity='adjoint', solver='rk4').to(device)

    lagFunc2 = LNN(num_angle=0, lagrangianNet=MLP([2, 32, 33, 1], activation='softplus'))
    lagGrad = NeuralODE(lagFunc2, sensitivity='adjoint', solver='rk4').to(device)

    hamFunc = HNN(num_angle=0, hamiltonianNet=MLP([2, 32, 33, 1], activation='tanh'))
    hamODE = NeuralODE(hamFunc, sensitivity='adjoint', solver='rk4').to(device)

    lagFunc = LNN(num_angle=0, lagrangianNet=MLP([2, 32, 33, 1], activation='softplus'))
    lagODE = NeuralODE(lagFunc, sensitivity='adjoint', solver='rk4').to(device)

    massMatrixNet = PSD([1, 32, 16, 1], activation='tanh')
    potentialNet = MLP([1, 32, 16, 1], activation='tanh')

    symFunc = LNNODE(num_angle=0, massMatrixNet=massMatrixNet, potentialNet=potentialNet)
    symGrad = NeuralODE(symFunc, sensitivity='adjoint', solver='rk4').to(device)

    massMatrixNet2 = PSD([1, 32, 16, 1], activation='tanh')
    potentialNet2 = MLP([1, 32, 16, 1], activation='tanh')

    symFunc2 = LNNODE(num_angle=0, massMatrixNet=massMatrixNet2, potentialNet=potentialNet2)
    symODE = NeuralODE(symFunc2, sensitivity='adjoint', solver='rk4').to(device)

    models = [baseline, baselineODE, hamGrad, lagGrad, hamODE, lagODE, symGrad, symODE]
    return models

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from Learner import HNNLearner, LNNLearner, ODELearner, BaselineLearner

def create_learner(models):
    baselinelearn = BaselineLearner(models[0])
    baselineODElearn = ODELearner(models[1], 7)

    HNNgradlearn = HNNLearner(models[2])
    LNNgradlearn = LNNLearner(models[3])

    HNNODElearn = ODELearner(models[4], 7)
    LNNODElearn = ODELearner(models[5], 7)

    symGradlearn = LNNLearner(models[6])
    symODElearn = ODELearner(models[7], 7)

    # all_learners = [baselinelearn, baselineODElearn,
    #                 HNNgradlearn, LNNgradlearn,
    #                 HNNODElearn, LNNODElearn,
    #                 symGradlearn, symODElearn]
    all_learners = [baselinelearn, baselineODElearn,
                    HNNgradlearn, LNNgradlearn,
                    HNNODElearn, LNNODElearn,
                    symGradlearn, symODElearn]
    return all_learners

samples = [10, 20, 30, 40, 50, 100, 150, 200, 250, 300, 500, 700, 1000]

for run in range(10):
    print("\n||||||| Current Run {} |||||||\n".format(run))
    for num_sample in samples:
        print("\n===== Current Sample {} =====\n".format(num_sample))
        dataModule = DataModule(training_set, num_sample)
        dataModule.setup()
        i = 0

        all_model = create_models()
        all_learner = create_learner(all_model)

        for learner in all_learner:
            print("\n--- Current Model:{} ---\n".format(i))
            early_stopping = EarlyStopping('val_loss', verbose=False, min_delta=min_delta, patience=patience)
            trainer = pl.Trainer(callbacks=[early_stopping], min_epochs=100, max_epochs=10000)
            try:
                trainer.fit(learner, dataModule)
            except:
                print("===================================================")
                print("====== Run {}, Sample {}, Model {} can't fit ======".format(run, num_sample, i))
                print("===================================================")
            i += 1

        torch.save(all_model, "model/model3_5_{}sample_{}run.pt".format(num_sample, run))
