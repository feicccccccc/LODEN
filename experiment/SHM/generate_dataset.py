import numpy as np
import torch

import sys
from utils import SHM_1D_model

sys.path.append("../../")

k = 1.
m = 1.

dt = 0.05
steps = 500

num_test_pt = 100
test_pt_upper = np.array([6, 6])
test_pt_lower = -test_pt_upper

test_set = []
for i in range(num_test_pt):
    init_condition = np.random.uniform(test_pt_lower, test_pt_upper)
    current_set = SHM_1D_model(x0=init_condition[0],
                               p0=init_condition[1],
                               t0=0.,
                               t1=(steps - 1) * dt,
                               steps=steps,
                               k=k,
                               m=m)
    test_set.append(torch.cat(current_set, dim=1))

test_set = torch.stack(test_set, dim=0)

# torch.save(test_set, 'data/SHM_test_set.pt')

k = 1.
m = 1.

dt = 0.05
steps = 145

num_training_pt = 20
data_pt_upper = np.array([2, 2])
data_pt_lower = -test_pt_upper

data_set = []
for i in range(num_training_pt):
    init_condition = np.random.uniform(test_pt_lower, test_pt_upper)
    current_set = SHM_1D_model(x0=init_condition[0],
                               p0=init_condition[1],
                               t0=0.,
                               t1=(steps-1) * dt,
                               steps=steps,
                               k=k,
                               m=m)
    data_set.append(torch.cat(current_set, dim=1))

data_set = torch.stack(data_set, dim=0)

torch.save(data_set, 'data/SHM_training_set_full.pt')
