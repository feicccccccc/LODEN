"""
Test training with env and perform test
"""

import numpy as np
import torch
from torchdyn.models import NeuralODE
import matplotlib.pyplot as plt

import sys
sys.path.append('/Users/fei/Documents/UCL/Thesis/Code/MBRL_HNN/ilqr')
sys.path.append('/Users/fei/Documents/UCL/Thesis/Code/MBRL_HNN')

from cost import QRCost
from cartpole import GeneralDyanmic, CartPoleDynamic
from dynamic import constrain, tensor_constrain
from controller import iLQR

from gym_env import CartPoleCustomEnv
from gym.wrappers import Monitor

np.set_printoptions(suppress=True)
np.random.seed(0)
device = 'cpu'

models = torch.load("../Cartpole/model/gradvsODEvsFD.pt")
lagODE_model = torch.load("../Cartpole/model/gradvsODEvsFD_lagODE.pt")
models.insert(1, lagODE_model)

test_model = torch.load("../Cartpole/model/symODE3_test.pt")

for model in models:
    model.eval()

test_model = test_model

dynamics = GeneralDyanmic(0.02, test_model, state_size=5, action_size=1)

# env init
env = Monitor(CartPoleCustomEnv(), './gym_data/lagODE_cartpole', force=True)
init_state = np.array([0, 0, np.pi, 0])
# init_state = np.array([0.05, 0.05, 0.05, 0.05])
obs0 = env.reset(init_state)


env.seed(0)

# Controller init
J_hist = []


def on_iteration(iteration_count, xs, us, J_opt, accepted, converged):
    J_hist.append(J_opt)
    info = "converged" if converged else ("accepted" if accepted else "failed")
    # if converged or accepted:
    final_state = CartPoleDynamic.reduce_state(xs[-1])
    print("iter", iteration_count, info, J_opt, final_state, constrain(us[0:2], -10, 10)[:, 0])


dt = 0.02

# Note that the augmented state is not all 0.
x_goal = CartPoleDynamic.augment_state(np.array([0., 0., 0., 0.]))
Q = np.eye(dynamics.state_size)
Q[0, 0] = 1.0
Q[1, 1] = 0.0
Q[3, 3] = 0.0
Q[4, 4] = 0.0
Q_terminal = 100. * np.eye(dynamics.state_size)
R = np.array([[0.01]])
cost = QRCost(Q, R, Q_terminal=Q_terminal, x_goal=x_goal)

N = 80
x0 = CartPoleDynamic.augment_state(init_state)
np.random.seed(0)
us_init = np.random.uniform(-1, 1, (N, dynamics.action_size))
controller = iLQR(dynamics, cost, N)

# Record (S, A)
all_obs = []
all_pred = []
all_pred_us = []
all_info = []
all_u = []

# First iteration to get better init for u
last_obs = x0
pred, pred_u = controller.fit(last_obs, us_init, n_iterations=500, on_iteration=on_iteration, tol=1e-2)
pred_u_constrained = constrain(pred_u, -10, 10)
last_u = pred_u_constrained[0]

all_pred.append(pred)
all_pred_us.append(pred_u)
flag = True

for i in range(300):
    print("step {} action {}".format(i, last_u))
    # Record
    all_obs.append(last_obs)
    all_u.append(last_u)

    next_obs, reward, done, info = env.step(last_u)
    env.render()

    all_info.append(np.array([info['th'], info['ddx'], info['ddth']]))

    if np.abs(info['th']) % (2*np.pi) <= 0.5 and flag:
        flag = False
        controller = iLQR(dynamics, cost, 20)
        all_pred_us[-1] = all_pred_us[-1][:20, :]
        print("=== near pole ===")
    pred, pred_u = controller.fit(next_obs, np.vstack([all_pred_us[-1][1:, :], all_pred_us[-1][-1, :].reshape(1, 1)]), n_iterations=20, on_iteration=on_iteration, tol=1e-2)
    pred_u_constrained = constrain(pred_u, -10, 10)
    all_pred.append(pred)
    all_pred_us.append(pred_u)

    last_obs = next_obs
    last_u = pred_u_constrained[0]

    print("th: {}, dth: {} a: {}".format(info['th'], next_obs[3], pred_u_constrained[0:2][:, 0]))

env.close()

torch.save({'all_obs': all_obs,
            'all_pred': all_pred,
            'all_pred_us': all_pred_us,
            'all_info': all_info,
            'all_u': all_u}, 'data/approx_mpc_ilqr_cartpole.pt')
