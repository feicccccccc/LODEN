"""
Test training with env and perform test
"""

import numpy as np
import torch
from torchdyn.models import NeuralODE
import matplotlib.pyplot as plt

import sys
sys.path.append('/Users/fei/Documents/UCL/Thesis/Code/MBRL_HNN')
sys.path.append('/Users/fei/Documents/UCL/Thesis/Code/MBRL_HNN/ilqr')

from ilqr.cost import QRCost
from ilqr.cartpole import PendulumDynamic, GeneralDyanmic
from ilqr.dynamic import constrain, tensor_constrain
from ilqr.controller import iLQR

# from cost import QRCost
# from cartpole import PendulumDynamic, GeneralDyanmic
# from dynamic import constrain, tensor_constrain
# from controller import iLQR

from gym_env import PendulumCustomEnv
from gym.wrappers import Monitor

np.set_printoptions(suppress=True)
np.random.seed(0)
device = 'cpu'

grad = NeuralODE(PendulumDynamic().to(device), sensitivity='adjoint', solver='euler')
dynamics = GeneralDyanmic(0.05, grad, state_size=3, action_size=1)

# env init
env = Monitor(PendulumCustomEnv(), './gym_data/true_pendulum', force=True)
init_state = np.array([np.pi, 0])
obs0 = env.reset(init_state)

env.seed(0)

# Test script
test_th = 1
test_x = [np.cos(test_th), np.sin(test_th), 2]
test_control = -3

out1 = dynamics.f(test_x, test_control, 0)
out2 = dynamics.f_x(test_x, test_control, 0)
out3 = dynamics.f_u(test_x, test_control, 0)

# Controller init
J_hist = []


def on_iteration(iteration_count, xs, us, J_opt, accepted, converged):
    J_hist.append(J_opt)
    info = "converged" if converged else ("accepted" if accepted else "failed")
    # if converged:
    final_state = PendulumDynamic.reduce_state(xs[-1])
    print("iteration", iteration_count, info, J_opt, final_state)


dt = 0.05

# Note that the augmented state is not all 0.
x_goal = PendulumDynamic.augment_state(np.array([0.0, 0.0]))
Q = np.eye(dynamics.state_size)
Q[2, 2] = 0.0
Q_terminal = 100. * np.eye(dynamics.state_size)
R = np.array([[0.1]])
cost = QRCost(Q, R, Q_terminal=Q_terminal, x_goal=x_goal)

N = 100
x0 = PendulumDynamic.augment_state(init_state)
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
pred, pred_u = controller.fit(last_obs, us_init, n_iterations=100, on_iteration=on_iteration)
pred_u_constrained = constrain(pred_u, -2, 2)
last_u = pred_u_constrained[0]

all_pred.append(pred)
all_pred_us.append(pred_u)

for i in range(200):

    # Record
    all_obs.append(last_obs)
    all_u.append(last_u)

    next_obs, reward, done, info = env.step(last_u)
    env.render()

    all_info.append(np.array([info['th'], info['dth'], info['ddth']]))

    pred, pred_u = controller.fit(next_obs, np.vstack([all_pred_us[-1][1:, :], all_pred_us[-1][-1, :].reshape(1, 1)]), n_iterations=1, on_iteration=on_iteration)
    pred_u_constrained = constrain(pred_u, -2, 2)
    all_pred.append(pred)
    all_pred_us.append(pred_u)

    last_obs = next_obs
    last_u = pred_u_constrained[0]

    print("step {}".format(i))
    print("th: {}, dth: {} a: {}".format(info['th'], info['dth'], last_u))

# torch.save({'all_obs': all_obs,
#             'all_pred': all_pred,
#             'all_pred_us': all_pred_us,
#             'all_info': all_info,
#             'all_u': all_u}, 'data/true_mpc_ilqr_pendulum.pt')

env.close()
