import numpy as np
import torch

import sys
import warnings
from gym_env import CartPoleCustomEnv, CartPoleCustomEnv
from torch.utils.data import TensorDataset

sys.path.append("../../")

env = CartPoleCustomEnv()
env.seed(0)
dt = env.dt

time_horizon = 200

num_test_pt = 100
stat_high = np.array([8 * np.pi / 4, 0.])
stat_low = -stat_high

control_high = np.array([1.])
control_low = -control_high


def create_data_set(num_data, stat_high, stat_low, control_high, control_low, time_horizon, split=20):
    for i in range(num_test_pt):
        trajs = []
        extras = []  # for extra info
        controls = []
        trajs_img = []
        data_count = 0

        while data_count < num_data:
            traj = []
            extra = []

            # Decide current control
            init_state = np.random.uniform(stat_high, stat_low)
            control = np.random.uniform(control_low, control_high, int(time_horizon / split))
            control = np.repeat(control, split)

            prev_obs, _ = env.reset(init_state)

            success_flag = True
            for t in range(time_horizon):
                traj.append(prev_obs)

                next_obs, reward, done, info = env.step(control[t])

                if isinstance(env, CartPoleCustomEnv):
                    extra.append(np.array([info['th'], info['dth'], info['ddth']]))

                elif isinstance(env, CartPoleCustomEnv):
                    extra.append(np.array([info['ddx'], info['ddth']]))
                else:
                    extra.append(np.array([0]))
                if done:
                    warnings.warn('Trajectory end in the middle of target time step')
                    success_flag = False
                    break

                prev_obs = next_obs

            if success_flag:
                trajs.append(traj)
                extras.append(extra)
                controls.append(control)
                data_count += 1

        # Constant Control data
        controls = torch.tensor(controls).view(num_data, time_horizon, 1)

        # Time stamp
        t_span = torch.linspace(dt, time_horizon * dt, time_horizon)
        t_span = t_span.repeat(num_data, 1).view(num_data, time_horizon, 1)

        X = torch.Tensor(trajs).float()  # obs
        U = controls.float()
        T = t_span.float()
        I = torch.Tensor(extras).float()

        dataset = TensorDataset(X, U, T, I)

        return dataset


test_set = create_data_set(num_test_pt, stat_high, stat_low, control_high, control_low, time_horizon)
torch.save(test_set, 'data/Pendulum_test_set_new.pt')

env = CartPoleCustomEnv()
env.seed(0)

time_horizon = 7

num_data_pt = 1000
stat_high = np.array([np.pi, 2.])
stat_low = np.array([-np.pi, -2.])

control_high = np.array([2.])
control_low = -control_high
# data_set = create_data_set(num_data_pt, stat_high, stat_low, control_high, control_low, time_horizon, split=7)
#
# torch.save(data_set, 'data/Pendulum_training_set_new.pt')
