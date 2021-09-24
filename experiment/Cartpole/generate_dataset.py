import numpy as np
import torch

import sys
import warnings
from gym_env import PendulumCustomEnv, CartPoleCustomEnv
from torch.utils.data import TensorDataset

sys.path.append("../../")

env = CartPoleCustomEnv()
env.seed(0)
dt = env.dt

time_horizon = 100
# x, x_dot, theta, theta_dot = self.state
num_test_pt = 100
stat_high = np.array([2, 0, 2, 0])
stat_low = -stat_high

control_high = np.array([10.])
control_low = -control_high


def create_data_set(num_data, stat_high, stat_low, control_high, control_low, time_horizon, split=10, no_act_portion=0.6):
    for i in range(num_test_pt):
        trajs = []
        extras = []  # for extra info
        controls = []
        trajs_img = []
        data_count = 0

        while data_count < num_data:
            print(data_count)
            traj = []
            extra = []

            # Decide current control
            init_state = np.random.uniform(stat_high, stat_low)
            control = np.random.uniform(control_low, control_high, int(time_horizon / split))
            # control = np.random.choice([0, 10, -10], int(time_horizon / split))
            np.random.randint(-10, 10 + 1, int(time_horizon / split))
            control = np.repeat(control, split)

            if np.random.rand() < no_act_portion:
                control = np.zeros_like(control)

            prev_obs, _ = env.reset(init_state)

            success_flag = True
            for t in range(time_horizon):
                traj.append(prev_obs)

                next_obs, reward, done, info = env.step(control[t])

                if isinstance(env, PendulumCustomEnv):
                    extra.append(np.array([info['th'], info['dth'], info['ddth']]))

                elif isinstance(env, CartPoleCustomEnv):
                    extra.append(np.array([info['th'], info['ddx'], info['ddth']]))
                else:
                    extra.append(np.array([0]))
                if done:
                    print('Trajectory end in the middle of target time step')
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
torch.save(test_set, 'data/Cartpole_test_set_test.pt')

env = CartPoleCustomEnv()
env.seed(0)

time_horizon = 7

num_data_pt = 20000
stat_high = np.array([1, 2, np.pi, 8])
stat_low = np.array([-1, -2, -np.pi, -8])

control_high = np.array([10.])
control_low = -control_high

data_set = create_data_set(num_data_pt, stat_high, stat_low, control_high, control_low, time_horizon, split=time_horizon, no_act_portion=0)
torch.save(data_set, 'data/Cartpole_training_set_20k_test.pt')
