"""
Fix Original Pendulum physics and rendering problem
- Fix integrator
- Fix rendering problem (incorrect representation of angle)
    - Angle at y axis, clockwise as positive
"""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

from scipy.integrate import solve_ivp


class PendulumCustomEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, g=9.81, full_phase=False):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.g = g
        self.m = 1.
        self.l = 1.
        self.viewer = None

        self.full_phase = full_phase  # Control reset distribution

        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        # u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        reward = -(angle_normalize(th + np.pi) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2))

        # # Explicit Euler Integrator
        # thdotdot = (g / l * np.sin(th) + 1. / (2. * m * l ** 2.) * u)
        # newthdot = thdot + thdotdot * dt
        # newth = angle_normalize(th + thdot * dt)

        # # Implicit Euler Integrator
        # thdotdot = (g / l * np.sin(th) + 1. / (2. * m * l ** 2.) * u)
        # newthdot = thdot + thdotdot * dt
        # newth = th + newthdot * dt

        # # Second Order RK2 Integrator
        # thdotdot = (g / l * np.sin(th) + 1. / (2. * m * l ** 2.) * u)  # Assume u is uniform
        # k2 = (g / l * np.sin(th) + .5 * thdotdot * dt) + 1. / (2. * m * l ** 2.) * u
        # newthdot = thdot + k2 * dt
        # newth = th + newthdot * dt

        # RK4 Integrator with scipy integrate
        # Simple Pendulum for continuous rod, make point mass for phase diagram
        def inverted_pendulum(t, y):
            th = y[0]
            v = y[1]
            grad = (v, -3. * g / (2. * l) * np.sin(th + np.pi) + 3. / (m * l ** 2.) * u)  # dxdt, dvdt
            # grad = (v, -g / l * np.sin(th) + u / (2 * m * l ** 2.))  # dxdt, dvdt
            return grad

        # RK4 is not precise enough
        sol = solve_ivp(fun=inverted_pendulum, t_span=[0, dt], y0=self.state, method='RK45')

        thdotdot = inverted_pendulum(None, self.state)[1]

        newth, newthdot = sol.y[:, -1]
        newth = angle_normalize(newth)

        self.state = np.array([newth, newthdot])
        return self._get_obs(), reward, False, {'th': th, 'dth': thdot, 'ddth': thdotdot}
        # The gradient from the dynamic

    def reset(self, state=None):
        # Modified for custom init
        if state is None:
            if self.full_phase:
                high = np.array([np.pi, np.pi])
            else:
                high = np.array([np.pi, 1])
            self.state = self.np_random.uniform(low=-high, high=high)
        else:
            th, th_dot = state
            self.state = np.array([angle_normalize(th), th_dot])
        self.last_u = 0
        thdotdot = -self.g / self.l * np.sin(self.state[0])
        return self._get_obs(), {'th': self.state[0], 'dth': self.state[1], 'ddth': thdotdot}

    def _get_obs(self):
        theta, thetadot = self.state
        # Recover true angle starting from x-axis, rotating anti-clockwise
        # x-pos, y-pos, thetadot
        # return np.array([np.sin(theta), -np.cos(theta), thetadot])

        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u is not None:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)
            # self.imgtrans.scale = (0, 0 / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


if __name__ == "__main__":
    rewards = []
    observations = []
    infos = []

    control = np.random.uniform(-2, 2, int(1000 / 20))
    control = np.repeat(control, 10)

    test_env = PendulumCustomEnv()
    test_env.reset(np.array([6*np.pi/4, 0]))
    for i in range(1000):
        obs, reward, done, info = test_env.step(0)
        rewards.append(reward)
        observations.append(obs)
        infos.append(np.array([info['th'], info['dth'], info['ddth']]))
        test_env.render()
    infos = np.array(infos)
    print("done")
