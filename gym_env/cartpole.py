"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R

Copied from Gym
Ref: https://coneural.org/florian/papers/05_cart_pole.pdf

Modified for
- higher order integrator
- Continuous Control
- Full Trajectory
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

import scipy.integrate

solve_ivp = scipy.integrate.solve_ivp


class CartPoleCustomEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -pi                     pi
        3       Pole Angular Velocity     -Inf                    Inf

    Actions:
        Type: Continuous
        Num   Action                      Min                     Max
        0     Left / Right Force          -2                      +2

        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it

    Reward:
        Reward follow PILCO defination:
            c(x) = 1 - exp(-||x - x_ref||^2 / sigma^2)

    Starting State:
        Uniform distribution

    Episode Termination:
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)

        # TODO: Can change this
        self.friction_pole_const = 0
        self.friction_cart_const = 0

        # self.force_mag = 10.0
        self.max_force = 10.0

        self.dt = 0.02  # seconds between state updates
        # self.kinematics_integrator = 'euler'

        # Angle at which to fail the episode
        self.theta_threshold_radians = 2 * np.pi  # Useless
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        # (x, x_dot, th, th_dot)

        self.x_ref = np.array([0, 0, 0, 0])

        high = np.array([
            self.x_threshold * 2,  # x
            1.0,  # cos th
            1.0,  # sin th
            np.finfo(np.float32).max,  # dx
            100 * np.pi],  # dth
            dtype=np.float32)

        self.action_space = spaces.Box(
            low=-self.max_force,
            high=self.max_force, shape=(2,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def dynamics(self, t, y, u):
        """
        Should return the grad of the system, and fit into ivp solver
        Check https://coneural.org/florian/papers/05_cart_pole.pdf
        :param t: time, should be zero since the dynamic is time independent
        :param y: observation, [x, xdot, th, dth]
        :param us: control, force acting on the cart u[1] act on dth and should always be zero
        :return: should be the same as observation
        """
        f = np.zeros_like(y)
        # Get ddth
        x, dx, th, dth = y
        cos = math.cos(th)  # Use math for speed
        sin = math.sin(th)

        # mc = self.masscart
        # ml = self.masspole
        # l = self.polemass_length
        # fc = self.friction_cart_const  # TODO: Hard to be model by current version, need dissipation version
        # fl = self.friction_pole_const
        # g = self.gravity

        # Nc = 0
        # # We can just assume Nc >= 0, coz the cart is confined into y = 0
        # sgn = np.array((Nc * dx >= 0)).astype(float)
        # ddth_temp1 = (-u - ml * l * (dth ** 2) * (sin + fc * cos * sgn)) / (mc + ml)
        # ddth_temp2 = ml * cos * (cos - fc * sgn) / (mc + ml)
        # ddth = (g * sin + cos * (ddth_temp1 + fc * g * sgn) - (fl * dth) / (ml * l)) / (l * (4 / 3 - ddth_temp2))
        #
        # Nc = (mc + ml) * g - ml * l * (ddth * sin + dth ** 2 * cos)
        #
        # if Nc < 0:
        #     sgn = np.array((Nc * dx >= 0)).astype(float)
        #     ddth_temp1 = (-u - ml * l * (dth ** 2) * (sin + fc * cos * sgn)) / (mc + ml)
        #     ddth_temp2 = ml * cos * (cos - fc * sgn) / (mc + ml)
        #     ddth = (g * sin + cos * (ddth_temp1 + fc * g * sgn) - (fl * dth) / (ml * l)) / (l * (4 / 3 - ddth_temp2))
        #
        #     Nc = (mc + ml) * g - ml * l * (ddth * sin + dth ** 2 * cos)
        #
        # ddx = (u + ml * l * ((dth ** 2) * sin - ddth * cos) - fc * Nc * sgn) / (mc + ml)

        temp = (u + self.polemass_length * dth ** 2 * sin) / self.total_mass
        thetaacc = (self.gravity * sin - cos * temp) / (
                    self.length * (4.0 / 3.0 - self.masspole * cos ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * cos / self.total_mass

        f[0] = dx
        f[1] = xacc
        f[2] = dth
        f[3] = thetaacc
        return f

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        _, ddx, _, ddth = self.dynamics(0, self.state, action)

        sol = solve_ivp(fun=lambda t, y: self.dynamics(t, y, action), t_span=[0, self.dt], y0=self.state,
                        method='RK45')
        self.state = sol.y[:, -1]
        x, x_dot, theta, theta_dot = self.state

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
        )

        if not done:
            reward = self._get_reward(action)
        elif self.steps_beyond_done is None:
            # Pole just fell!
            # assert False, 'Should not terminate for full swing'
            self.steps_beyond_done = 0
            reward = -100.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            # assert False, 'Should not terminate for full swing'
            self.steps_beyond_done += 1
            reward = -100.0

        # TODO: Test different representation?
        return self._get_obs(), reward, done, {'th': theta, 'ddx': ddx, 'ddth': ddth}

    def reset(self, state=None):
        if state is None:
            x = self.np_random.uniform(low=-1., high=1.)
            x_dot = self.np_random.uniform(low=-1., high=1.)
            theta = self.np_random.uniform(low=-np.pi, high=np.pi)
            theta_dot = self.np_random.uniform(low=-3, high=3)
            self.state = np.array([x, x_dot, theta, theta_dot])
        else:
            self.state = state

        x, x_dot, theta, theta_dot = self.state
        self.steps_beyond_done = None
        _, ddx, _, ddth = self.dynamics(0, self.state, 0)
        return self._get_obs(), {'th': theta, 'ddx': ddx, 'ddth': ddth}

    def _get_obs(self):
        x, x_dot, theta, theta_dot = self.state
        # return np.array([x, theta, x_dot, theta_dot])
        return np.array([x, np.cos(theta), np.sin(theta), x_dot, theta_dot])

    def _get_reward(self, action):
        x_ref = np.array([0, 0, 0, 0])
        x, dx, th, dth = self.state
        cost = 0.1 * (x_ref[0] - x) ** 2 + (x_ref[2] - self.angle_normalize(th)) ** 2 + 0.1 * (
                (x_ref[1] - dx) ** 2 + (x_ref[3] - dth) ** 2) + 0.001 * action ** 2
        cost = (1 - np.exp(-cost / (2 * 16)))
        return -cost

    def angle_normalize(self, x):
        return (((x + math.pi) % (2 * math.pi)) - math.pi)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


if __name__ == "__main__":
    rewards = []
    observations = []
    infos = []

    test_env = CartPoleCustomEnv()
    test_env.reset(np.array([0, 0, np.pi / 2, 0]))
    for i in range(200):
        obs, reward, done, info = test_env.step(0)
        rewards.append(reward)
        observations.append(obs)
        infos.append(np.array([info['th'], info['ddx'], info['ddth']]))
        test_env.render()
    infos = np.array(infos)
    observations = np.array(observations)
    print("done")
    None
