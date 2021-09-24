"""
Reference https://github.com/anassinator/ilqr
Port the Theano code to Pytorch
"""

import numpy as np
import torch
from torchdyn.models import NeuralODE

from .dynamic import Dynamics, tensor_constrain


class CartPoleDynamic(torch.nn.Module):
    """
    Model the ture dynamic as a torch.nn.Module
    """
    def __init__(self, constrained=True, min_bounds=-10, max_bounds=10):
        super(CartPoleDynamic, self).__init__()
        self.constrained = constrained
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds

    def forward(self, x_u):
        f = torch.zeros_like(x_u)
        f.requires_grad_(False)
        x, cos, sin, dx, dth, u = torch.split(x_u, [1, 1, 1, 1, 1, 1], dim=1)

        gravity = 9.80665
        masscart = 1.0
        masspole = 0.1
        total_mass = (masspole + masscart)
        length = 0.5
        polemass_length = (masspole * length)

        if self.constrained:
            force = tensor_constrain(u, self.min_bounds, self.max_bounds)
        else:
            force = u

        temp = (force + polemass_length * dth ** 2 * sin) / total_mass
        ddth = (gravity * sin - cos * temp) / (length * (4.0 / 3.0 - masspole * cos ** 2 / total_mass))
        ddx = temp - polemass_length * ddth * cos / total_mass

        f[:, [0]] = dx
        f[:, [1]] = -sin * dth  # Different from the test model
        f[:, [2]] = cos * dth
        f[:, [3]] = ddx
        f[:, [4]] = ddth
        f[:, [5]] = 0
        return f

    @classmethod
    def augment_state(cls, state):
        """Augments angular state into a non-angular state by replacing theta
        with sin(theta) and cos(theta).

        In this case, it converts:

            [x, x', theta, theta'] -> [x, x', sin(theta), cos(theta), theta']

        Args:
            state: State vector [reducted_state_size].

        Returns:
            Augmented state size [state_size].
        """
        if state.ndim == 1:
            x, x_dot, theta, theta_dot = state
        else:
            x = state[..., 0].reshape(-1, 1)
            x_dot = state[..., 1].reshape(-1, 1)
            theta = state[..., 2].reshape(-1, 1)
            theta_dot = state[..., 3].reshape(-1, 1)

        return np.hstack([x, np.cos(theta), np.sin(theta), x_dot, theta_dot])

    @classmethod
    def reduce_state(cls, state):
        """Reduces a non-angular state into an angular state by replacing
        sin(theta) and cos(theta) with theta.

        In this case, it converts:

            [x, x', sin(theta), cos(theta), theta'] -> [x, x', theta, theta']

        Args:
            state: Augmented state vector [state_size].

        Returns:
            Reduced state size [reducted_state_size].
        """
        if state.ndim == 1:
            x, cos_theta, sin_theta, x_dot, theta_dot = state
        else:
            x = state[..., 0].reshape(-1, 1)
            x_dot = state[..., 3].reshape(-1, 1)
            sin_theta = state[..., 2].reshape(-1, 1)
            cos_theta = state[..., 1].reshape(-1, 1)
            theta_dot = state[..., 4].reshape(-1, 1)

        theta = np.arctan2(sin_theta, cos_theta)
        return np.hstack([x, x_dot, theta, theta_dot])


class PendulumDynamic(torch.nn.Module):
    """
    Model the ture dynamic as a torch.nn.Module
    """
    def __init__(self, constrained=True, min_bounds=-2, max_bounds=2):
        super(PendulumDynamic, self).__init__()
        self.constrained = constrained
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds

    def forward(self, x_u):
        f = torch.zeros_like(x_u)
        f.requires_grad_(False)
        cos, sin, dth, u = torch.split(x_u, [1, 1, 1, 1], dim=1)

        g = 9.80665
        m = 1.0
        l = 1.0

        if self.constrained:
            u = tensor_constrain(u, self.min_bounds, self.max_bounds)

        theta = torch.atan2(sin, cos)

        ddth = -3. * g / (2. * l) * torch.sin(theta + np.pi) + 3. / (m * l ** 2.) * u

        f[:, [0]] = -sin * dth  # Different from the test model
        f[:, [1]] = cos * dth
        f[:, [2]] = ddth
        f[:, [3]] = 0
        return f

    @classmethod
    def augment_state(cls, state):
        """Augments angular state into a non-angular state by replacing theta
        with sin(theta) and cos(theta).

        In this case, it converts:

            [theta, theta'] -> [sin(theta), cos(theta), theta']

        Args:
            state: State vector [reducted_state_size].

        Returns:
            Augmented state size [state_size].
        """
        if state.ndim == 1:
            theta, theta_dot = state
        else:
            theta = state[..., 0].reshape(-1, 1)
            theta_dot = state[..., 1].reshape(-1, 1)

        return np.hstack([np.cos(theta), np.sin(theta), theta_dot])

    @classmethod
    def reduce_state(cls, state):
        """Reduces a non-angular state into an angular state by replacing
        sin(theta) and cos(theta) with theta.

        In this case, it converts:

            [sin(theta), cos(theta), theta'] -> [theta, theta']

        Args:
            state: Augmented state vector [state_size].

        Returns:
            Reduced state size [reducted_state_size].
        """
        if state.ndim == 1:
            cos_theta, sin_theta, theta_dot = state
        else:
            cos_theta = state[..., 0].reshape(-1, 1)
            sin_theta = state[..., 1].reshape(-1, 1)
            theta_dot = state[..., 2].reshape(-1, 1)

        theta = np.arctan2(sin_theta, cos_theta)
        return np.hstack([theta, theta_dot])


class GeneralDyanmic(Dynamics):
    """
    Wrapper for Neural ODE model
    """

    def __init__(self,
                 dt,
                 model: NeuralODE,
                 state_size,
                 action_size,
                 **kwargs):
        self.dt = dt
        self._model = model  # use a NeuralODE model here
        self._t_span = torch.linspace(0, dt, 2)

        self._state_size = state_size
        self._action_size = action_size

        super(GeneralDyanmic, self).__init__()

    def f(self, x, u, i):
        # Constrain action space.
        tensor_x = torch.tensor(x).view(1, -1).float()
        tensor_u = torch.tensor(u).view(1, -1).float()

        x_u = torch.cat([tensor_x, tensor_u], dim=1)
        pred = self._model.trajectory(x_u, self._t_span)[-1, 0, :-1]  # (last time, batch, ignore control dim)
        # pred = (self.dt * self._model.defunc(0, x_u)[0, :] + x_u[0, :])[:-1]

        return pred.detach().numpy()

    def f_x(self, x, u, i):
        """Partial derivative of dynamics model with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            df/dx [state_size, state_size].
        """

        # th = np.arctan2(x[1], x[0])
        # dth = x[2]

        tensor_x = torch.tensor(x).view(1, -1).float()
        tensor_u = torch.tensor(u).view(1, -1).float()

        # Too slow
        dummy_func = lambda dum: self._model.trajectory(torch.cat([dum, tensor_u], dim=1), self._t_span)[-1, :, :-1].reshape(1, -1).sum(0)
        # dummy_func = lambda dum: (
        #             self.dt * self._model.defunc(0, torch.cat([dum, tensor_u], dim=1))[0, :-1] + dum[0, :])
        J_ref = torch.autograd.functional.jacobian(dummy_func, tensor_x).permute(1, 0, 2)[0, ...]
        return J_ref

        # # Construct the Jacobian directly from the grad
        # # TODO: only for pendulum
        # tensor_x.requires_grad_(True)
        # J = []
        # x_u = torch.cat([tensor_x, tensor_u], dim=1)
        # pred = (self.dt * self._model.defunc(0, x_u)[0, :] + x_u[0, :])[:-1]
        # for pred_i in pred:
        #     grad = torch.autograd.grad(pred_i, tensor_x, retain_graph=True)[0].numpy()
        #     J.append(grad)
        # J = np.vstack(J)
        # return J

    def f_u(self, x, u, i):
        """Partial derivative of dynamics model with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            df/du [state_size, action_size].
        """

        tensor_x = torch.tensor(x).view(1, -1).float()
        tensor_u = torch.tensor(u).view(1, -1).float()

        dummy_func = lambda dum: self._model.trajectory(torch.cat([tensor_x, dum], dim=1), self._t_span)[-1, :, :-1].reshape(1, -1).sum(0)
        J = torch.autograd.functional.jacobian(dummy_func, tensor_u).permute(1, 0, 2)[0, ...]

        # dummy_func = lambda dum: (
        #         self.dt * self._model.defunc(0, torch.cat([tensor_x, dum], dim=1))[0, :-1] + tensor_x[0, :])
        # J = torch.autograd.functional.jacobian(dummy_func, tensor_u).permute(1, 0, 2)[0, ...]

        return J.detach().numpy()

    def f_xx(self, x, u, i):
        """Second partial derivative of dynamics model with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/dx^2 [state_size, state_size, state_size].
        """
        raise NotImplementedError(
            "Hessians are not supported in BatchAutoDiffDynamics yet")

    def f_ux(self, x, u, i):
        """Second partial derivative of dynamics model with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/dudx [state_size, action_size, state_size].
        """
        raise NotImplementedError(
            "Hessians are not supported in BatchAutoDiffDynamics yet")

    def f_uu(self, x, u, i):
        """Second partial derivative of dynamics model with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/du^2 [state_size, action_size, action_size].
        """
        raise NotImplementedError(
            "Hessians are not supported in BatchAutoDiffDynamics yet")

    @property
    def state_size(self):
        """State size."""
        return self._state_size

    @property
    def action_size(self):
        """Action size."""
        return self._action_size

    @property
    def has_hessians(self):
        """Whether the second order derivatives are available."""
        return False
