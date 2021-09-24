"""
Function for plotting trajectory from Gym Env and our learned model.
"""
import torch
import torch.nn as nn

from scipy.integrate import solve_ivp


def traj_pred(dt, model: nn.Module, y0: torch.Tensor, controls: torch.Tensor):
    """
    Generate one trajectory. Since we have a fixed control here, We need to solve the ivp independently
    :param dt: Sampling interval, should be equal to the env
    :param model: Our Model of the ODE
    :param y0: initial condition
    :param controls: control
    :return: None
    """
    num_steps, control_dim = controls.shape

    t_eval = torch.linspace(0, dt * (num_steps-1), num_steps)

    def dummy(t, y):
        # y should be in [r, cos q, sin q, dr, dq] form
        # piecewise constant control, same for Gym. Interesting to see the behaviour in real env.
        # open loop design
        idx = int(torch.tensor(t) / dt)
        y_u = torch.cat([torch.tensor(y), controls[idx]]).float()
        grad = model.defunc(0, y_u.view(1, -1))
        grad = grad.detach().numpy()[0, :-1]
        return grad

    sol = solve_ivp(dummy, [t_eval[0], t_eval[-1]], y0, method='RK45', t_eval=t_eval, max_step=100)
    return sol
