from typing import Optional
import numpy as np

import torch
import torch.nn as nn

from nn import MLP, MatrixNet


class Baseline(nn.Module):
    def __init__(self,
                 num_angle: int,
                 num_raw: int,
                 num_control: Optional[int] = None,
                 ODENet: Optional[nn.Module] = None,
                 controlNet: Optional[nn.Module] = None
                 ):

        super().__init__()
        self.num_angle = num_angle if num_angle is not None else 0
        self.num_control = num_control
        self.num_raw = num_raw
        self._is_control = True if controlNet is not None else False

        self.ODENet = ODENet
        self.controlNet = controlNet

    def forward(self, x):
        # x.shape (bs, [r, cos th, sin th, dr, dth, u])
        # x.shape (bs, [r, cos th, sin th, dr, dth])
        bs, _ = x.shape
        q_dim = self.num_raw + 2 * self.num_angle
        dq_dim = self.num_raw + self.num_angle

        if self._is_control:
            split = [self.num_raw + 2 * self.num_angle + self.num_raw + self.num_angle, self.num_control]
            q_dq, u = torch.split(x, split, dim=1)
            split2 = [self.num_raw + 2 * self.num_angle, self.num_raw + self.num_angle]
            q, dq = torch.split(q_dq, split2, dim=1)
        else:
            q_dq = x
            split = [self.num_raw + 2 * self.num_angle, self.num_raw + self.num_angle]
            q, dq = torch.split(q_dq, split, dim=1)

        dq_ddq = self.ODENet(q_dq)
        split = [self.num_raw, self.num_angle, self.num_raw + self.num_angle]
        dr, dth, ddq = torch.split(dq_ddq, split, dim=1)

        x1 = q[:, self.num_raw:self.num_raw + self.num_angle]  # cos th
        x2 = q[:, self.num_raw + self.num_angle:]  # sin th
        dx1 = -x2 * dth
        dx2 = x1 * dth

        if self._is_control:
            force = torch.einsum('bij,bj->bi', self.controlNet(q), u)
            ddq_force = ddq + force
            return torch.cat([dr, dx1, dx2, ddq_force, torch.zeros_like(u)], dim=1).to(x)  # du = 0, constant control
        else:
            return torch.cat([dr, dx1, dx2, ddq], dim=1).to(x)


if __name__ == "__main__":
    num_raw = 3
    num_angle = 2
    batch = 16

    num_q = num_raw + 2 * num_angle
    num_dq = num_raw + num_angle

    num_control = 2

    # test_data = torch.randint(1, 10, (batch, num_q + num_dq + num_control)).float()
    test_data = torch.randint(1, 10, (batch, num_q + num_dq)).float()
    test_ODE = MLP([num_q + num_dq, 32, 2 * (num_raw + num_angle)], activation='relu')
    test_control = MatrixNet([num_q + num_dq, 32, num_dq], control_dim=num_control)

    test_model = Baseline(num_angle=num_angle,
                          num_raw=num_raw,
                          ODENet=test_ODE)

    print(test_model.forward(test_data).shape)
