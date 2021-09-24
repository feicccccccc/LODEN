"""
Module for Hamiltonian Neural Networks with Control (SymODE)

Reference:
https://arxiv.org/abs/1906.01563  (HNN)
https://arxiv.org/abs/1909.12077  (SymODE)
https://torchdyn.readthedocs.io/en/latest/tutorials/08_hamiltonian_nets.html
"""
from typing import Optional
import numpy as np

import torch
import torch.nn as nn

from nn import PSD


class HNN(nn.Module):
    def __init__(self,
                 num_angle: Optional[int] = None,
                 hamiltonianNet: Optional[nn.Module] = None,
                 massMatrixNet: Optional[nn.Module] = None,
                 potentialNet: Optional[nn.Module] = None,
                 controlNet: Optional[nn.Module] = None
                 ):
        super().__init__()
        self.num_angle = num_angle if num_angle is not None else 0
        self.num_control = None
        self.num_raw, self._is_structured, self._is_control = self._check_input_network(hamiltonianNet,
                                                                                        massMatrixNet,
                                                                                        potentialNet,
                                                                                        controlNet)

        if self._is_structured:
            self.massMatrixNet = massMatrixNet
            self.potentialNet = potentialNet
            self.H = self.Structured_H
        else:
            self.H = hamiltonianNet

        self.controlNet = controlNet

    def forward(self, x):
        # x.shape (bs, [r, cos th, sin th, pr, pth, u])
        with torch.set_grad_enabled(True):
            x = x.requires_grad_(True)
            if self._is_control:
                q_p, u = torch.split(x, [2 * self.num_raw + 3 * self.num_angle, self.num_control], dim=1)
            else:
                q_p = x

            H = self.H(q_p).sum()
            gradH = torch.autograd.grad(H, q_p, allow_unused=False, create_graph=True)[0]

        split = [self.num_raw, self.num_angle, self.num_angle, self.num_raw, self.num_angle]
        dHdr, dHdx1, dHdx2, dHdpr, dHdpth = torch.split(gradH, split, dim=1)

        idx = np.cumsum(split)
        x1 = x[:, idx[0]: idx[1]]
        x2 = x[:, idx[1]: idx[2]]

        dr = dHdpr
        dx1 = -x2 * dHdpth
        dx2 = x1 * dHdpth
        dpr = -dHdr
        dpth = x2 * dHdx1 - x1 * dHdx2

        q, p = torch.split(q_p, [self.num_raw + 2 * self.num_angle, self.num_raw + self.num_angle], dim=1)
        dpr_dpth = torch.cat([dpr, dpth], dim=1)

        if self._is_control:
            force = torch.einsum('bij,bj->bi', self.controlNet(q), u)
            dpr_dpth += force
            return torch.cat([dr, dx1, dx2, dpr_dpth, torch.zeros_like(u)], dim=1).to(x)  # du = 0, constant control
        else:
            return torch.cat([dr, dx1, dx2, dpr_dpth], dim=1).to(x)

    def Structured_H(self, x):
        """
        Compute the Hamiltonian with structured way
        H = 1/2 p.T M-1(q) p + V(q)
        """
        bs, dim = x.shape

        split = [self.num_raw + 2 * self.num_angle,
                 self.num_raw + self.num_angle]
        q, p = torch.split(x, split, dim=1)

        # Structured HNN
        M = self.massMatrixNet.forward(q)
        V = self.potentialNet.forward(q)

        # Reshape for bmm (batch, matrix dim)
        p = p.view(bs, -1, 1)
        p_T = torch.transpose(p, 1, 2)

        # H = 1/2 p.T M p + V
        # TODO: Numerical stability
        temp = torch.bmm(p_T, torch.bmm(M, p))
        all_H = 0.5 * torch.squeeze(temp, 2) + V
        return all_H

    def _check_input_network(self, hamiltonianNet, massMatrixNet, potentialNet, controlNet):
        h_none = hamiltonianNet is None
        m_none = massMatrixNet is None
        v_none = potentialNet is None
        c_none = controlNet is None

        if h_none and m_none and v_none:
            raise AssertionError('No NN module is defined')
        if not h_none and (not m_none or not v_none):
            raise AssertionError('Use either structured and non-structured HNN')
        if h_none and (m_none or v_none):
            raise AssertionError('massMatrixNet or potentialNet is not present')

        if h_none and (not m_none and not v_none):
            is_structured = True
        else:
            is_structured = False

        if c_none:
            is_control = False
        else:
            is_control = True

        # Check dimension
        if not is_structured:
            depth = len(list(hamiltonianNet.parameters())) // 2
            _, input_dim = hamiltonianNet.get_parameter('layers.0.weight').shape
            output_dim, _ = hamiltonianNet.get_parameter('layers.{}.weight'.format(depth - 1)).shape

            assert output_dim == 1, 'Output must be a Scalar Hamiltonian'
            assert (input_dim - 3 * self.num_angle) % 2 == 0, 'Input must be a even number (coordinate and momentum)'

            num_raw = (input_dim - 3 * self.num_angle) // 2

            if is_control:
                depth_c = len(list(controlNet.parameters())) // 2
                _, input_dim_c = controlNet.get_parameter('layers.0.weight').shape
                output_dim_c, _ = controlNet.get_parameter('layers.{}.weight'.format(depth_c - 1)).shape
                assert input_dim_c == num_raw + 2 * self.num_angle, 'Input of Control Matrix must match q dimension'
                self.num_control = controlNet._control_dim

            return num_raw, is_structured, is_control
        else:
            assert isinstance(massMatrixNet, PSD), 'Mass Matrix must be PSD net'

            depth_m = len(list(massMatrixNet.parameters())) // 2
            depth_v = len(list(potentialNet.parameters())) // 2

            _, input_dim_m = massMatrixNet.get_parameter('layers.0.weight').shape
            output_dim_m, _ = massMatrixNet.get_parameter('layers.{}.weight'.format(depth_m - 1)).shape

            _, input_dim_v = potentialNet.get_parameter('layers.0.weight').shape
            output_dim_v, _ = potentialNet.get_parameter('layers.{}.weight'.format(depth_v - 1)).shape

            assert output_dim_v == 1, 'Output must be a Scalar Potential'
            assert input_dim_m == input_dim_v, 'Dimension of input of Mass and Potential must be the same'

            num_raw = input_dim_m - 2 * self.num_angle
            total_dim = num_raw + self.num_angle  # Before Projection
            assert output_dim_m == (total_dim ** 2 - total_dim) // 2 + total_dim, \
                'Output of Mass matrix does not match number of Canonical coordinate'

            if is_control:
                depth_c = len(list(controlNet.parameters())) // 2
                _, input_dim_c = controlNet.get_parameter('layers.0.weight').shape
                output_dim_c, _ = controlNet.get_parameter('layers.{}.weight'.format(depth_c - 1)).shape

                assert input_dim_c == num_raw + 2 * self.num_angle, 'Input of Control Matrix must match q dimension'
                self.num_control = controlNet._control_dim

            return num_raw, is_structured, is_control
