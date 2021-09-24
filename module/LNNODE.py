"""
Module for Lagrangian Neural Networks with Control (LNNODE)
Equivalent SymODE, with faster computation

Reference:
https://arxiv.org/abs/1906.01563
https://torchdyn.readthedocs.io/en/latest/tutorials/09_lagrangian_nets.html
"""

from typing import Optional
import numpy as np

import torch
import torch.nn as nn

from nn import PSD


class LNNODE(nn.Module):
    def __init__(self,
                 num_angle: Optional[int] = None,
                 massMatrixNet: Optional[nn.Module] = None,
                 potentialNet: Optional[nn.Module] = None,
                 controlNet: Optional[nn.Module] = None
                 ):

        super().__init__()
        self.num_angle = num_angle if num_angle is not None else 0
        self.num_control = None
        self.num_raw, self._is_control = self._check_input_network(massMatrixNet,
                                                                   potentialNet,
                                                                   controlNet)

        self.massMatrixNet = massMatrixNet
        self.potentialNet = potentialNet
        self.L = self.Structured_L

        self.controlNet = controlNet

    def forward(self, x):
        # x.shape (bs, [r, cos th, sin th, pr, pth, u])
        bs, _ = x.shape
        with torch.set_grad_enabled(True):
            x = x.requires_grad_(True)
            if self._is_control:
                split = [self.num_raw + 2 * self.num_angle, self.num_raw + self.num_angle, self.num_control]
                q, dq, u = torch.split(x, split, dim=1)
            else:
                split = [self.num_raw + 2 * self.num_angle, self.num_raw + self.num_angle]
                q, dq = torch.split(x, split, dim=1)

            x1 = q[:, self.num_raw:self.num_raw + self.num_angle]
            x2 = q[:, self.num_raw + self.num_angle:]
            dx1 = -x2 * dq[:, self.num_raw:]
            dx2 = x1 * dq[:, self.num_raw:]

            q_dim = self.num_raw + 2 * self.num_angle
            dq_dim = self.num_raw + self.num_angle

            # # =====
            # # Get dMdt
            # drdx1dx2 = torch.cat([dq[:, :self.num_raw], dx1, dx2], dim=1)
            # dummy_func = lambda dum: self.massMatrixNet(dum).reshape(bs, -1).sum(0)
            # Jq_M = torch.autograd.functional.jacobian(dummy_func, q).permute(1, 0, 2)
            #
            # dMdq = Jq_M.permute(0, 2, 1).reshape(bs, q_dim, dq_dim, dq_dim)
            # dMdt = torch.einsum('bijk,bi->bjk', dMdq, drdx1dx2)
            # # =====

            # =====
            # use Mdq as the target
            M = self.massMatrixNet(q)
            Mdq = torch.einsum('bij,bj->bi', M, dq)
            dqMdq = torch.zeros(bs, dq_dim, dq_dim)
            for i in range(dq_dim):
                curMdq = Mdq[:, i]
                dqiMdq = torch.autograd.grad(curMdq.sum(), q, create_graph=True)[0]
                dr_term, dx1_term, dx2_term = torch.split(dqiMdq, [self.num_raw, self.num_angle, self.num_angle], dim=1)
                dth_term = -x2 * dx1_term + x1 * dx2_term
                dqMdq[:, i, :] = torch.cat([dr_term, dth_term], dim=1)
            # =====

            # Get dq (1/2 dq M(q) dq - V(q))
            all_L = self.L(torch.cat([q, dq], dim=1))
            gradL = torch.autograd.grad(all_L.sum(), q, create_graph=True)[0]
            split = [self.num_raw, self.num_angle, self.num_angle]
            dLdr, dLdx1, dLdx2 = torch.split(gradL, split, dim=1)
            dLdth = -x2 * dLdx1 + x1 * dLdx2
            dLdq = torch.cat([dLdr, dLdth], dim=1)

            if self._is_control:
                force = torch.einsum('bij,bj->bi', self.controlNet(q), u)
                # Mddq = force + dLdq - torch.einsum('bij,bj->bi', dMdt, dq)
                Mddq = force + dLdq - torch.einsum('bij,bj->bi', dqMdq, dq)
            else:
                # Mddq = dLdq - torch.einsum('bij,bj->bi', dMdt, dq)
                Mddq = dLdq - torch.einsum('bij,bj->bi', dqMdq, dq)

            L = torch.linalg.cholesky(self.massMatrixNet(q))
            ddq = torch.cholesky_solve(Mddq.reshape(bs, -1, 1), L).reshape(bs, -1)
            dr = dq[:, :self.num_raw]
        if self._is_control:
            return torch.cat([dr, dx1, dx2, ddq, torch.zeros_like(u)], dim=1).to(x)  # du = 0, constant control
        else:
            return torch.cat([dr, dx1, dx2, ddq], dim=1).to(x)

    def Structured_L(self, x):
        """
        Compute the Hamiltonian with structured way
        L = 1/2 q.T M(q) q - V(q)
        """
        bs, dim = x.shape

        split = [self.num_raw + 2 * self.num_angle,
                 self.num_raw + self.num_angle]
        q, dq = torch.split(x, split, dim=1)

        # Structured LNN
        M = self.massMatrixNet.forward(q)
        V = self.potentialNet.forward(q)

        # Reshape for bmm (batch, matrix dim)
        dq = dq.view(bs, -1, 1)
        dq_T = torch.transpose(dq, 1, 2)

        # H = 1/2 q.T M q - V
        temp = torch.bmm(dq_T, torch.bmm(M, dq))
        all_L = 0.5 * torch.squeeze(temp, 2) - V
        return all_L

    def Structured_H(self, x):
        """
        Compute the Hamiltonian with structured way
        L = 1/2 q.T M(q) q - V(q)
        """
        bs, dim = x.shape

        split = [self.num_raw + 2 * self.num_angle,
                 self.num_raw + self.num_angle]
        q, dq = torch.split(x, split, dim=1)

        # Structured LNN
        M = self.massMatrixNet.forward(q)
        V = self.potentialNet.forward(q)

        # Reshape for bmm (batch, matrix dim)
        dq = dq.view(bs, -1, 1)
        dq_T = torch.transpose(dq, 1, 2)

        # H = 1/2 q.T M q - V
        temp = torch.bmm(dq_T, torch.bmm(M, dq))
        all_H = 0.5 * torch.squeeze(temp, 2) + V
        return all_H

    def H_p_legendre(self, x):
        """
        Perform legendre transform
        TODO: p seems not something we can control
        :param x:
        :return:
        """
        with torch.set_grad_enabled(True):
            x = x.requires_grad_(True)
            split = [self.num_raw + 2 * self.num_angle,
                     self.num_raw + self.num_angle]
            q, dq = torch.split(x, split, dim=1)
            all_L = self.L(x)
            gradL = torch.autograd.grad(all_L.sum(), x, create_graph=True)[0]  # Conjugate Momentum

        _, p = torch.split(gradL, split, dim=1)
        all_H = torch.einsum('ij, ik -> i', p, dq) - all_L[:, 0]
        # TODO: Check for higher dim output

        return all_H, p

    def _check_input_network(self, massMatrixNet, potentialNet, controlNet):
        m_none = massMatrixNet is None
        v_none = potentialNet is None
        c_none = controlNet is None

        if m_none and v_none:
            raise AssertionError('No NN module is defined')
        if m_none or v_none:
            raise AssertionError('massMatrixNet or potentialNet is not present')

        if c_none:
            is_control = False
        else:
            is_control = True

        # Check dimension
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
            'Output of Mass matrix does not match number of dq coordinate'

        if is_control:
            depth_c = len(list(controlNet.parameters())) // 2
            _, input_dim_c = controlNet.get_parameter('layers.0.weight').shape
            output_dim_c, _ = controlNet.get_parameter('layers.{}.weight'.format(depth_c - 1)).shape
            self.num_control = controlNet._control_dim
            assert input_dim_c == num_raw + 2 * self.num_angle, 'Control input must match q dimension'

        return num_raw, is_control
