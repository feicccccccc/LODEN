"""
Module for Lagrangian Neural Networks with Control (LNNODE)

Reference:
https://arxiv.org/abs/1906.01563
https://torchdyn.readthedocs.io/en/latest/tutorials/09_lagrangian_nets.html
"""

from typing import Optional
import numpy as np

import torch
import torch.nn as nn

from nn import PSD


class LNN(nn.Module):
    def __init__(self,
                 num_angle: Optional[int] = None,
                 lagrangianNet: Optional[nn.Module] = None,
                 massMatrixNet: Optional[nn.Module] = None,
                 potentialNet: Optional[nn.Module] = None,
                 controlNet: Optional[nn.Module] = None
                 ):
        super().__init__()
        self.num_angle = num_angle if num_angle is not None else 0
        self.num_control = None
        self.num_raw, self._is_structured, self._is_control = self._check_input_network(lagrangianNet,
                                                                                        massMatrixNet,
                                                                                        potentialNet,
                                                                                        controlNet)

        if self._is_structured:
            self.massMatrixNet = massMatrixNet
            self.potentialNet = potentialNet
            self.L = self.Structured_L
        else:
            self.L = lagrangianNet

        self.controlNet = controlNet
        self.eps = 1e-6

    def forward(self, x):
        # x.shape (bs, [r, cos th, sin th, pr, pth, u])
        with torch.set_grad_enabled(True):
            x = x.requires_grad_(True)
            if self._is_control:
                q_dq, u = torch.split(x, [2 * self.num_raw + 3 * self.num_angle, self.num_control], dim=1)
            else:
                q_dq = x

            bs, _ = q_dq.shape
            q_dim = self.num_raw + 2 * self.num_angle
            dq_dim = self.num_raw + self.num_angle

            L = self.L(q_dq).sum()
            gradL = torch.autograd.grad(L, q_dq, create_graph=True)[0]

            split = [self.num_raw, self.num_angle, self.num_angle, self.num_raw + self.num_angle]
            dLdr, dLdx1, dLdx2, dLddr_dLddth = torch.split(gradL, split, dim=1)
            # dLddr_dLddth is a row vector in terms of math, we loop through it so doesn't matter.

            idx = np.cumsum(split)
            x1 = q_dq[:, idx[0]: idx[1]]
            x2 = q_dq[:, idx[1]: idx[2]]

            Jrth_dLdrdth = torch.zeros(bs, dq_dim, dq_dim)  # Jacobian w.r.t. r on dLddq
            Hdrdth_L = torch.zeros(bs, dq_dim, dq_dim)  # Hessian w.r.t. dr on L

            # Compute Hessian and Jacobian
            # For each dimension (generalised coordinate)
            for i in range(dq_dim):
                dLddr_dLddth_i = dLddr_dLddth[:, i]

                # (bs, dim), follow col vector convention (grad.T)
                grad_dLddr_dLddth_i = torch.autograd.grad(dLddr_dLddth_i.sum(), q_dq,
                                                          create_graph=True)[0]
                split = [self.num_raw, self.num_angle, self.num_angle, self.num_raw + self.num_angle]
                Jr_i, Jx1_i, Jx2_i, Hdrdth_i = torch.split(grad_dLddr_dLddth_i, split, dim=1)

                # Change of variable
                Jth_dLdth_j = -x2 * Jx1_i + x1 * Jx2_i
                Jrth_dLdrdth_j = torch.cat([Jr_i, Jth_dLdth_j], dim=1)

                Jrth_dLdrdth[:, i, :] = Jrth_dLdrdth_j
                Hdrdth_L[:, i, :] = Hdrdth_i

        Jq_dLddq = Jrth_dLdrdth
        Hdq_L = Hdrdth_L

        dLdth = -x2 * dLdx1 + x1 * dLdx2
        dLdrdth = torch.cat([dLdr, dLdth], dim=1)

        if self.controlNet is not None:
            control_matrix = self.controlNet(q_dq[:, :self.num_raw + 2 * self.num_angle])
            control = torch.einsum('ijk,ik->ij', control_matrix, u)
        else:
            control = torch.zeros_like(dLdrdth)

        Hdq_L += torch.eye(self.num_raw + self.num_angle) * self.eps  # TODO: computation stability is not clear here

        # (bs, dq_dim, dq_dim) (bs, dq_dim)
        dq_term = torch.einsum('bij, bj -> bi', Jq_dLddq, q_dq[:, self.num_raw + 2 * self.num_angle:])

        # ddq = torch.einsum('ijk, ij -> ik', torch.linalg.pinv(Hdq_L), control + dLdrdth - dq_term)
        try:
            ddq = torch.linalg.solve(Hdq_L, control + dLdrdth - dq_term)
        except:
            print("The trajectory diverge")
            ddq = torch.einsum('ijk, ij -> ik', torch.linalg.pinv(Hdq_L), control + dLdrdth - dq_term)

        dr_start_idx = self.num_raw + 2 * self.num_angle
        dth_start_idx = dr_start_idx + self.num_raw

        dr = q_dq[:, dr_start_idx:dr_start_idx + self.num_raw]
        dx1 = -x2 * q_dq[:, dth_start_idx:dth_start_idx + self.num_angle]
        dx2 = x1 * q_dq[:, dth_start_idx:dth_start_idx + self.num_angle]
        # ddr = ddq[:, :self.num_raw]
        # ddth = ddq[:, self.num_raw:self.num_raw + self.num_angle]

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
        p_T = torch.transpose(dq, 1, 2)

        # H = 1/2 q.T M q - V
        temp = torch.bmm(p_T, torch.bmm(M, dq))
        all_L = 0.5 * torch.squeeze(temp, 2) - V
        return all_L

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

    def _check_input_network(self, lagrangianNet, massMatrixNet, potentialNet, controlNet):
        l_none = lagrangianNet is None
        m_none = massMatrixNet is None
        v_none = potentialNet is None
        c_none = controlNet is None

        if l_none and m_none and v_none:
            raise AssertionError('No NN module is defined')
        if not l_none and (not m_none or not v_none):
            raise AssertionError('Use either structured and non-structured LNN')
        if l_none and (m_none or v_none):
            raise AssertionError('massMatrixNet or potentialNet is not present')

        if l_none and (not m_none and not v_none):
            is_structured = True
        else:
            is_structured = False

        if c_none:
            is_control = False
        else:
            is_control = True

        # Check dimension
        if not is_structured:
            depth = len(list(lagrangianNet.parameters())) // 2
            _, input_dim = lagrangianNet.get_parameter('layers.0.weight').shape
            output_dim, _ = lagrangianNet.get_parameter('layers.{}.weight'.format(depth - 1)).shape

            assert output_dim == 1, 'Output must be a Scalar Lagrangian'
            assert (input_dim - 3 * self.num_angle) % 2 == 0, 'Input must be a even number (coordinate and momentum)'

            num_raw = (input_dim - 3 * self.num_angle) // 2

            if is_control:
                depth_c = len(list(controlNet.parameters())) // 2
                _, input_dim_c = controlNet.get_parameter('layers.0.weight').shape
                output_dim_c, _ = controlNet.get_parameter('layers.{}.weight'.format(depth_c - 1)).shape

                assert input_dim_c == num_raw + 2 * self.num_angle, 'Control input must match q dimension'
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
                'Output of Mass matrix does not match number of dq coordinate'

            if is_control:
                depth_c = len(list(controlNet.parameters())) // 2
                _, input_dim_c = controlNet.get_parameter('layers.0.weight').shape
                output_dim_c, _ = controlNet.get_parameter('layers.{}.weight'.format(depth_c - 1)).shape
                self.num_control = controlNet._control_dim
                assert input_dim_c == num_raw + 2 * self.num_angle, 'Control input must match q dimension'

            return num_raw, is_structured, is_control
