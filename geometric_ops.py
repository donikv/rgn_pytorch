import collections
from copy import deepcopy

import numpy as np
import torch

# Constants
from numpy.core.multiarray import ndarray
from torch import nn

NUM_DIMENSIONS = 3
NUM_DIHEDRALS = 3
BOND_LENGTHS = torch.empty(3)
BOND_ANGLES = torch.empty(3)
BOND_LENGTHS[0], BOND_LENGTHS[1], BOND_LENGTHS[2] = [145.801, 152.326, 132.868]
BOND_ANGLES[0], BOND_ANGLES[1], BOND_ANGLES[2] = [2.124, 1.941, 2.028]


def calculate_dihedrals(p, alphabet):
    """Converts the given input of weigths over the alphabet into a triple of dihederal angles

    Args:
        p: [BATCH_SIZE, NUM_ANGLES]
        alphabet:  [NUM_ANGLES, NUM_DIHEDRALS]

    Returns:
        [BATCH_SIZE, NUM_DIHEDRALS]

    """

    sins = torch.sin(alphabet)
    coss = torch.cos(alphabet)

    y_coords = p @ sins
    x_coords = p @ coss

    return torch.atan2(y_coords, x_coords)


def drmsd(u, v, mask=None):
    #type: (torch.Tensor, torch.Tensor, torch.Tensor) -> (torch.Tensor)
    diffs = torch.zeros(0).double()
    for batch in range(u.shape[1]):
        u_b, v_b = u[:, batch].double(), v[:, batch].double()
        diff = torch.pairwise_distance(u_b, v_b, keepdim=True)
        diffs = torch.cat([diffs, diff], dim=1)
    diffs = torch.mul(diffs, mask).transpose(0, 1)
    norm = diffs.norm(dim=0)
    return norm


def calculate_coordinates(pred_torsions, r=BOND_LENGTHS, theta=BOND_ANGLES):
    #type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> (torch.Tensor)
    num_steps = pred_torsions.shape[0]
    batch_size = pred_torsions.shape[1]
    num_dihedrals = 3

    A = torch.tensor([0., 0., 1.])
    B = torch.tensor([0., 1., 0.])
    C = torch.tensor([1., 0., 0.])

    broadcast = torch.ones((batch_size, 3))
    pred_coords = torch.stack([A * broadcast, B * broadcast, C * broadcast])

    for ix, triplet in enumerate(pred_torsions[1:]):
        pred_coords = geometric_unit(pred_coords, triplet,
                                     theta,
                                     r)
    return pred_coords


def geometric_unit(pred_coords, pred_torsions, bond_angles, bond_lens):
    for i in range(3):
        # coordinates of last three atoms
        A, B, C = pred_coords[-3], pred_coords[-2], pred_coords[-1]

        # internal coordinates
        T = bond_angles[i]
        R = bond_lens[i]
        P = pred_torsions[:, i]

        # 6x3 one triplet for each sample in the batch
        D2 = torch.stack([-R * torch.ones(P.size()) * torch.cos(T),
                          R * torch.cos(P) * torch.sin(T),
                          R * torch.sin(P) * torch.sin(T)], dim=1)

        # bsx3 one triplet for each sample in the batch
        BC = C - B
        bc = BC / torch.norm(BC, 2, dim=1, keepdim=True)

        AB = B - A

        N = torch.cross(AB, bc)
        n = N / torch.norm(N, 2, dim=1, keepdim=True)

        M = torch.stack([bc, torch.cross(n, bc), n], dim=2)

        D = torch.bmm(M, D2.view(-1, 3, 1)).squeeze() + C
        pred_coords = torch.cat([pred_coords, D.view(1, -1, 3)])

    return pred_coords

