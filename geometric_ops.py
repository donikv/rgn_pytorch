import collections
from copy import deepcopy

import numpy as np
import torch

# Constants
from numpy.core.multiarray import ndarray
from torch import nn

NUM_DIMENSIONS = 3
NUM_DIHEDRALS = 3
BOND_LENGTHS = torch.tensor([145.801, 152.326, 132.868]).cuda(0)
BOND_ANGLES = torch.tensor([2.124, 1.941, 2.028]).cuda(0)
# BOND_LENGTHS[0], BOND_LENGTHS[1], BOND_LENGTHS[2] =
# BOND_ANGLES[0], BOND_ANGLES[1], BOND_ANGLES[2] =


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
    diffs = torch.zeros([u.shape[1], u.shape[0]]).double().cuda(0)
    i = 0
    for batch in range(u.shape[1]):
        u_b, v_b = calculate_pairwise_distances(u[:, batch].double()), calculate_pairwise_distances(v[:, batch].double())

        diff = u_b - v_b
        diff = diff.norm(dim=1)
        if mask is not None:
            mask_b = mask[:, batch].double()
            diff = torch.mul(diff, mask_b)
            print(diff.shape)
        diffs[i] = diffs[i] + diff
        i = i + 1
    diffs = diffs.transpose(0, 1)
    norm = diffs.norm(dim=0)
    return norm

def calculate_pairwise_distances(u):
    #type: (torch.Tensor) -> (torch.Tensor)
    """Calcualtes the pairwise distances between all atoms in the given tensor

    Args:
        u: tensor [3L ,3]
    Returns:
        [3L,3L] with diagonal elements 0
    """
    diffs = torch.zeros([u.shape[0], u.shape[0]]).double().cuda(0)
    i = 0
    for atom in u:
        diff = (u-atom).norm(dim=1)
        diffs[i] = diffs[i] + diff
        i = i + 1
    return diffs.transpose(0, 1)


def calculate_coordinates(pred_torsions, r=BOND_LENGTHS, theta=BOND_ANGLES):
    #type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> (torch.Tensor)
    num_steps = pred_torsions.shape[0]
    batch_size = pred_torsions.shape[1]
    num_dihedrals = 3

    A = torch.tensor([0., 0., 1.]).cuda(0)
    B = torch.tensor([0., 1., 0.]).cuda(0)
    C = torch.tensor([1., 0., 0.]).cuda(0)

    broadcast = torch.ones((batch_size, 3)).cuda(0)
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
        D2 = torch.stack([-R * torch.ones(P.size()).cuda(0) * torch.cos(T),
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

