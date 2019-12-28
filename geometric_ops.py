import collections
from copy import deepcopy

import numpy as np
import torch


def move_to_gpu(self):
    if torch.cuda.is_available():
        return self.to('cuda:0', non_blocking=True)
    return self


setattr(torch.Tensor, 'move_to_gpu', move_to_gpu)

# Constants
from numpy.core.multiarray import ndarray
from torch import nn


NUM_DIMENSIONS = 3
NUM_DIHEDRALS = 3
BOND_LENGTHS = torch.tensor([145.801, 152.326, 132.868]).move_to_gpu()
BOND_ANGLES = torch.tensor([2.124, 1.941, 2.028]).move_to_gpu()
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
    diffs = torch.zeros(0, requires_grad=True).move_to_gpu()
    L = u.shape[0]/3
    for batch in range(u.shape[1]):
        u_b, v_b = u[:, batch], v[:, batch]
        mask_b = mask[:, batch]
        diff = calculate_pairwise_distances(u_b, v_b, mask_b)
        diffs = torch.cat([diffs, diff.norm(dim=0, keepdim=True)/(L*(L-1))])
    norm = diffs.norm(dim=0)
    return diffs

def calculate_pairwise_distances(u, v, mask=None):
    #type: (torch.Tensor, torch.Tensor, torch.Tensor) -> (torch.Tensor)
    """Calcualtes the pairwise distances between all atoms in the given tensor

    Args:
        u: tensor [3L, 3]
        v: tensor [3L, 3]
        mask: tensor [3L, 1]
    Returns:
        [3L,1] with diagonal elements 0
    """
    diffs = torch.zeros(0, requires_grad=True).move_to_gpu()
    for atom_id in range(u.shape[0]):
        if mask is not None and mask[atom_id] == 0:
            continue
        diff = ((u-u[atom_id]).norm(dim=1)-(v-v[atom_id]).float().norm(dim=1)).norm(keepdim=True, dim=0)  # [3L, 1]
        diffs = torch.cat([diffs, diff])
    return diffs


def calculate_coordinates(pred_torsions, r=BOND_LENGTHS, theta=BOND_ANGLES):
    #type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> (torch.Tensor)
    num_steps = pred_torsions.shape[0]
    batch_size = pred_torsions.shape[1]
    num_dihedrals = 3

    A = torch.tensor([0., 0., 1.]).move_to_gpu()
    B = torch.tensor([0., 1., 0.]).move_to_gpu()
    C = torch.tensor([1., 0., 0.]).move_to_gpu()

    broadcast = torch.ones((batch_size, 3)).move_to_gpu()
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
        D2 = torch.stack([-R * torch.ones(P.size()).move_to_gpu() * torch.cos(T),
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

def geometric_unit2(pred_coords, pred_torsions, bond_angles, bond_lens):
    # coordinates of last three atoms
    A = torch.stack([pred_coords[-3], pred_coords[-3], pred_coords[-3]])
    B = torch.stack([pred_coords[-2], pred_coords[-2], pred_coords[-2]])
    C = torch.stack([pred_coords[-1], pred_coords[-1], pred_coords[-1]])

    # internal coordinates
    T = torch.stack((bond_angles[0],bond_angles[1],bond_angles[2]))
    R = torch.stack((bond_lens[0],bond_lens[1],bond_lens[2]))
    P = torch.stack((bond_lensions[:, 0],pred_torsions[:, 1],pred_torsions[:, 2]))

    # 6x3x3 one triplet for each sample in the batch
    D2 = torch.stack([-R * torch.ones(P.size()).move_to_gpu() * torch.cos(T),
                      R * torch.cos(P) * torch.sin(T),
                      R * torch.sin(P) * torch.sin(T)], dim=1)

    # bsx3x3 one triplet for each sample in the batch
    BC = C - B
    print(BC.size())
    bc = BC / torch.norm(BC, 2, dim=1, keepdim=True)

    AB = B - A

    N = torch.cross(AB, bc)
    n = N / torch.norm(N, 2, dim=1, keepdim=True)

    M = torch.stack([bc, torch.cross(n, bc), n], dim=3)
    print(M.size())
    D = torch.bmm(M, D2.view(-1, 3, 1, 3)).squeeze() + C
    pred_coords = torch.cat([pred_coords, D.view(1, -1, 3, 3)])

    return pred_coords
