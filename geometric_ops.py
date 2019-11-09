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

    sins = np.sin(alphabet)
    coss = np.cos(alphabet)

    y_coords = p @ sins
    x_coords = p @ coss

    return torch.atan2(y_coords, x_coords)


def calculate_coordinates_old(dihedral, r=BOND_LENGTHS, theta=BOND_ANGLES):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> (torch.Tensor)
    """ Takes triplets of dihedral angles (omega, phi, psi) and returns 3D points ready for use in
        reconstruction of coordinates. Bond lengths and angles are based on idealized averages.

    Args:
        dihedral: [NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]

    Returns:
        [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS]
    """

    num_steps = dihedral.shape[0]
    batch_size = dihedral.shape[1]  # important to use get_shape() to keep batch_size fixed for performance reasons

    r_cos_theta = torch.Tensor(r * torch.cos(np.pi - theta))  # [NUM_DIHEDRALS]
    r_sin_theta = torch.Tensor(r * torch.sin(np.pi - theta))  # [NUM_DIHEDRALS]

    pt_x = torch.Tensor(np.tile(r_cos_theta.reshape((1, 1, -1)),
                   (num_steps, batch_size, 1)))  # [NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]
    pt_y = torch.mul(torch.cos(dihedral), r_sin_theta)  # [NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]
    pt_z = torch.mul(torch.sin(dihedral), r_sin_theta)  # [NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]

    pt = torch.stack([pt_x, pt_y, pt_z])  # [NUM_DIMS, NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]
    pt_perm = pt.permute(dims=[1, 3, 2, 0])  # [NUM_STEPS, NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMS]
    pt = pt_perm.reshape([num_steps * NUM_DIHEDRALS, batch_size, NUM_DIMENSIONS])
    # [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMS]

    # Convert points to coordinats
    s = pt.shape[0]
    num_frags = num_steps
    Triplet = collections.namedtuple('Triplet', 'a, b, c')
    init_mat = np.array([[-np.sqrt(1.0 / 2.0), np.sqrt(3.0 / 2.0), 0], [-np.sqrt(2.0), 0, 0], [0, 0, 0]],
                        dtype='float32')
    init_coords = Triplet(*[torch.reshape(np.tile(row[None], torch.stack([num_frags * batch_size, 1])),
                                       [num_frags, batch_size, NUM_DIMENSIONS]) for row in init_mat])
    print(init_coords)
    # NUM_DIHEDRALS x [NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS]
    pt = tf.pad(pt, [[0, r], [0, 0], [0, 0]])  # [NUM_FRAGS x FRAG_SIZE, BATCH_SIZE, NUM_DIMENSIONS]
    pt = pt.reshape([num_frags, -1, batch_size,
                    NUM_DIMENSIONS])  # [NUM_FRAGS, FRAG_SIZE,  BATCH_SIZE, NUM_DIMENSIONS]
    pt = pt.permute(dims=[1, 0, 2, 3])

    # extension function used for single atom reconstruction and whole fragment alignment
    def extend(tri, pt, multi_m):
        # type: (Triplet, torch.Tensor, bool) -> (torch.Tensor)
        """
        Args:
            tri: NUM_DIHEDRALS x [NUM_FRAGS/0,         BATCH_SIZE, NUM_DIMENSIONS]
            pt:                  [NUM_FRAGS/FRAG_SIZE, BATCH_SIZE, NUM_DIMENSIONS]
            multi_m: bool indicating whether m (and tri) is higher rank. pt is always higher rank; what changes is what the first rank is.
        """
        normaliztor = torch.nn.BatchNorm1d(batch_size)
        bc = normaliztor(tri[2] - tri[1])  # [NUM_FRAGS/0, BATCH_SIZE, NUM_DIMS]
        n = normaliztor(torch.cross(tri[1] - tri[0], bc))  # [NUM_FRAGS/0, BATCH_SIZE, NUM_DIMS]
        if multi_m:  # multiple fragments, one atom at a time.
            m = torch.Tensor.permute(torch.stack([bc, torch.cross(n, bc), n]), dims=[1, 2, 3, 0])  # [NUM_FRAGS,   BATCH_SIZE, NUM_DIMS, 3 TRANS]
        else:  # single fragment, reconstructed entirely at once.
            # WARNING possibly wrong
            s = np.pad([[0, 1]], pt.shape)  # FRAG_SIZE, BATCH_SIZE, NUM_DIMS, 3 TRANS
            m = torch.Tensor.permute(torch.stack([bc, np.cross(n, bc), n]), dims=[1, 2, 0])  # [BATCH_SIZE, NUM_DIMS, 3 TRANS]
            m = np.reshape(np.tile(m, [s[0], 1, 1]), s)  # [FRAG_SIZE, BATCH_SIZE, NUM_DIMS, 3 TRANS]
        coord = torch.add(torch.squeeze(torch.matmul(m, pt.expand(3)), dim=3), tri[2])  # [NUM_FRAGS/FRAG_SIZE, BATCH_SIZE, NUM_DIMS]
        return coord

    # loop over FRAG_SIZE in NUM_FRAGS parallel fragments, sequentially generating the coordinates for each fragment across all batches
    i = 0
    s_padded = pt.shape[0]  # FRAG_SIZE
    coords_np = torch.empty(s_padded)
    # FRAG_SIZE x [NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS]

    def loop_extend(i, tri, coords_ta):  # FRAG_SIZE x [NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS]
        coord = extend(tri, pt[i], True)
        coords_ta[i] = coord
        return [i + 1, Triplet(tri[1], tri[2], coord), coords_ta]

    tri = init_coords[0]
    tris = deepcopy(coords_np)
    while i < s_padded:
        i, tri, coords_np = loop_extend(i, tri, coords_np)
        tris[i-1] = tri
    tris = tri
    # NUM_DIHEDRALS x [NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS],
    # FRAG_SIZE x [NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS]

    # loop over NUM_FRAGS in reverse order, bringing all the downstream fragments in alignment with current fragment
    coords_pretrans = coords_np.permute(dims=[1, 0, 2, 3])  # [NUM_FRAGS, FRAG_SIZE, BATCH_SIZE, NUM_DIMENSIONS]
    i = coords_pretrans.shape[0] - 2  # NUM_FRAGS

    def loop_trans(i, coords):
        transformed_coords = extend(Triplet(*[di[i] for di in tris]), coords, False)
        return [i - 1, np.concatenate([coords_pretrans[i], transformed_coords], 0)]

    coords_trans = np.empty(i, dtype=np.float32)
    while i > -1:
        i, coord = loop_trans(i, coords_pretrans[-1])
        coords_trans[i + 1] = coord
    # [NUM_FRAGS x FRAG_SIZE, BATCH_SIZE, NUM_DIMENSIONS]

    # lose last atom and pad from the front to gain an atom ([0,0,0], consistent with init_mat), to maintain correct atom ordering
    coords = torch.constant_pad_nd(coords_trans[:s - 1], ([1, 0], [0, 0], [0, 0]))  # [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS]

    return coords


def drmsd(u, v, weights=None):
    """ Computes the dRMSD of two tensors of vectors.

        Vectors are assumed to be in the third dimension. Op is done element-wise over batch.

    Args:
        u, v:    [NUM_STEPS, BATCH_SIZE, NUM_DIMENSIONS]
        weights: [NUM_STEPS, NUM_STEPS, BATCH_SIZE]

    Returns:
                 [BATCH_SIZE]
    """

    diffs = pairwise_distance(u) - pairwise_distance(v)  # [NUM_STEPS, NUM_STEPS, BATCH_SIZE]

    input_tensor_sq = np.square(diffs)
    input_tensor_sq = input_tensor_sq #* weights

    norms = np.sqrt(np.reduce_sum(input_tensor_sq, axis=[1, 2]))  # [NUM_STEPS, NUM_STEPS, BATCH_SIZE]

    return norms


def pairwise_distance(u):
    """ Computes the pairwise distance (l2 norm) between all vectors in the tensor.

        Vectors are assumed to be in the third dimension. Op is done element-wise over batch.

    Args:
        u: [NUM_STEPS, BATCH_SIZE, NUM_DIMENSIONS]

    Returns:
           [NUM_STEPS, NUM_STEPS, BATCH_SIZE]

    """
    diffs = u - np.expand_dims(u, 1)  # [NUM_STEPS, NUM_STEPS, BATCH_SIZE, NUM_DIMENSIONS]
    reduction_indices = [3]

    input_tensor_sq = np.square(diffs)

    norms = np.sqrt(np.reduce_sum(input_tensor_sq, axis=reduction_indices))  # [NUM_STEPS, NUM_STEPS, BATCH_SIZE]

    return norms


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


# def geometric_unit(pred_coords, pred_torsions, bond_angles, bond_lens, start):
#     for i in range(3):
#         # coordinates of last three atoms
#         A, B, C = pred_coords[start-3], pred_coords[start-2], pred_coords[start-1]
#
#         # internal coordinates
#         T = bond_angles[i]
#         R = bond_lens[i]
#         P = pred_torsions[int(start / 3)]
#         # print(P.shape)
#
#         # 6x3 one triplet for each sample in the batch
#         D2 = torch.stack([-R * torch.ones(P.size()) * torch.cos(T),
#                           R * torch.cos(P) * torch.sin(T),
#                           R * torch.sin(P) * torch.sin(T)], dim=1)
#         # print(D2.shape)
#
#         # bsx3 one triplet for each sample in the batch
#         BC = C - B
#         bc = BC / torch.norm(BC, 2, dim=0, keepdim=True)
#
#         AB = B - A
#
#         N = torch.cross(AB, bc)
#         n = N / torch.norm(N, 2, dim=0, keepdim=True)
#
#         M = torch.stack([bc, torch.cross(n, bc), n], dim=1)
#         # print(M.shape)
#
#         D = torch.matmul(M, D2).squeeze() + C
#         pred_coords = torch.cat([pred_coords, D], dim=1)
#         print(pred_coords)
#
#     return pred_coords
