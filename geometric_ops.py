import collections

import numpy as np
import torch

# Constants
from numpy.core.multiarray import ndarray

NUM_DIMENSIONS = 3
NUM_DIHEDRALS = 3
BOND_LENGTHS = np.array([145.801, 152.326, 132.868], dtype='float32')
BOND_ANGLES = np.array([2.124, 1.941, 2.028], dtype='float32')


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

    return np.arctan2(y_coords, x_coords)


def calculate_coordinates(dihedral, coordinates, r=BOND_LENGTHS, theta=BOND_ANGLES):
    # type: (ndarray, ndarray, ndarray, ndarray) -> (ndarray)
    """ Takes triplets of dihedral angles (omega, phi, psi) and returns 3D points ready for use in
        reconstruction of coordinates. Bond lengths and angles are based on idealized averages.

    Args:
        dihedral: [NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]

    Returns:
        [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS]
    """

    num_steps = dihedral.shape[0]
    batch_size = dihedral.shape[1]  # important to use get_shape() to keep batch_size fixed for performance reasons

    r_cos_theta = r * np.cos(np.pi - theta)  # [NUM_DIHEDRALS]
    r_sin_theta = r * np.sin(np.pi - theta)  # [NUM_DIHEDRALS]

    pt_x = np.tile(r_cos_theta.reshape((1, 1, -1)),
                   (num_steps, batch_size, 1))  # [NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]
    pt_y = np.multiply(np.cos(dihedral), r_sin_theta)  # [NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]
    pt_z = np.multiply(np.sin(dihedral), r_sin_theta)  # [NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]

    pt = np.stack([pt_x, pt_y, pt_z])  # [NUM_DIMS, NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]
    pt_perm = pt.transpose(axes=[1, 3, 2, 0])  # [NUM_STEPS, NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMS]
    pt = pt_perm.reshape([num_steps * NUM_DIHEDRALS, batch_size, NUM_DIMENSIONS])
    # [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMS]

    # Convert points to coordinats
    s = pt.shape[0]
    num_frags = num_steps
    Triplet = collections.namedtuple('Triplet', 'a, b, c')
    init_mat = np.array([[-np.sqrt(1.0 / 2.0), np.sqrt(3.0 / 2.0), 0], [-np.sqrt(2.0), 0, 0], [0, 0, 0]],
                        dtype='float32')
    init_coords = Triplet(*[np.reshape(np.tile(row[np.newaxis], np.stack([num_frags * batch_size, 1])),
                                       [num_frags, batch_size, NUM_DIMENSIONS]) for row in init_mat])
    # NUM_DIHEDRALS x [NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS]

    pt = pt.reshape([num_frags, -1, batch_size,
                    NUM_DIMENSIONS])  # [NUM_FRAGS, FRAG_SIZE,  BATCH_SIZE, NUM_DIMENSIONS]
    pt = pt.transpose(axes=[1, 0, 2, 3])

    # extension function used for single atom reconstruction and whole fragment alignment
    def extend(tri, pt, multi_m):
        # type: (Triplet, ndarray, bool) -> (ndarray)
        """
        Args:
            tri: NUM_DIHEDRALS x [NUM_FRAGS/0,         BATCH_SIZE, NUM_DIMENSIONS]
            pt:                  [NUM_FRAGS/FRAG_SIZE, BATCH_SIZE, NUM_DIMENSIONS]
            multi_m: bool indicating whether m (and tri) is higher rank. pt is always higher rank; what changes is what the first rank is.
        """
        normaliztor = torch.nn.BatchNorm1d(batch_size)
        bc = normaliztor(tri.c - tri.b)  # [NUM_FRAGS/0, BATCH_SIZE, NUM_DIMS]
        n = normaliztor(np.cross(tri.b - tri.a, bc))  # [NUM_FRAGS/0, BATCH_SIZE, NUM_DIMS]
        if multi_m:  # multiple fragments, one atom at a time.
            m = np.transpose(np.stack([bc, np.cross(n, bc), n]), axes=[1, 2, 3, 0])  # [NUM_FRAGS,   BATCH_SIZE, NUM_DIMS, 3 TRANS]
        else:  # single fragment, reconstructed entirely at once.
            # WARNING possibly wrong
            s = np.pad([[0, 1]], pt.shape)  # FRAG_SIZE, BATCH_SIZE, NUM_DIMS, 3 TRANS
            m = np.transpose(np.stack([bc, np.cross(n, bc), n]), axes=[1, 2, 0])  # [BATCH_SIZE, NUM_DIMS, 3 TRANS]
            m = np.reshape(np.tile(m, [s[0], 1, 1]), s)  # [FRAG_SIZE, BATCH_SIZE, NUM_DIMS, 3 TRANS]
        coord = np.add(np.squeeze(np.matmul(m, np.expand_dims(pt, 3)), axis=3), tri.c)  # [NUM_FRAGS/FRAG_SIZE, BATCH_SIZE, NUM_DIMS]
        return coord

    # loop over FRAG_SIZE in NUM_FRAGS parallel fragments, sequentially generating the coordinates for each fragment across all batches
    i = 0
    s_padded = pt.shape[0]  # FRAG_SIZE
    coords_np = np.empty(s_padded, dtype=np.float32)
    # FRAG_SIZE x [NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS]

    def loop_extend(i, tri, coords_ta):  # FRAG_SIZE x [NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS]
        coord = extend(tri, pt[i], True)
        coords_ta[i] = coord
        return [i + 1, Triplet(tri.b, tri.c, coord), coords_ta]

    tri = init_coords[0]
    tris = coords_np.__deepcopy__()
    while i < s_padded:
        i, tri, coords_np = loop_extend(i, tri, coords_np)
        tris[i-1] = tri
    tris = tri
    # NUM_DIHEDRALS x [NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS],
    # FRAG_SIZE x [NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS]

    # loop over NUM_FRAGS in reverse order, bringing all the downstream fragments in alignment with current fragment
    coords_pretrans = np.transpose(coords_np,
                                   axes=[1, 0, 2, 3])  # [NUM_FRAGS, FRAG_SIZE, BATCH_SIZE, NUM_DIMENSIONS]
    i = coords_pretrans.shape[0] - 2  # NUM_FRAGS

    def loop_trans(i, coords):
        transformed_coords = extend(Triplet(*[di[i] for di in tris]), coords, False)
        return [i - 1, np.concat([coords_pretrans[i], transformed_coords], 0)]

    coords_trans = np.empty(i, dtype=np.float32)
    while i > -1:
        i, coord = loop_trans(i, coords_pretrans[-1])
        coords_trans[i + 1] = coord
    # [NUM_FRAGS x FRAG_SIZE, BATCH_SIZE, NUM_DIMENSIONS]

    # lose last atom and pad from the front to gain an atom ([0,0,0], consistent with init_mat), to maintain correct atom ordering
    coords = np.pad(coords_trans[:s - 1], [[1, 0], [0, 0], [0, 0]])  # [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS]

    return coords
