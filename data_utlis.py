import os
import numpy as np
from io import StringIO

import torch
from torch import LongTensor
from torch.autograd import Variable
from torch.nn import Embedding
from tqdm import tqdm
import bcolz
from torch.utils.data.dataset import Dataset

residue_letter_codes = {'GLY': 'G','PRO': 'P','ALA': 'A','VAL': 'V','LEU': 'L',
                        'ILE': 'I','MET': 'M','CYS': 'C','PHE': 'F','TYR': 'Y',
                        'TRP': 'W','HIS': 'H','LYS': 'K','ARG': 'R','GLN': 'Q',
                        'ASN': 'N','GLU': 'E','ASP': 'D','SER': 'S','THR': 'T'}

aa2ix= {'G': 0,'P': 1,'A': 2,'V': 3,'L': 4,
          'I': 5,'M': 6,'C': 7,'F': 8,'Y': 9,
          'W': 10,'H': 11,'K': 12,'R': 13,'Q': 14,
          'N': 15,'E': 16,'D': 17,'S': 18,'T': 19}

def load_data(pn_path):
    # type: (String) -> ndarray
    """
    Loads the data from the given path.

    Args:
        pn_path: path to the data

    Returns:
        np.array of size len(ids), each element is a tuple of (id: Any, seq: [sequence_length], pssm: [], xyz: [], msk_idx: [])
    """
    ids = []
    seqs = []
    evs = []
    coords = []
    masks = ['init', '/n']
    id_next, pri_next, ev_next, ter_next, msk_next = False, False, False, False, False
    with open(pn_path + 'text_sample') as fp:
        for line in tqdm(iter(fp.readline, '')):
            if id_next:
                ids.append(line[:-1])
            elif pri_next:
                seqs.append(line[:-1])
            elif ev_next:
                evs.append(np.genfromtxt(StringIO(line)))
            elif ter_next:
                coords.append(np.genfromtxt(StringIO(line)))
            elif msk_next:
                masks.append(line[:-1])

            if np.core.defchararray.find(line, "[ID]", end=5) != -1:
                id_next = True
                masks.pop()
                masks.pop()
                pri_next, ev_next, ter_next, msk_next = False, False, False, False
            elif np.core.defchararray.find(line, "[PRIMARY]", end=10) != -1:
                pri_next = True
                ids.pop()
                id_next, ev_next, ter_next, msk_next = False, False, False, False
            elif np.core.defchararray.find(line, "[EVOLUTIONARY]", end=15) != -1:
                ev_next = True
                seqs.pop()
                id_next, pri_next, ter_next, msk_next = False, False, False, False
            elif np.core.defchararray.find(line, "[TERTIARY]", end=11) != -1:
                ter_next = True
                evs.pop()
                id_next, pri_next, ev_next, msk_next = False, False, False, False
            elif np.core.defchararray.find(line, "[MASK]", end=7) != -1:
                msk_next = True
                coords.pop()
                id_next, pri_next, ev_next, ter_next = False, False, False, False

    pssm = evs
    xyz = coords
    data = np.empty(len(ids), dtype=object)

    # loop through each evolutionary section
    for i in range(len(ids)):
        # first store the id and sequence
        id = ids[i]
        seq = seqs[i]

        # next get the PSSM matrix for the sequence
        sp = 21 * i
        ep = 21 * (i + 1)
        psi = np.array(pssm[sp:ep])
        pssmi = np.stack([p for p in psi], axis=1)

        # then get the coordinates
        sx = 3 * i
        ex = 3 * (i + 1)
        xi = np.array(xyz[sx:ex])
        xyzi = np.stack([c for c in xi], axis=1) / 100  # have to scale by 100 to match PDB

        # lastly convert the mask to indices
        msk_idx = np.where(np.array(list(masks[i])) == '+')[0]

        # bracket id or get "setting an array element with a sequence"
        zt = np.array([[id], seq, pssmi, xyzi, msk_idx])
        data[i] = zt
    return data


def encode_protein_padded(sequence, max_len, protein_names=residue_letter_codes):
    vocab = ['<pad>'] + sorted(set([char for char in residue_letter_codes.values()]))
    print(vocab)
    vectorized_seq = [vocab.index(tok) for tok in sequence]
    embed = Embedding(len(vocab), len(vocab))
    seq_tensor = Variable(torch.zeros(max_len)).long()
    seq_tensor[:len(vectorized_seq)] = LongTensor(seq_tensor)
    return embed(seq_tensor).detach().numpy()


class ProteinNetDataset(Dataset):
    def __init__(self, proteinnet_path):
        self.data = load_data(proteinnet_path)
        self.lens = LongTensor(list(map(lambda x: len(x[1]), self.data)))
        self.max_len = self.lens.max()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name, sequence, pssm, coords, mask = self.data[idx]
        length = len(sequence)
        sequence_vec = encode_protein_padded(sequence, self.max_len)
        seq_pssm = np.concatenate([sequence_vec, pssm], axis=1)

        sample = {'name': name,
                  'sequence': seq_pssm,
                  'coords': coords,
                  'length': length,
                  'mask': mask
                  }

        return sample

