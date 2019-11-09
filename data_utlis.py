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
    with open(pn_path) as fp:
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
        msk_idx = np.array(list(map(lambda x: x[0] if x[1] == '+' else 0, enumerate(masks[i]))))

        # bracket id or get "setting an array element with a sequence"
        zt = np.array([[id], seq, pssmi, xyzi, msk_idx])
        data[i] = zt
    return data


def encode_protein_padded(sequence, max_len, protein_names=residue_letter_codes):
    vocab = ['<pad>'] + sorted(set([char for char in residue_letter_codes.values()]))

    vectorized_seq = [vocab.index(tok) for tok in sequence]
    embed = Embedding(len(vocab), len(vocab))
    seq_tensor = Variable(torch.zeros((1, max_len))).long()
    print(seq_tensor.shape)
    seq_tensor[0, :len(vectorized_seq)] = LongTensor(seq_tensor)
    return embed(seq_tensor).detach().numpy()[0]


def pad_and_embed(data):
    seqs = list(map(lambda x: x[1], data))
    pssms = list(map(lambda x: x[2], data))
    coords = list(map(lambda x: x[3], data))
    mask = list(map(lambda x: x[4], data))

    vocab = ['<pad>'] + sorted(set([char for char in residue_letter_codes.values()]))
    vectorized_seqs = [[vocab.index(tok) for tok in seq] for seq in seqs]

    embed = Embedding(len(vocab), len(vocab))

    seq_lengths = LongTensor(list(map(len, vectorized_seqs)))
    seq_tensor = Variable(torch.zeros((len(vectorized_seqs), seq_lengths.max()))).long()
    extended_pssm = np.zeros((len(vectorized_seqs), seq_lengths.max(), pssms[0].shape[1]))
    extended_coords = np.zeros((len(vectorized_seqs), seq_lengths.max()*3, 3))
    extended_mask = np.zeros((len(vectorized_seqs), seq_lengths.max()))

    for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seqlen] = LongTensor(seq)
        extended_pssm[idx, :seqlen] = pssms[idx]
        extended_coords[idx, :seqlen*3] = coords[idx]
        extended_mask[idx, :seqlen] = mask[idx]

    # sort data and new encoded sequences by length, possibly not needed
    # seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    # seq_tensor = seq_tensor[perm_idx]
    # data = data[perm_idx]

    embedded_seq_tensor = embed(seq_tensor)
    for idx, (seq, seqlen) in enumerate(zip(embedded_seq_tensor, data)):
        data[idx][1] = embedded_seq_tensor[idx].detach().numpy()
        data[idx][2] = extended_pssm[idx]
        data[idx][3] = extended_coords[idx]
        data[idx][4] = extended_mask[idx]
    return data


class ProteinNetDataset(Dataset):
    def __init__(self, proteinnet_path):
        self.data = pad_and_embed(load_data(proteinnet_path))
        self.lens = LongTensor(list(map(lambda x: len(x[1]), self.data)))
        self.max_len = self.lens.max()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name, sequence, pssm, coords, mask = self.data[idx]
        length = len(sequence)
        #sequence_vec = encode_protein_padded(sequence, self.max_len)
        seq_pssm = np.concatenate([sequence, pssm], axis=1)

        sample = {'name': name,
                  'sequence': seq_pssm,
                  'coords': coords,
                  'length': length,
                  'mask': mask
                  }

        return sample

