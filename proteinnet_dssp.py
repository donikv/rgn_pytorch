import os
from pathlib import Path

from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB.StructureBuilder import StructureBuilder

from data_utlis import load_data

residue_letter_codes = {'GLY': 'G','PRO': 'P','ALA': 'A','VAL': 'V','LEU': 'L',
                        'ILE': 'I','MET': 'M','CYS': 'C','PHE': 'F','TYR': 'Y',
                        'TRP': 'W','HIS': 'H','LYS': 'K','ARG': 'R','GLN': 'Q',
                        'ASN': 'N','GLU': 'E','ASP': 'D','SER': 'S','THR': 'T'}

residue_letter_codes = {v: k for k, v in residue_letter_codes.items()}
atoms = ['CA', 'CB', 'N']

def proteinnet_to_pdb_file(data, file, id):
    with open(file, "w") as fp:
        seq = data[1]
        coord = data[3]
        lines = ["HEADER {0}\n".format(id)]
        line = "SEQRES 1 A {0}".format(len(seq))
        for idx, _ in enumerate(seq):
            name = residue_letter_codes[seq[idx]]
            line += " {0}".format(name)
        lines.append(line+'\n')
        for idx, _ in enumerate(seq):
            name = residue_letter_codes[seq[idx]]
            for i in range(3):
                coords = coord[3 * idx + i : 3 * idx + i + 3]
                print(coords)
                if len(coords) != 3:
                    break 
                lines.append("ATOM {0} {5} {1} A {6} {2:5.2f} {3:5.2f} {4:5.2f} 0 0 {7}\n".format(3*idx+i+1, name, coords[i][0], coords[i][1], coords[i][2], atoms[i], idx+1, atoms[i][0]))
        fp.write("END\n")
        fp.writelines(lines)
        fp.close()

if __name__ == '__main__':
    pn_test = os.curdir + '/../rgn_pytorch/data/text_sample'
    data = load_data(pn_test)
    for d in data:
        id = str(d[0][0]).replace('#','')
        pdb_file_name = pn_test+'_'+id+'.pdb'
        proteinnet_to_pdb_file(d, pdb_file_name, id)
        p = PDBParser(PERMISSIVE=0)
        # pdb_file_name = os.curdir + '/../rgn_pytorch/data/1mbs'
        structure = p.get_structure("1", pdb_file_name)
        header = p.get_header()
        print(header)
        model = structure[0]
        dssp = DSSP(model, pdb_file_name)
        # DSSP data is accessed by a tuple (chain_id, res_id)
        a_key = list(dssp.keys())[2]
        # (dssp index, amino acid, secondary structure, relative ASA, phi, psi,
        # NH_O_1_relidx, NH_O_1_energy, O_NH_1_relidx, O_NH_1_energy,
        # NH_O_2_relidx, NH_O_2_energy, O_NH_2_relidx, O_NH_2_energy)

