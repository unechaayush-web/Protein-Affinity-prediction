import os
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.Data.IUPACData import protein_letters_3to1

pdb_folder = r"C:\Users\VVDN\Desktop\Data Scientics\Data\Protein"
parser = PDBParser(QUIET=True)
data = []
for file in os.listdir(pdb_folder):
    if file.endswith('.pdb'):
        path = os.path.join(pdb_folder, file)

        structure = parser.get_structure('protein', path)
        sequence = ""   # reset for each file

        for model in structure:
            for chain in model:
                for residue in chain:
                    if is_aa(residue):
                        resname = residue.get_resname().capitalize()
                        if resname in protein_letters_3to1:
                            sequence += protein_letters_3to1[resname]

        #  ONLY ONCE per file
        data.append({'file': file, 'sequence': sequence})
df = pd.DataFrame(data)
df.to_csv('protein_sequence.csv',index=False)
print('Sequence extraction completed')
