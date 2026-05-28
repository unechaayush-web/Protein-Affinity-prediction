from rdkit import Chem
import os
import pandas as pd

sdf_folder = r'C:\Users\VVDN\Desktop\Data Scientics\Data\ligand'
smiles_list = []

for file in os.listdir(sdf_folder):
    if file.endswith('.sdf'):
        path = os.path.join(sdf_folder,file)
        supplier = Chem.SDMolSupplier(path)

        for mol in supplier:
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)

        smiles_list.append({'file':file,'smiles':smiles})

df = pd.DataFrame(smiles_list)
df.to_csv('smiles_dataset.csv',index=False)
print('Conversion completed')
