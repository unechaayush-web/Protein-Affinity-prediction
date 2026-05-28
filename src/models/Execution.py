print('Execution')
from fastapi import FastAPI
from embedding_service import get_protein_embedding
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, DataStructs
import numpy as np
import joblib
from Bio.Seq import Seq

model = joblib.load('Binding_prediction/Affinity_predictor.pkl')

app = FastAPI()

@app.post('/analyze-protein')
def analyze(sequence: str):   
    emb = get_protein_embedding(sequence)
    embedding = list(emb)

    if len(embedding) == 320:
        embedding.append(0.0)
    return embedding  # return actual embedding (not dict)


def smiles_info(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  
    else:
        return {
            'LogP': Descriptors.MolLogP(mol),
            'MolWt': Descriptors.MolWt(mol),
            'H_Donor': Descriptors.NumHDonors(mol),
            'H_Acceptor': Descriptors.NumHAcceptors(mol),
            'TPSA': Descriptors.TPSA(mol),
            'Rotatable_Bonds': Descriptors.NumRotatableBonds(mol),
            'Rings': rdMolDescriptors.CalcNumRings(mol)
        }


# fingerprint
def Fingerprint_generator(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    fp = AllChem.GetMorganFingerprintAsBitVect( 
        mol, radius=radius, nBits=n_bits
    )
    arr = np.zeros((n_bits,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def Smiles(smiles, sequence):
    info = smiles_info(smiles)
    if info is None:
        return "Invalid SMILES"

    descriptors = np.array(list(info.values())).reshape(1, -1)

    fingerprint = Fingerprint_generator(smiles)
    if fingerprint is None:
        return "Invalid SMILES"
    fingerprint = fingerprint.reshape(1, -1)

    protein_embedding = analyze(sequence)   # embedding array
    protein_embedding = np.array(protein_embedding).reshape(1, -1)

    # combine features
    features = np.concatenate((fingerprint, descriptors, protein_embedding), axis=1)

    prediction = model.predict(features)

    return prediction


if __name__ == '__main__':
    while True:
        print('Hello Im Ultron your Binding Affinity prediction model')
        User = input('Do you want to check affinity(Yes/Exit):')
        if User == 'Exit':
            print('Thanks for using ')
            break
        
        smiles = input('Enter your smile: ')
        protein = input('Enter your protein sequence (if not say Gene seq): ')
        
        if protein == 'Gene seq':
            gene = input('Enter gene seq: ')
            
            dna = Seq(gene)                 
            rna = dna.transcribe()           
            protein_1 = rna.translate()
            
            print('Binding Affinity is:\n')
            print(Smiles(smiles, str(protein_1)))
        else:
            print('Binding Affinity is:\n')
            print(Smiles(smiles, protein))
