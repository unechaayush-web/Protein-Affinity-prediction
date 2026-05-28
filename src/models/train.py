print('training model')
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, DataStructs
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib


# Load data
data_smiles = pd.read_csv('Data/smiles_dataset.csv')
data_protein = pd.read_csv('Data/protein_sequence.csv')

df = pd.DataFrame(data_smiles)

# Clean data
df_clean = df.dropna(subset=['smiles'])

# Molecular Descriptors
def smiles_info(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    return {
        'LogP': Descriptors.MolLogP(mol),
        'MolWt': Descriptors.MolWt(mol),
        'H_Donor': Descriptors.NumHDonors(mol),
        'H_Acceptor': Descriptors.NumHAcceptors(mol),
        'TPSA': Descriptors.TPSA(mol),
        'Rotatable_Bonds': Descriptors.NumRotatableBonds(mol),
        'Rings': rdMolDescriptors.CalcNumRings(mol)
    }

# Fingerprints
def fingerprint_generator(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius, nBits=n_bits
    )
    
    arr = np.zeros((n_bits,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# Apply features

# Descriptors
desc_df = df_clean['smiles'].apply(smiles_info)
desc_df = pd.DataFrame(desc_df.tolist())

# Fingerprints
fp_array = df_clean['smiles'].apply(fingerprint_generator)
fp_df = pd.DataFrame(fp_array.tolist())
fp_df.columns = [f'fp_{i}' for i in range(fp_df.shape[1])]

# Combine all features
ligand_features = pd.concat([df_clean.reset_index(drop=True),
                             desc_df,
                             fp_df], axis=1)

print(ligand_features.head())

protein_df = pd.read_csv('Data/protein_embeddings.csv')

pair_df =pd.read_csv('Data/pair.csv')# Contain 
data = pair_df.merge(protein_df,on='file')
data_2 = data.merge(ligand_features,on='smiles')

print(data_2.head(4))

X = data_2.drop(columns=['Affinity_x','smiles','file_x','sequence_x','sequence_y'])
X = X.select_dtypes(include=[np.number])
y = data_2['Affinity_x']

print(X.shape)
print(y.shape)
print(X.dtypes)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

Scaler = StandardScaler()
X_Scaled = Scaler.fit_transform(X_train)
X_test_scale = Scaler.transform(X_test)

model = XGBRegressor(n_estimators=50,learning_rate=0.1,max_depth=3)
model.fit(X_Scaled,y_train)
y_pred = model.predict(X_test_scale)

cross_validation = cross_val_score(model,X,y,cv=5,scoring='accuracy')

MSE = mean_squared_error(y_test,y_pred)
MAE = mean_absolute_error(y_test,y_pred)
print('MSE:',MSE)
print('MAE:',MAE)
print('Avg score:',cross_validation.mean())

joblib.dump(model,'Affinity_predictor.pkl')
print('Model safed succesfully')
