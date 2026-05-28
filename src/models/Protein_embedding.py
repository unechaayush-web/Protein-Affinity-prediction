import pandas as pd
import torch
import esm
from tqdm import tqdm

df = pd.read_csv('Data\protein_sequence.csv')
sequences = df['sequence'].tolist()
model,alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.evel()

def get_protein_embedding(seq):
    data = [('protein',seq)]
    batch_label,batch_strs,batch_tokens = batch_converter(data)

    with torch.no_grad():
        results = model(batch_tokens)

    embeddings = results['representations'][6]

    protein_vector = embeddings.mean(1).squeeze().numpy()
    return protein_vector

all_embeddings = []
for seq in tqdm(sequences):
    vec = get_protein_embedding(seq)
    all_embeddings.append(vec)

embedding_df = pd.DataFrame(all_embeddings)

embedding_df.columns =[f"feat_{i}"for i in range(embedding_df.shpe[1])]

final_df = pd.concat([df,embedding_df],axis=1)

final_df.to_csv('protein_embedding.csv',index=False)
print('Done')
