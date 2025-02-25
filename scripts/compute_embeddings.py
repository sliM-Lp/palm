import os
# from esm.models.esmc import ESMC
# from esm.sdk.api import ESMProtein, LogitsConfig
import torch
from eba import plm_extractor as plm


working_dir = '/scicore/home/schwede/pantol0000/repositories/alphabeta_classic'
scop_fasta_file = os.path.join(working_dir, 'data/SCOPe40.fasta')
# output_path_emb = os.path.join(working_dir, 'data/SCOPe40_embeddings_ESMc.pt')
# output_path_logits = os.path.join(working_dir, 'data/SCOPe40_logits_ESMc.pt')
output_path_emb = os.path.join(working_dir, 'data/SCOPe40_embeddings_ProtT5.pt')


### load language model extractor: ProtT5 or ESMb1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
protT5_ext = plm.load_extractor('ProtT5', 'residue', device=device)


scop_sequences = dict()
with open(scop_fasta_file, 'r') as file:
    seq_id = ''
    for line in file:
        if line[0]=='>':
            seq_id = line[1:].strip()
            scop_sequences[seq_id] = ''
        else:
            scop_sequences[seq_id] = line.strip()

# if torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")

# client = ESMC.from_pretrained("esmc_300m").to(device)

embeddings = dict()
# logits = dict()
for i in scop_sequences:
    # protein = ESMProtein(scop_sequences[i])
    # protein_tensor = client.encode(protein) 
    # logits_output = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))

    # logits[i] = logits_output.logits
    # embeddings[i] = logits_output.embeddings.squeeze()

    embeddings[i] = protT5_ext.extract(scop_sequences[i])

    if len(scop_sequences[i])!=embeddings[i].shape[0]:
        print(f'Length missmatch: {i}, {scop_sequences[i]}-{embeddings[i].shape[0]}')

# print(f'Stored: {len(embeddings)}, {len(logits)}.')

torch.save(embeddings, output_path_emb)
# torch.save(logits, output_path_logits)

