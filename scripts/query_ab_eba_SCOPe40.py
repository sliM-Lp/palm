import os
import sys
import torch
from scipy import spatial
from matplotlib import pyplot as plt
from eba import methods 
from eba import score_matrices as sm
from eba import plm_extractor as plm

alphabet_size = int(sys.argv[1])
# pLM = sys.argv[2]
query_id = sys.argv[2]

pLM = 'ProtT5'

working_dir = '/scicore/home/schwede/pantol0000/repositories/alphabeta_classic'
scop_lookup_file = os.path.join(working_dir, 'data/scop_lookup.fix.tsv')
scop_fasta_file = os.path.join(working_dir, 'data/SCOPe40.fasta')

###load embeddings and new codebook
alphabet_path = f'/scicore/home/schwede/pantol0000/repositories/alphabeta_classic/alphabets/{pLM}/kmeans_{alphabet_size}'
codebook = torch.load(f'{alphabet_path}/codebook.pt')
ab_sequences = torch.load(f'{alphabet_path}/alphabeta.pt')

###load std embeddings
alphabet_path = f'/scicore/home/schwede/pantol0000/repositories/alphabeta_classic/alphabets/{pLM}/kmeans_{alphabet_size}_std'
codebook_std = torch.load(f'{alphabet_path}/codebook.pt')
ab_sequences_std = torch.load(f'{alphabet_path}/alphabeta.pt')

###build embeddings
query_list_embedding = [torch.tensor(codebook[x]).unsqueeze(dim=0) for x in ab_sequences[query_id]]
query_embeddings = torch.cat(query_list_embedding , dim=0)

alphabeta_embeddings = dict()
for s in ab_sequences:
    emb_list = [torch.tensor(codebook[x]).unsqueeze(dim=0) for x in ab_sequences[s]]
    alphabeta_embeddings[s] = torch.cat(emb_list, dim=0)


###build embeddings std
query_list_embedding_std = [torch.tensor(codebook_std[x]).unsqueeze(dim=0) for x in ab_sequences_std[query_id]]
query_embeddings_std = torch.cat(query_list_embedding_std , dim=0)

alphabeta_std_embeddings = dict()
for s in ab_sequences_std:
    emb_list = [torch.tensor(codebook_std[x]).unsqueeze(dim=0) for x in ab_sequences_std[s]]
    alphabeta_std_embeddings[s] = torch.cat(emb_list, dim=0)


cosine = dict()
cosine_std = dict()
for seq in ab_sequences:
    if seq!=query_id:

        cos_sim_matrix = sm.compute_cosine_similarity_matrix_plain(query_embeddings, alphabeta_embeddings[seq])
        cosine[seq] = methods.compute_eba(cos_sim_matrix, gap_open_penalty=0.0, gap_extend_penalty=0.0)

        #compute std
        cos_sim_matrix = sm.compute_cosine_similarity_matrix_plain(query_embeddings_std, alphabeta_std_embeddings[seq])
        cosine_std[seq] = methods.compute_eba(cos_sim_matrix, gap_open_penalty=0.0, gap_extend_penalty=0.0)



results_path = os.path.join(working_dir,'SCOP40_benchmark_results',f'kmean_{alphabet_size}/{pLM}/cosine')
if not os.path.exists(results_path):
    os.mkdir(results_path)

results_file = os.path.join(results_path, f'{query_id}.pt')
torch.save(cosine,results_file)

#### std
results_path = os.path.join(working_dir,'SCOP40_benchmark_results',f'kmean_{alphabet_size}/{pLM}/cosine_std')
if not os.path.exists(results_path):
    os.mkdir(results_path)

results_file = os.path.join(results_path, f'{query_id}.pt')
torch.save(cosine_std,results_file)







