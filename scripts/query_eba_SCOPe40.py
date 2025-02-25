import os
import sys
import torch
from scipy import spatial
from matplotlib import pyplot as plt
from eba import methods 
from eba import score_matrices as sm
from eba import plm_extractor as plm

query_id = sys.argv[1]
working_dir = '/scicore/home/schwede/pantol0000/repositories/alphabeta_classic'

###load embeddings and new codebook
embeddings_path = os.path.join(working_dir, 'data/SCOPe40_embeddings_ProtT5.pt')
embeddings = torch.load(embeddings_path, map_location=torch.device('cpu'))

###build embeddings
query_embeddings = embeddings[query_id]

eba_scores = dict()
for seq in embeddings:
    if seq!=query_id:
        # sim_matrix = sm.compute_similarity_matrix(query_embeddings, embeddings[seq])
        sim_matrix = sm.compute_cosine_similarity_matrix_plain(query_embeddings, embeddings[seq])
        eba_scores[seq] = methods.compute_eba(sim_matrix)

results_path = os.path.join(working_dir,'SCOP40_benchmark_results','EBA_cosine/ProtT5')
if not os.path.exists(results_path):
    os.mkdir(results_path)

results_file = os.path.join(results_path, f'{query_id}.pt')
torch.save(eba_scores,results_file)


