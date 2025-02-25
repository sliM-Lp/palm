import torch
import os
from scipy.cluster.vq import kmeans2
# from sklearn.cluster import kmeans_plusplus
# from sklearn.cluster import SpectralClustering
import sys


alphabet_size = int(sys.argv[1])

scop_embeddings_path = '../data/SCOPe40_embeddings_ProtT5.pt'
embeddings_dict = torch.load(scop_embeddings_path, map_location=torch.device('cpu'))
print(f"Landscape of {len(embeddings_dict)} proteins")

standard_embedding_single = dict()
standard_embedding_land = dict()

sum_embeddings = torch.zeros(1024)
sum_squares_embeddings = torch.zeros(1024)
total_count = 0

for p in embeddings_dict:
    channel_sum = embeddings_dict[p].sum(dim=0)
    channel_squared_sum = (embeddings_dict[p] ** 2).sum(dim=0)
    length = embeddings_dict[p].size(0)

    sum_embeddings += channel_sum
    sum_squares_embeddings += channel_squared_sum
    total_count += length

    single_mean = channel_sum/length
    single_std = torch.sqrt( (channel_squared_sum/length) - (single_mean ** 2))
    
    standard_embedding_single[p] = (embeddings_dict[p] - single_mean)/single_std

mean_per_channel = sum_embeddings / total_count
print(f'Average {mean_per_channel}')
# Compute the variance for each channel
variance_per_channel = (sum_squares_embeddings / total_count) - (mean_per_channel ** 2)
# Compute the standard deviation for each channel
std_per_channel = torch.sqrt(variance_per_channel)
print(f'Standard deviation {std_per_channel}')


for p in embeddings_dict:
    standard_embedding_land[p] = (embeddings_dict[p] - mean_per_channel)/std_per_channel


tensor_list = [x for x in standard_embedding_land.values()]

### make sure to keep the order
tensor_list = list()
sequences_ids = list()
for x in standard_embedding_land:
    tensor_list.append(standard_embedding_land[x])
    sequences_ids.append(x)


data_landscape = torch.cat(tensor_list, dim=0)
print(f'Number of residues: {data_landscape.shape[0]}')

codebook,labels = kmeans2(data_landscape, alphabet_size, minit='points')

alphabets_folder = '/scicore/home/schwede/pantol0000/repositories/alphabeta_classic/alphabets'
alphabet_path = os.path.join(alphabets_folder, f'ProtT5/kmeans_{alphabet_size}_std')

if not os.path.exists(alphabet_path):
    os.mkdir(alphabet_path)

codebook_path = os.path.join(alphabet_path,'codebook.pt')
torch.save(codebook, codebook_path)

conversion_path = os.path.join(alphabets_folder, f'kmeans_{alphabet_size}')

alphabeta = dict()
start = 0
for p in sequences_ids :
    seq_length = embeddings_dict[p].shape[0]
    alphabeta[p] = labels[start:start+seq_length]
    start+= seq_length


fasta_path = os.path.join(alphabet_path,'alphabeta.pt')
torch.save(alphabeta, fasta_path)


