from transformers import T5Tokenizer, T5EncoderModel
import torch
import sys
import re
import os

data_dir = sys.argv[1]
fasta_file = sys.argv[2]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#_half
# Load the tokenizer"Rostlab/prot_t5_xl_uniref50"
tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50-enc', local_files_only=True, do_lower_case=False)

# Load the model
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50-enc", local_files_only=True).to(device)

# only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
# model.to(torch.float32) if device==torch.device("cpu")

# prepare your protein sequences as a list
sequences = list()
protein_ids = list()
fasta_path = os.path.join(data_dir,fasta_file)
with open(fasta_path, 'r') as file:
    for line in file:
        if line[0]=='>':
            protein_ids.append(line[1:].strip())
        else:
            sequences.append(line.strip())


# replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]
n_seq = len(sequences)
print(f'{n_seq} sequences to embed')

K=50
iterations = 1 + int(n_seq/K)
embeddings = dict()
### convert them in blocks of K
for i in range(iterations):
    print(i*K)
    block_sequences = sequences[i*K:(i+1)*K]
    block_protein_ids = protein_ids[i*K:(i+1)*K]
    # tokenize sequences and pad up to the longest sequence in the batch
    ids = tokenizer(block_sequences, add_special_tokens=True, padding="longest")

    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    # generate embeddings
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

    embeddings_path = os.path.join(data_dir, 'SCOPe40_embeddings_ProtT5.pt')

    
    for idx,j in enumerate(block_protein_ids):
        embeddings[j] = embedding_repr.last_hidden_state[idx,:len(block_sequences[idx])]


    torch.save(embeddings,embeddings_path)

