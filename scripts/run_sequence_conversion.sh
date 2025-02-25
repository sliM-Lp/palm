#!/bin/bash
  
#SBATCH --job-name=ProtT5_emb
#SBATCH --cpus-per-task=6                 
#SBATCH --mem=80G                           
#SBATCH --qos=gpu6hours
#SBATCH --output=outputs/ProtT5_emb.out
#SBATCH --error=errors/ProtT5_emb.err
#SBATCH --partition=a100-80g    
#SBATCH --gres=gpu:1
#SBATCH --reservation=schwede


ml purge
source ~/.bashrc
mamba activate eba
python3 /scicore/home/schwede/pantol0000/repositories/alphabeta_classic/scripts/sequences_to_pLM_embeddings.py /scicore/home/schwede/pantol0000/repositories/alphabeta_classic/data SCOPe40.fasta
mamba deactivate 