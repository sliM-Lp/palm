#!/bin/bash
  
#SBATCH --job-name=ProtT5_SCOPe40
#SBATCH --cpus-per-task=6                 
#SBATCH --mem=80G                           
#SBATCH --qos=gpu1day
#SBATCH --output=outputs/ProtT5_SCOPe40.out
#SBATCH --error=errors/ProtT5_SCOPe40.err
#SBATCH --partition=a100-80g    
#SBATCH --gres=gpu:1
#SBATCH --reservation=schwede


ml purge
source ~/.bashrc
mamba activate eba
python3 /scicore/home/schwede/pantol0000/repositories/alphabeta_classic/scripts/compute_embeddings.py
mamba deactivate