#!/bin/bash
  
#SBATCH --job-name=load_clusters
#SBATCH --cpus-per-task=6                 
#SBATCH --mem=80G                           
#SBATCH --qos=gpu6hours
#SBATCH --output=outputs/load_clusters.out
#SBATCH --error=errors/load_clusters.err
#SBATCH --partition=a100-80g    
#SBATCH --gres=gpu:1
# # SBATCH --reservation=schwede


ml purge
source ~/.bashrc
mamba activate eba
python3 /scicore/home/schwede/pantol0000/repositories/alphabeta_classic/scripts/load_clusters.py
mamba deactivate 