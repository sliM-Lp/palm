#!/bin/bash
  
#SBATCH --job-name=128Kmean2_SCOPe40
#SBATCH --cpus-per-task=6                 
#SBATCH --mem=80G                           
#SBATCH --qos=6hours
#SBATCH --output=outputs/128Kmean2_SCOPe40.out
#SBATCH --error=errors/128Kmean2_SCOPe40.err
# SBATCH --partition=a100-80g    
# SBATCH --gres=gpu:1
# SBATCH --reservation=schwede


ml purge
source ~/.bashrc
mamba activate alphabeta
python3 /scicore/home/schwede/pantol0000/repositories/alphabeta_classic/scripts/kmean2_clustering.py 128
mamba deactivate