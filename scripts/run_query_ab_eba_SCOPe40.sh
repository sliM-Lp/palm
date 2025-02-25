#!/bin/bash
  
#SBATCH --job-name=ab_SCOPe40
#SBATCH --cpus-per-task=6                 
#SBATCH --mem=80G                           
#SBATCH --qos=1day
#SBATCH --output=outputs/ab_SCOPe40.out
#SBATCH --error=errors/ab_SCOPe40.err


ml purge
source ~/.bashrc
mamba activate eba
python3 /scicore/home/schwede/pantol0000/repositories/alphabeta_classic/scripts/query_ab_eba_SCOPe40.py 20 d1dlwa_
mamba deactivate