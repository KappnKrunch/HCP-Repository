#!/bin/bash
#SBATCH --job-name=test
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --output=out.log
#SBATCH --error=err.log
#SBATCH --partition=gpu --nodelist=gpu09       

module load miniconda3/base
source dMRI/bin/activate

jupyter lab --no-browser --port=8888

