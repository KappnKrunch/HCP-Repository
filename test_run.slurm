#!/bin/bash
#SBATCH --job-name=test
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --time=10:00:00
#SBATCH --output=out.log
#SBATCH --error=err.log
#SBATCH --partition=DCU ##--nodelist=gpu09       

module load miniconda3/base
source dMRI/bin/activate

export PYTHONUNBUFFERED=TRUE

python -u /home1/mforbush/grips2024/code/loadDataTest.py

