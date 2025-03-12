#!/bin/bash -l
#SBATCH --job-name=train_afnocast
#SBATCH --time=01:00:00
#SBATCH --account=s1144
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --environment=modulus
#SBATCH --partition=normal
#SBATCH --cpus-per-task=72

cd $SCRATCH
source modulus-venv/bin/activate
export PYTHONPATH=/capstor/scratch/cscs/acarpent/GeoSatCast/

/capstor/scratch/cscs/acarpent/run_tests.sh