#! /bin/bash

#SBATCH --job-name=mergepthat
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=10
#SBATCH --partition=std
#SBATCH --time=24:00:00
#SBATCH --array=1-20
#SBATCH --output=/storage/u/alice/AnalysisResults/slurm-%A_%a.out

srun merge_LHC19f4.sh $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID