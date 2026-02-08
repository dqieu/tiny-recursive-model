#!/bin/bash -l
#SBATCH --job-name=trm
#SBATCH --chdir=./  # Set the working directory
#SBATCH --output=slurm_logs/%x_%j_out.txt     # Output file (%x=job-name, %j=job-id)
#SBATCH --error=slurm_logs/%x_%j_err.txt      # Error file
#SBATCH --time=3tra:00:00             # Time limit (hh:mm:ss)
#SBATCH --cpus-per-task=24          # Number of CPU cores per task
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                  # Number of tasks
#SBATCH --gpus-per-node=1         # Number of GPUs per node
#SBATCH --mail-type=BEGIN,END,FAIL        # Optional: email on job end/failure
#SBATCH --mail-user=dat.kieu@mail.utoronto.ca  # Your email address

module load StdEnv/2023 gcc arrow python/3.12

source ~/.trm/bin/activate

nvidia-smi --query-gpu=timestamp,index,power.draw,memory.used \
           --format=csv,noheader -l 1 > "./exps/{$SLURM_JOB_ID}.gpu.csv" &
LOGGER_PID=$!

OUTDIR="./exps/${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"

python -m playground -p /home/kieu/links/scratch/speech-health-ai/tmp/torgo/wavlm-basep-last/cache.hdf5 \
                        -e 1685 \
                        -b 16 \
                        --epochs 150 \
                        -o $OUTDIR

kill $LOGGER_PID