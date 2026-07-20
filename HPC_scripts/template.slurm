#!/bin/bash
#SBATCH --job-name=NAME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=PARTITION NAME
#SBATCH --time=7:30:00
#SBATCH --output=/home/USER_NAME/cardio_oncology_project/MIMIC-Cardio-Oncology/DIRECTORY_NAME/slurm_%j.out
#SBATCH --error=/home/USER_NAME/cardio_oncology_project/MIMIC-Cardio-Oncology/DIRECTORY_NAME/slurm_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=EMAIL

# ── environment ───────────────────────────────────────────────────────────────
module load python/3.12.5
module load cuda/12.6
source /home/NAME/cardio_oncology_project/cardio-oncology-env/bin/activate

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# ── project ───────────────────────────────────────────────────────────────────
PROJECT=/home/NAME/cardio_oncology_project/MIMIC-Cardio-Oncology
cd $PROJECT

echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "Working dir : $(pwd)"
echo "Python      : $(which python)"
echo "GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Started     : $(date)"
echo "─────────────────────────────────────────────"

# ── run ───────────────────────────────────────────────────────────────────────
python run_train_NAME.py******

echo "─────────────────────────────────────────────"
echo "Finished : $(date)"
