#!/bin/bash
#SBATCH --job-name=FlatBundle_mutag
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --array=0-119

source /home/parentd1/.bashrc
conda activate cooperative_sheaves

export WANDB_API_KEY=ab6e83a9d99012b14cff411df4e314597d9e6898
export WANDB__SERVICE_WAIT=300

SWEEP_IDS=(
  "andrerg00/FlatBundle_OVSQ_UPDT/8elj5qx1"
  "andrerg00/FlatBundle_OVSQ_UPDT/h5ia97l5"
  "andrerg00/FlatBundle_OVSQ_UPDT/yaz0npez"
  "andrerg00/FlatBundle_OVSQ_UPDT/sx3euiuu"
  "andrerg00/FlatBundle_OVSQ_UPDT/y4xv902j"
  "andrerg00/FlatBundle_OVSQ_UPDT/n2dmj9gs"
  "andrerg00/FlatBundle_OVSQ_UPDT/lydc7495"
  "andrerg00/FlatBundle_OVSQ_UPDT/r9f9d5lu"
)

NUM_SWEEPS=${#SWEEP_IDS[@]}

IDX=$(( SLURM_ARRAY_TASK_ID % NUM_SWEEPS ))
export SWEEP_ID=${SWEEP_IDS[$IDX]}

wandb agent "$SWEEP_ID"
