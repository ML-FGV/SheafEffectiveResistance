#!/bin/bash
#SBATCH --job-name=FlatBundle_mutag
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --array=0-59

source /home/parentd1/.bashrc
conda activate cooperative_sheaves

export WANDB_API_KEY=ab6e83a9d99012b14cff411df4e314597d9e6898
export WANDB__SERVICE_WAIT=300

SWEEP_IDS=(
  "andrerg00/FlatBundle_OVSQ_UPDT/vp5cdelo"
  "andrerg00/FlatBundle_OVSQ_UPDT/m7b9midq"
  "andrerg00/FlatBundle_OVSQ_UPDT/a9z7coz8"
  "andrerg00/FlatBundle_OVSQ_UPDT/xy5afmop"
  "andrerg00/FlatBundle_OVSQ_UPDT/n8jqyzyk"
  "andrerg00/FlatBundle_OVSQ_UPDT/qs4q3t1t"
  "andrerg00/FlatBundle_OVSQ_UPDT/zej0cyzs"
  "andrerg00/FlatBundle_OVSQ_UPDT/b3xqrbtj"
)

NUM_SWEEPS=${#SWEEP_IDS[@]}

IDX=$(( SLURM_ARRAY_TASK_ID % NUM_SWEEPS ))
export SWEEP_ID=${SWEEP_IDS[$IDX]}

wandb agent "$SWEEP_ID"
