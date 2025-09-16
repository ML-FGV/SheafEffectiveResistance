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
  "andrerg00/FlatBundle_OVSQ/0speuloi"
  "andrerg00/FlatBundle_OVSQ/mu372bim"
  "andrerg00/FlatBundle_OVSQ/ooe6ma9g"
  "andrerg00/FlatBundle_OVSQ/fu1qlhst"
  "andrerg00/FlatBundle_OVSQ/qkyfr0cd"
  "andrerg00/FlatBundle_OVSQ/kuiazt9e"
  "andrerg00/FlatBundle_OVSQ/b1c0foxj"
  "andrerg00/FlatBundle_OVSQ/20ha8zeg"
  "andrerg00/FlatBundle_OVSQ/3ecc79ra"
  "andrerg00/FlatBundle_OVSQ/nkfdihsx"
)

NUM_SWEEPS=${#SWEEP_IDS[@]}

IDX=$(( SLURM_ARRAY_TASK_ID % NUM_SWEEPS ))
export SWEEP_ID=${SWEEP_IDS[$IDX]}

wandb agent "$SWEEP_ID"
