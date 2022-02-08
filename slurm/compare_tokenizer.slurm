#!/bin/bash
#SBATCH --job-name=train_tokenizer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40         # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/train_tokenizer/%x-%j.out           # output file name
#SBATCH --account=six@cpu

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate thomas_data_tooling # Debug deepspeed temporarily

TOKENIZATION_REPO=$WORK/code/big_science/tokenization

pushd $TOKENIZATION_REPO

python compare_tokenizers.py \
    --dataset-name xnli \
    --subset-name en \
    --text-columns hypothesis,premise \
    --num-proc 40