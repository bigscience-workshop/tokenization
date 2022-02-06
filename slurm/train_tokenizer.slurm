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

echo "Sharding and compressing seed id ${SEED_ID}"

DATASET_PATH=$six_ALL_CCFRSCRATCH/tokenizer/dataset/tokenization_dataset # TODO: define where is concatenated dataset
SAVE_TOKENIZER_PATH=$six_ALL_CCFRSCRATCH/tokenizer/tokenizer

mkdir -p $SAVE_TOKENIZER_PATH

export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=$SCRATCH/to_delete

# Tokenization vocabulary size:
#   - ceil(150_000 / (8 * 128)) * 8 * 128
#   - special tokens:
#       - 200 sentinel tokens
# ceil(150_000 / (8 * 128)) * 8 * 128 - 200 = 150325

python train_convert_tokenizer_simple.py \
    --vocab_size 150325 \
    --data_name ${DATASET_PATH} \
    --output_folder ${SAVE_TOKENIZER_PATH} \
    --load_batch_size 1000 \
    --num_threads 80

