set -x -e

TOKENIZATION_REPO=~/code/tokenization

pushd $TOKENIZATION_REPO

TOKENIZER_NAME=tokenizer_alpha_weight_fixed_NFKC_no_split_numbers

DATASET_PATH=~/tokenization_dataset/alpha # TODO: define where is concatenated dataset
SAVE_TOKENIZER_PATH=~/tokenizer/tokenizer_alpha_weight_fixed_NFKC_no_split_numbers
LOGS_PATH=~/logs

mkdir -p $SAVE_TOKENIZER_PATH

export HF_DATASETS_OFFLINE=1

# Tokenization vocabulary size:
#   - ceil(150_000 / (8 * 128)) * 8 * 128
#   - special tokens:
#       - 200 sentinel tokens
# ceil(150_000 / (8 * 128)) * 8 * 128 - 200 = 150328

# --max_sequence_length 65536
# --input_sentence_size 12000000
python train_convert_tokenizer_simple.py \
    --vocab_size 150328 \
    --data_name ${DATASET_PATH} \
    --output_folder ${SAVE_TOKENIZER_PATH} \
    --load_batch_size 1000 \
    --input_sentence_size 24000000 \
    --max_sequence_length 8192 \
    --num_threads 48 \
    --normalizer nfkc \
    2>&1 | tee $LOGS_PATH/$TOKENIZER_NAME.txt
