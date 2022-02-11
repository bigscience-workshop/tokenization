import logging
import math
from pathlib import Path
from typing import List

import sentencepiece as spm
from datasets import load_dataset, utils
from datasets.utils.logging import set_verbosity_info
from transformers.convert_slow_tokenizer import SpmConverter
from transformers import PreTrainedTokenizerFast
import argparse

set_verbosity_info()
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", "-v", type=int, required=True)
    parser.add_argument("--data_name", "-d", type=str, required=True)
    parser.add_argument("--output_folder", "-o", type=Path, required=True)
    parser.add_argument("--num_threads", "-th", type=int, required=True)
    parser.add_argument("--load_batch_size", type=int, default=1)
    parser.add_argument("--max_sequence_length", type=int, required=True)
    parser.add_argument("--input_sentence_size", type=int, required=True)
    parser.add_argument("--nomalizer", type=int, required=True)

    return parser.parse_args()

def dataset_iterator(dataset, batch_size: int, sequence_length_in_byte: int):
    slices = [(start, min(len(dataset), start + batch_size)) for start in range(0, len(dataset), batch_size)]
    for start, end in utils.tqdm(
        slices,
        total=len(slices),
        unit="ba",
        disable=bool(utils.logging.get_verbosity() == utils.logging.NOTSET),
        desc="Loading dataset to sentencepiece",
    ):
        # Load things by batch.
        batch = dataset[start: end]
        batch_results = preprocess_text(batch, sequence_length_in_byte)
        for row_results in batch_results:
            for text in row_results:
                yield text

def preprocess_text(batch, sequence_length_in_byte: int) -> List[List[str]]:
    batch_results = []
    for text in batch["text"]:
        row_results = []

        # Removes None
        if not text:
            continue

        text = text.strip()

        if len(text) == 0:
            continue

        # Compute an average of the number of bytes needed to encode a character for that sequence
        # Needed since it will vary a lot depending on the language.
        avg_bytes_per_character = math.ceil(len(text.encode('utf8')) / len(text))

        sequence_length = sequence_length_in_byte // avg_bytes_per_character

        # shard text to be into substrings of size < sequence length
        start = 0
        end = min(sequence_length, len(text))
        while end - start != 0:
            if end - start < sequence_length:
                # Short sequence: we fit everything in size one line
                row_results.append(text[start: end])
                start = end
            else:
                candidates = text[start:end]
                matches = candidates.rsplit(" ", 1)
                if matches[0] == "":
                    # If whitespace is the first and only occurence in the sequence, We just feed everything
                    substring = candidates
                else:
                    substring = matches[0]

                start += len(substring)
                end = min(start + sequence_length, len(text))
                row_results.append(substring)

        batch_results.append(row_results)
    return batch_results

class SPMTokenizer:
    def __init__(self, vocab_file):
        self.vocab_file = vocab_file

# def reduce_max_text_length_on_shard(index:int, num_shards: int, dataset: Dataset, batch_size: int):
#     shard = dataset.shard(num_shards=num_shards, index=index)
#     return max([len(text) for text in dataset_iterator(shard, batch_size)])

def main():
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    args = get_args()
    logger.info(
        f"** The job is runned with the following arguments: **\n{args}\n **** "
    )
    tokenizer_path = args.output_folder / "tokenizer"

    dataset = load_dataset(args.data_name, data_files="**.jsonl.gz", split="train")

    logger.info(f"Dataset length: {len(dataset)}")
    # max_length = 0
    # for text in dataset_iterator(dataset, args.load_batch_size):
    #     length = len(text)
    #     if max_length < length:
    #         max_length = length

    # # Parallel version
    # with Pool(args.num_threads) as pool:
    #     max_per_shard = pool.map(
    #         partial(
    #             reduce_max_text_length_on_shard,
    #             num_shards=args.num_threads,
    #             dataset=dataset,
    #             batch_size=args.load_batch_size,
    #         ),
    #         range(args.num_threads)
    #     )
    #     max_length=max(max_per_shard)
    # logger.info(f"Max length: {max_length}")

    spm.SentencePieceTrainer.train(
        sentence_iterator=dataset_iterator(
            dataset,
            args.load_batch_size,
            sequence_length_in_byte=args.max_sequence_length
        ),
        input_sentence_size=args.input_sentence_size,
        shuffle_input_sentence=True,
        model_prefix=str(tokenizer_path.absolute()),
        vocab_size=args.vocab_size,
        model_type="bpe",
        max_sentence_length=args.max_sequence_length,
        num_threads=args.num_threads,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        byte_fallback=True,
        train_extremely_large_corpus=True,
        normalization_rule_name=args.normalizer
    )

    spm_model_path = tokenizer_path / f"tokenizer.model"
    original_tokenizer = SPMTokenizer(str(spm_model_path.absolute()))
    converter = SpmConverter(original_tokenizer)
    hf_tokenizer = converter.converted()
    tokenizer_json = tokenizer_path / f"tokenizer.json"
    hf_tokenizer.save(str(tokenizer_json.absolute()))

    # WIP:
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object = hf_tokenizer,
        unk_token="<unk>",
        eos_token="</s>",
        bos_token="<s>",
        pad_token="<pad>",
    )
    tokenizer.save_pretrained(
        tokenizer_path / f"tokenizer_hf"
    )

if __name__ == "__main__":
    main()
