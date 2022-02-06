import logging
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Callable

import sentencepiece as spm
from datasets import load_dataset, utils, Dataset
from datasets.utils.logging import set_verbosity_info
from transformers.convert_slow_tokenizer import SpmConverter
import argparse, os

set_verbosity_info()
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", "-v", type=int, required=True)
    parser.add_argument("--data_name", "-d", type=str, required=True)
    parser.add_argument("--output_folder", "-o", type=Path, required=True)
    parser.add_argument("--num_threads", "-th", type=int, required=True)
    parser.add_argument("--load_batch_size", type=int, default=1)
    return parser.parse_args()

def dataset_iterator(dataset, batch_size: int):
    # # WIP
    slices = [(start, min(len(dataset), start + batch_size)) for start in range(0, len(dataset), batch_size)]
    for start, end in utils.tqdm(
        slices,
        total=len(dataset),
        unit="ba",
        disable=bool(utils.logging.get_verbosity() == utils.logging.NOTSET),
        desc="Loading dataset",
    ):
        # Load things by batch.
        batch = dataset[start: end]
        for text in batch["text"]:
            # Removes None
            if not text:
                continue

            text = text.strip()

            # Remove all whitespaces
            if not text:
                continue

            yield text

    # for sample in dataset:
    #     text = sample["text"]
    #
    #     # Removes None
    #     if not text:
    #         continue
    #
    #     text = text.strip()
    #
    #     # Remove all whitespaces
    #     if not text:
    #         continue
    #
    #     yield text


class SPMTokenizer:
    def __init__(self, vocab_file):
        self.vocab_file = vocab_file

def reduce_max_text_length_on_shard(index:int, num_shards: int, dataset: Dataset, batch_size: int):
    shard = dataset.shard(num_shards=num_shards, index=index, contiguous=True)
    return max([len(text) for text in dataset_iterator(shard, batch_size)])

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

    # Parallel version
    with Pool(args.num_threads) as pool:
        max_per_shard = pool.map(
            partial(
                reduce_max_text_length_on_shard,
                num_shards=args.num_threads,
                dataset=dataset,
                batch_size=args.load_batch_size,
            ),
            range(args.num_threads)
        )
        max_length=max(max_per_shard)
    logger.info(f"Max length: {max_length}")

    spm.SentencePieceTrainer.train(
        sentence_iterator=dataset_iterator(dataset, args.load_batch_size),
        model_prefix=str(tokenizer_path.absolute()),
        vocab_size=args.vocab_size,
        model_type="bpe",
        max_sentence_length=max_length,
        num_threads=args.num_threads,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        byte_fallback=True,
        train_extremely_large_corpus=True
    )

    spm_model_path = tokenizer_path.rename(f"{tokenizer_path.name}.model")
    original_tokenizer = SPMTokenizer(spm_model_path)
    converter = SpmConverter(original_tokenizer)
    hf_tokenizer = converter.converted()
    tokenizer_json = tokenizer_path.rename(f"{tokenizer_path.name}.json")
    hf_tokenizer.save(str(tokenizer_json.absolute()))

if __name__ == "__main__":
    main()
