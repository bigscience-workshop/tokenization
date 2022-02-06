import logging
from pathlib import Path

import sentencepiece as spm
from datasets import load_dataset, utils
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
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()

def dataset_iterator(dataset, batch_size: int):
    # WIP
    # slices = [(start, min(len(datasets), start + batch_size)) for start in range(0, len(dataset), batch_size)]
    # for i in utils.tqdm(
    #     ,
    #     total=len(dataset),
    #     unit="ba",
    #     disable=bool(utils.logging.get_verbosity() == utils.logging.NOTSET),
    #     desc="Loading dataset",
    # ):
    #     # Load things by batch.
    #     batch = dataset[i: i+batch_size]["text"]
    #     for text in batch:
    #         yield text
    #
    for sample in dataset:
        text = sample["text"]

        # Removes None
        if not text:
            continue

        text = text.strip()

        # Remove all whitespaces
        if not text:
            continue

        yield text


class SPMTokenizer:
    def __init__(self, vocab_file):
        self.vocab_file = vocab_file

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
    max_length = 0
    for text in dataset_iterator(dataset, args.batch_size):
        length = len(text)
        if max_length < length:
            max_length = length
    logger.info(f"Max length: {max_length}")

    spm.SentencePieceTrainer.train(
        sentence_iterator=dataset_iterator(dataset, args.batch_size),
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
