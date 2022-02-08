import logging
from functools import partial
from pathlib import Path

from datasets import load_dataset
from datasets.utils.logging import set_verbosity_info
import argparse, os

from .train_convert_tokenizer_simple import preprocess_text

set_verbosity_info()
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", "-d", type=str, required=True)
    parser.add_argument("--pathological_samples_path", "-o", type=Path, required=True)
    parser.add_argument("--load_batch_size", type=int, default=1)
    parser.add_argument("--max_sequence_length", type=int, required=True)
    parser.add_argument("--input_sentence_size", type=int, required=True)
    parser.add_argument("--num_proc", type=int, required=True)

    return parser.parse_args()

def get_not_utf_8_compatible(batch, sequence_length: int):
    batch_results = preprocess_text(batch, sequence_length=sequence_length)

    return {
        "utf-8-not-compatible": [text.encode() for row_results in batch_results for text in row_results if text]
    }


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

    dataset = load_dataset(args.data_name, data_files="**.jsonl.gz", split="train")

    logger.info(f"Dataset length: {len(dataset)}")

    # Try to find all that are not castable to utf-8
    # FIXME: we use an approximation of byte length vs byte sequence
    sequence_length = args.input_sentence_size // 2
    dataset = dataset.map(
        partial(get_not_utf_8_compatible, sequence_length=sequence_length),
        batched=True,
        num_proc=args.pathological_samples_path,
        remove_columns=dataset.column_names
    )

    logger.info(f"Invalid text: {dataset}")
    dataset.to_json(
        args.save_path,
        num_proc=args.num_proc
    )


if __name__ == "__main__":
    main()
