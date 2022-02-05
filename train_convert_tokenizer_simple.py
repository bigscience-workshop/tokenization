from pathlib import Path

import sentencepiece as spm
from datasets import load_dataset, utils
from transformers.convert_slow_tokenizer import SpmConverter
import argparse, os

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
    for i in range(len(dataset)):
        yield dataset[i]["text"]


class SPMTokenizer:
    def __init__(self, vocab_file):
        self.vocab_file = vocab_file

def main():
    args = get_args()

    tokenizer_path = args.output_folder / "tokenizer"

    dataset = load_dataset(args.data_name, data_files="**.jsonl.gz", split="train")

    spm.SentencePieceTrainer.train(
        sentence_iterator=dataset_iterator(dataset, args.batch_size),
        model_prefix=str(tokenizer_path.absolute()),
        vocab_size=args.vocab_size,
        model_type="bpe",
        max_sentence_length=4096,
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
