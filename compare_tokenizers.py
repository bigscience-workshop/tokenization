import argparse
from functools import partial
from typing import Set, List, Dict

from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--dataset-name", type=str)
    parser.add_argument("--subset-name", type=str)
    parser.add_argument("--text-columns", type=lambda x: set(x.split(",")))
    parser.add_argument("--num-proc", type=int, default=1)

    args = parser.parse_args()

    if args.dataset_name is None:
        assert args.interactive
    else:
        assert args.text_columns is not None and len(args.text_columns) > 0

    return args

def check_encoding(tokenizer, text):
    print(tokenizer.convert_ids_to_tokens(tokenizer.encode(text)))

# def check_spm_is_equal_hf(hf_tokenizer, spm_tokenizer, text):
#     hf_tokenized = hf_tokenizer.convert_ids_to_tokens(hf_tokenizer.encode(text))
#     spm_tokenized = spm_tokenizer.encode(text, out_type=str)
#     print(f"Difference between my tokenizer vs multilingual one: {len(hf_tokenized)} vs {len(spm_tokenized)}")
#     print(hf_tokenized)
#     print(spm_tokenized)

def compare_to_previous_multilingual_tokenizer(hf_tokenizer, mul_tokenizer, text):
    hf_tokenized = hf_tokenizer.convert_ids_to_tokens(hf_tokenizer.encode(text))
    mul_tokenized = mul_tokenizer.convert_ids_to_tokens(mul_tokenizer.encode(text))
    print(f"Difference between my tokenizer vs multilingual one: {len(hf_tokenized)} vs {len(mul_tokenized)}")
    print(hf_tokenized)
    print(mul_tokenized)

def interactively_test_tokenizer(hf_tokenizer, mul_tokenizer):
    while True:
        print(" ++++++ New input +++")
        text = input()
        # check encoding
        print(" ++++++ Check encoding +++++")
        check_encoding(hf_tokenizer, text)
        print(" ++++++ Compare with previous alpha tokenizer +++++")
        compare_to_previous_multilingual_tokenizer(hf_tokenizer, mul_tokenizer, text)

def batch_tokenize(batch, tokenizer_name: str, tokenizer: PreTrainedTokenizerFast, text_columns: Set[str]):
    for text_column in text_columns:
        batch[f"{tokenizer_name}_{text_column}"] = tokenizer.batch_decode(tokenizer(batch[text_column]).input_ids)

    return batch

def batch_tokenize_on_all_tokenizers(batch, tokenizers: Dict[str, PreTrainedTokenizerFast], text_columns):
    for tokenizer_name, tokenizer in tokenizers.items():
        batch = batch_tokenize(batch, tokenizer_name, tokenizer, text_columns)
    return batch

def run_on_dataset(tokenizers: Dict[str, PreTrainedTokenizerFast], dataset, text_columns, num_proc):
    dataset = dataset.map(
        partial(batch_tokenize_on_all_tokenizers, tokenizers=tokenizers, text_columns=text_columns),
        batched=True,
        num_proc=num_proc
    )
    return dataset

def compute_metrics(dataset):
    # compute number of tokens (the lower the better)
    number_of_tokens = {}
    for column_name in dataset.column_names:
        number_of_tokens[column_name] = sum([len(elt) for elt in dataset[column_name]])
    print(number_of_tokens)

def main():
    args = get_args()
    # save_tokenizer()

    # Use samson's tokenized
    mul_tokenizer = AutoTokenizer.from_pretrained("bigscience/oscar_13_languages_alpha_weight")

    # Use HF tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bigscience-catalogue-lm-data/tokenizer_v0")

    if args.interactive:
        interactively_test_tokenizer(tokenizer, mul_tokenizer)
    else:
        dataset = load_dataset(args.dataset_name, args.subset_name, split="train")
        dataset = dataset.remove_columns(set(dataset.column_names) - args.text_columns)

        tokenizers = {
            "bs_tokenizer_v0": tokenizer,
            "samson_tokenizer": tokenizer,
        }

        dataset = run_on_dataset(tokenizers, dataset, text_columns=args.text_columns, num_proc=args.num_proc)

        compute_metrics(dataset)

    pass

if __name__ == "__main__":
    main()
