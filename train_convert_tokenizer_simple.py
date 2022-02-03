import sentencepiece as spm
from datasets import load_dataset
from transformers.convert_slow_tokenizer import SpmConverter
import argparse, os


parser = argparse.ArgumentParser()
parser.add_argument("--vocab_size", "-v", type=int, required=False, default=150000)
parser.add_argument("--data_name", "-d", type=str, required=True)
parser.add_argument("--output_folder", "-o", type=str, required=False, default='./')
parser.add_argument("--num_threads", "-th", type=int, required=False, default=90)


tokenizer_path = os.path.join(args.output_folder, "tokenizer")

def dataset_iterator(self, dataset):
    for i in range(len(dataset)):
        yield dataset[i]["text"] # assume relevant data is stored in 'text' field (datasets convention)


class SPMTokenizer:
    def __init__(self, vocab_file):
        self.vocab_file = vocab_file


dataset = load_dataset(args.data_name)

spm.SentencePieceTrainer.train(sentence_iterator=dataset_iterator(dataset),
                               model_prefix=tokenizer_path,
                               vocab_size=args.vocab_size,
                               model_type="bpe",
                               max_sentence_length=4096,
                               num_threads=args.num_threads,
                               byte_fallback=True,
                               train_extremely_large_corpus=True)

original_tokenizer = SPMTokenizer(tokenizer_path + ".model")
converter = SpmConverter(original_tokenizer)
hf_tokenizer = converter.converted()
hf_tokenizer.save(tokenizer_path + ".json")
