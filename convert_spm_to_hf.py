from transformers.convert_slow_tokenizer import SpmConverter
import sys

class SPMTokenizer:
    def __init__(self, vocab_file):
        self.vocab_file = vocab_file

input_file = sys.argv[1] + '.model'
output_file = sys.argv[1] + '.json'
        
original_tokenizer = SPMTokenizer(input_file)
converter = SpmConverter(original_tokenizer)
tokenizer = converter.converted()
tokenizer.save(output_file)
