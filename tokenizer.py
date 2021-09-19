import sentencepiece as spm
from pathlib import Path
from typing import List
from tokenizers import Tokenizer, pre_tokenizers, decoders
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.trainers import BpeTrainer


class Tokenizer:
    def format_model_path(self, directory: str, model: str, name: str, vocab_size: int, language: str = 'multi') -> str:
        return str(Path(directory, '_'.join([model, language, name, str(vocab_size)])))

    def dataset_iterator(self, dataset):
        for i in range(len(dataset)):
            yield dataset[i]["text"]
    
    def encode(self, sentence: str):
        pass
    
    def decode(self, tokens: List[int]):
        pass


class SentencePieceTokenizer(Tokenizer):
    def __init__(self, directory: str = None, model: str = None, language: str = None, name: str = None, vocab_size: int = None):
        super().__init__()
        self.tokenizer = spm.SentencePieceProcessor()
        if directory and vocab_size and name and model and language:
            self.load(directory, model, language, name, vocab_size)
    
    def train(self, dataset, directory: str, model: str, language: str, name: str, 
              vocab_size: int, num_threads: int = 90, byte_fallback: bool = False, user_defined_symbols: List[str] = []) -> None:
        save_location = self.format_model_path(directory, model, name, vocab_size, language)
        print('Saving to: ' + save_location)
        spm.SentencePieceTrainer.train(sentence_iterator=self.dataset_iterator(dataset), 
                                        model_prefix=save_location, 
                                        vocab_size=vocab_size, 
                                        model_type=model,
                                        max_sentence_length=4096,
                                        num_threads=num_threads,
                                        byte_fallback=byte_fallback,
                                        user_defined_symbols=user_defined_symbols,
                                        train_extremely_large_corpus=True)        
    
    def load(self, directory: str, model: str, language: str, name: str, vocab_size: int) -> None:
        '''Load trained model. `name` can be used to track extra variables'''
        model_path = self.format_model_path(directory, model, name, vocab_size, language) + '.model'
        self.tokenizer.load(model_path)
    
    def encode(self, sentence: str) -> List[int]:
        return self.tokenizer.encode_as_ids(sentence)
    
    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)


class HFByteBPETokenizer(Tokenizer):
    def __init__(self, directory: str = None, language: str = None, name: str = None, vocab_size: int = None):
        super().__init__()
        if directory and vocab_size and name and language:
            self.load(directory, language, name, vocab_size)
    
    def format_model_path(self, directory: str, name: str, vocab_size: int, language: str = 'multi') -> str:
        return super().format_model_path(directory, 'bytebpe', name, vocab_size, language) + '.json'
        
    def load(self, directory: str, language: str, name: str, vocab_size: int) -> None:
        save_location = self.format_model_path(directory, name, vocab_size, language)
        self.tokenizer = Tokenizer.from_file(save_location)
        
    def train(self, dataset, directory: str,
              language: str, name: str, vocab_size: int):
        save_location = self.format_model_path(directory, name, vocab_size, language)
        print('Saving to: ' + save_location)
        tokenizer = Tokenizer(BPE())
        tokenizer.normalizer = NFKC()
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        tokenizer.decoders = decoders.ByteLevel()

        trainer = BpeTrainer(vocab_size=vocab_size,
                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                      special_tokens=["<PAD>", "<BOS>", "<EOS>"])
        tokenizer.train_from_iterator(self.dataset_iterator(dataset), trainer)
        tokenizer.save(save_location)
        
    def encode(self, sentence: str) -> List[int]:
        return self.tokenizer.encode(sentence).ids
    
    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)
    
    def dataset_iterator(self, dataset, batch_size=1000):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i : i + batch_size]["text"]
