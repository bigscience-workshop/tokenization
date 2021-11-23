from datasets import load_dataset, concatenate_datasets
import sentencepiece as spm
import argparse, time, json, os
from tokenizer import SentencePieceTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--vocab_size", "-v", type=int, required=True)
parser.add_argument("--byte_fallback", "-b", action="store_true")
parser.add_argument("--segmenters", "-s", type=str, required=False, default='bpe')
parser.add_argument("--parts", "-p", type=float, required=False, default=1)
parser.add_argument("--language", "-l", type=str, required=False, default=None)
parser.add_argument("--data_base_folder", "-d", type=str, required=False, default='data/shuffled_new')
parser.add_argument("--data_files", type=str, required=False, default=None)
parser.add_argument("--output_folder", "-o", type=str, required=False, default='segmenters/retrained_shuffled')
parser.add_argument("--num_threads", "-th", type=int, required=False, default=90)

args = parser.parse_args()

def construct_file_path(base_folder, lg, part):
    return os.path.join(base_folder, lg, lg+'_shuf_part_'+str(part)+'.txt')

if args.language:
    lgs = args.language.split(",")
else:
    lgs = 'ja ko th vi ta hu tr de cs ru lt id el hi ar fr en zh fi'.split(' ')
    normalize_parts = args.parts if args.parts < 1 else int(args.parts) 
    lgs = [lg for lg in lgs if os.path.exists(construct_file_path(args.data_base_folder, lg, normalize_parts))] 
    # make sure parts exist

print(lgs)

def trunc(example):
    return {'text': example['text'][:65000]}

if args.data_files:
    paths = [os.path.join(args.data_base_folder, path) for path in args.data_files.split(',')]
    oscar_datasets = load_dataset('text', data_files=paths).shuffle(seed=108)['train']
elif args.parts >= 1:
    parts = [i for i in range(1, int(args.parts)+1)]
    oscar_datasets = {lg: load_dataset('text', 
                                       data_files=[construct_file_path(args.data_base_folder, lg, i) 
                                                   for i in parts]).shuffle(seed=108)['train'] 
                      for lg in lgs}
else:
    oscar_datasets = {lg: load_dataset('text', 
                                       data_files=construct_file_path(args.data_base_folder, lg, args.parts))
                          .shuffle(seed=108)['train'] 
                      for lg in lgs}

def dataset_iterator(dataset):
    for i in range(len(dataset)):
        yield dataset[i]["text"]

def make_folder(folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

def train_tokenizer(train_dataset, segmenter, lg, vocab_size, output_dir=''):
    tokenizer = SentencePieceTokenizer()
    make_folder(output_dir)
    start = time.time()
    tokenizer.train(train_dataset, output_dir, segmenter, lg, str(args.parts), vocab_size,
                    num_threads=args.num_threads,
                    byte_fallback=args.byte_fallback)
    return time.time() - start

make_folder('timings')

for segmenter in args.segmenters.split(','):
    timing_file = 'timings/timings_per_lg_'+segmenter+'_'+str(args.parts)+'_oscar_vocab_'+str(args.vocab_size)+'.json'
    timings = json.load(open(timing_file, 'r')) if os.path.exists(timing_file) else {}
    if args.data_files:
        lg = "-".join(lgs)
        time_taken = train_tokenizer(oscar_datasets, segmenter, lg, args.vocab_size, os.path.join(args.output_folder, lg))
        timings[lg] = time_taken
    else:
        for lg in lgs:
            print(segmenter, lg)
            time_taken = train_tokenizer(oscar_datasets[lg], segmenter, lg, args.vocab_size, os.path.join(args.output_folder, lg))
            timings[lg] = time_taken

    json.dump(timings, open('timings/timings_per_lg_'+segmenter+'_'+str(args.parts)+'_oscar_vocab_'+str(args.vocab_size)+'.json', 'w'), indent=4)
