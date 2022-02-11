import argparse
import logging
from pathlib import Path

import youtokentome as yttm
from datasets.utils.logging import set_verbosity_info

set_verbosity_info()
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--txt-data", type=str, required=True)
    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument("--num-proc", type=int, required=True)

    return parser.parse_args()

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
    tokenizer_path = args.output_folder / "tokenizer_yttm"

    yttm.BPE.train(
        data=args.txt_data,
        model=str(tokenizer_path.absolute()),
        vocab=args.vocab_size,
        coverage=0.9995,
        n_threads=args.num_proc,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3
    )

if __name__ == "__main__":
    main()
