# Add src module directory to system path for subsecuent imports.
import sys
sys.path.insert(0, '../src')

# From packages.
import os
import pytorch_lightning as pl
import argparse
from distutils.util import strtobool

# From repository.
from arg_manager import ArgManager
from constants import *
from data import ParallelDataPreProcessor
from metric_logging import MetricLogger
from plotting import plot_metric
from tokenizer import TokenizerBuilder
from transformer import *
from util import *


def main():
    # Define arguments with argparse.
    arg_manager = ArgManager()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Experiment.
    parser.add_argument('--lang', default='de', type=str, help='The tokenizer language.')
    parser.add_argument('--fresh-run', default=False, type=strtobool, help='Ignores all cashed data on disk, reruns generation and overwrites everything.')
    parser.add_argument('--vocab-size', default=16000, type=int, help='The vocabulary size of the tokenizer.')
    parser.add_argument('--char-coverage', default=1.0, type=float, help='The character coverage (percentage) of the tokenizer.')
    parser.add_argument('--seed', default=0, type=int, help='The random seed of the program.')

    # Parse args.
    args = parser.parse_args()

    # Print args.
    print(f'Arguments:')
    print(args)
    
    # Set seed.
    set_seed(args.seed)

    # Create tokenizers dir.
    create_dir(CONST_TOKENIZERS_DIR)

    # Train tokenizer.
    TokenizerBuilder(args.lang).build(args.vocab_size, args.char_coverage, fresh_run=args.fresh_run)


if __name__ == '__main__':
    main()
