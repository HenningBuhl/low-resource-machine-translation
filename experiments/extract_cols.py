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
    parser.add_argument('--file', default='data/raw', type=str, help='The file which contains parallel data.')
    #parser.add_argument('--delimiter', default='tab', type=str, choices=['tab'],  help='Whether to ignore the first row.')
    parser.add_argument('--skip-header', default=False, type=strtobool, help='Whether to ignore the first row.')
    parser.add_argument('--indices', default=[0, 1], type=int, nargs="*", help='The indices to extract to a separate file each.')
    
    # Parse args.
    args = parser.parse_args()

    # Print args.
    print(f'Arguments:')
    print(args)

    # Read lines of file.
    with open(args.file, 'r', encoding='utf8') as f:
        lines = f.readlines()

    # Extract parallel data from file.
    output_lines = {i:[] for i in args.indices}
    for i, line in enumerate(lines):
        if i == 0 and args.skip_header:
            continue
        cells = line.replace('\n', '').split('\t')
        for index in args.indices:
            output_lines[index].append(cells[index])

    # Save extracted columns.
    for index in args.indices:
        with open(f'{args.file}.{index}', 'w', encoding='utf8') as f:
          f.write('\n'.join(output_lines[index]))


if __name__ == '__main__':
    main()
