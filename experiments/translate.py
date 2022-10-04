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
from tokenizer import TokenizerBuilder
from transformer import *
from util import *


def main():
    # Define arguments with argparse.
    arg_manager = ArgManager()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Run.
    parser.add_argument('--model-path', default='models/model', type=str, help='The path of the model to use.')
    parser.add_argument('--text', default='example', type=str, help='The text to translate.')
    parser.add_argument('--seed', default=0, type=int, help='The random seed of the program.')

    # TODO add inference methods, their args and metrics.

    # Parse args.
    args = parser.parse_args()

    # Set seed.
    from pytorch_lightning import seed_everything
    seed_everything(args.seed, workers=True)

    # Load arguments.
    m_args = load_dict(os.path.join(args.model_path, 'args.json'))

    # TODO read args.model_type to enable cascaded translation (put code creating inference_fn in separate class. Then use the same code in benchmark).

    # Load tokenizers.
    src_tokenizer = TokenizerBuilder(m_args.src_lang).build()
    tgt_tokenizer = TokenizerBuilder(m_args.tgt_lang).build()

    # Load model.
    model = load_model_from_path(args.model_path, src_tokenizer, tgt_tokenizer)
    model = model.to(device)

    # Translate.
    translation = model.translate(args.text, method='greedy')
    print(f'Translated:\n\t{args.text}\nTo:\n\t{translation}')


if __name__ == '__main__':
    main()
