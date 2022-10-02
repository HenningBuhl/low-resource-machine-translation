# Add src module directory to system path for subsecuent imports.
import sys
sys.path.insert(0, '../src')

# From packages.
import pytorch_lightning as pl
import argparse
from distutils.util import strtobool

# From repository.
from arg_manager import *
from constants import *
from data import *
from layers import *
from metric_logging import *
from plotting import *
from tokenizer import *
from transformer import *
from util import *

def main():
    # Define arguments with argparse.
    arg_manager = ArgManager()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Experiment.
    parser.add_argument('--model-path', default='models/model', type=str, help='The path of the model to use.')
    parser.add_argument('--text', default='example', type=str, help='The text to translate.')
    parser.add_argument('--seed', default=0, type=int, help='The random seed of the program.')

    # Parse args.
    if is_notebook():
        sys.argv = ['-f']  # Used to make argparse work in jupyter notebooks (all args must be optional).
        args, _ = parser.parse_known_args()  # -f can lead to unknown argument.
    else:
        args = parser.parse_args()

    # Set seed.
    from pytorch_lightning import seed_everything
    seed_everything(args.seed, workers=True)

    # Load arguments.
    m_args = load_dict(os.path.join(args.model_path, 'args.json'))

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
