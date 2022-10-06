from constants import *
from tqdm import tqdm
from util import *

import os
import sentencepiece as spm


class TokenizerBuilder():
    '''A class training or loading a tokenizer.'''

    def __init__(self, lang, other_lang=None):
        # Tokenizer paths.
        self.tokenizer_path = os.path.join(CONST_TOKENIZERS_DIR, lang)
        self.tokenizer_sp_path = os.path.join(self.tokenizer_path, lang)

        # Monolingual files.
        self.files = []
        mono_data_dir = os.path.join(CONST_DATA_DIR, lang)
        if os.path.exists(mono_data_dir):
            self.files.extend([os.path.join(mono_data_dir, mono_file) for mono_file in get_files(mono_data_dir)])

        if other_lang is not None:
            # Parallel corpus data dir.
            data_dir = get_parallel_data_dir(CONST_DATA_DIR, lang, other_lang)

            # Files from parallel corpus.
            train_file = os.path.join(data_dir, lang, 'train.txt')
            self.files.append(train_file)

            val_file = os.path.join(data_dir, lang, 'val.txt')
            self.files.append(val_file)

    def build(self, vocab_size=16000, character_coverage=1.0, model_type='unigram', fresh_run=False):
        '''Trains a tokenizer with the given files as sources. If the tokenizer already exists, it is loaded from disk'''

        # Check if the tokenizer can/should be loaded from disk.
        if os.path.exists(self.tokenizer_path) and not fresh_run:
            print('Loading tokenizer from disk.')
            tokenizer = spm.SentencePieceProcessor()
            tokenizer.Load(f'{self.tokenizer_sp_path}.model')
            return tokenizer
        else:
            print('Training tokenizer.')
            if not os.path.exists(self.tokenizer_path):
                os.mkdir(self.tokenizer_path)

            template = "--input={} \
                    --pad_id={} \
                    --bos_id={} \
                    --eos_id={} \
                    --unk_id={} \
                    --model_prefix={} \
                    --vocab_size={} \
                    --character_coverage={} \
                    --model_type={}"
                    #--input_sentence_size={} \  # TODO Add and test.
                    #--shuffle_input_sentence={} \  # TODO add and test.
                    # TODO is seed argument available?

            config = template.format(','.join(self.files),
                                    pad_id,
                                    sos_id,
                                    eos_id,
                                    unk_id,
                                    self.tokenizer_sp_path,
                                    vocab_size,
                                    character_coverage,
                                    model_type)

            spm.SentencePieceTrainer.Train(config)
            tokenizer = spm.SentencePieceProcessor()
            tokenizer.Load(f'{self.tokenizer_sp_path}.model')
            return tokenizer
