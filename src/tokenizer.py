from constants import *
from tqdm import tqdm

import os
from path_management import CONST_TOKENIZERS_DIR, CONST_DATA_DIR, get_files, get_parallel_data_dir
import sentencepiece as spm


class TokenizerBuilder():
    '''A class training or loading a tokenizer.'''
    def __init__(self, lang, other_lang):  # TODO if other_lang is None, not a baseline training and tokenizer MUST already exist.
        # Parallel corpus data dir.
        data_dir = get_parallel_data_dir(lang, other_lang)

        # Tokenizer paths.
        self.tokenizer_path = os.path.join(CONST_TOKENIZERS_DIR, lang)
        self.tokenizer_sp_path = os.path.join(self.tokenizer_path, lang)

        # Files from parallel corpus.
        train_file = os.path.join(data_dir, lang, 'train.txt')
        val_file = os.path.join(data_dir, lang, 'val.txt')

        # Monolingual files.
        mono_data_files = []
        mono_data_dir = os.path.join(CONST_DATA_DIR, lang)
        if os.path.exists(mono_data_dir):
            mono_data_files = get_files(mono_data_dir)
        self.files = [train_file] + [val_file] + mono_data_files

    def build(self, vocab_size=16000, character_coverage=1.0, model_type='unigram', fresh_run=False):
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
