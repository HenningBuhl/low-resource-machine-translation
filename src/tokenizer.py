from constants import *
from tqdm import tqdm
from path_management import TokenizerPathManager

import os
import sentencepiece as spm


class TokenizerBuilder():
    '''A class training or loading a tokenizer.'''
    def __init__(self, lang, data_dir, mono_data_dir):
        self.lang = lang
        self.data_dir = data_dir
        self.mono_data_dir = mono_data_dir

        self.tpm = TokenizerPathManager(lang, data_dir, mono_data_dir)

    def build(self, vocab_size=16000, character_coverage=1.0, model_type='unigram'):
      if os.path.exists(self.tpm.tokenizer_path):
          print('Loading tokenizer from disk.')
          tokenizer = spm.SentencePieceProcessor()
          tokenizer.Load(f'{self.tpm.tokenizer_sp_path}.model')
          return tokenizer
      else:
          print('Training tokenizer.')
          os.mkdir(self.tpm.tokenizer_path)

          template = "--input={} \
                  --pad_id={} \
                  --bos_id={} \
                  --eos_id={} \
                  --unk_id={} \
                  --model_prefix={} \
                  --vocab_size={} \
                  --character_coverage={} \
                  --model_type={}"

          config = template.format(','.join(self.tpm.files),
                                  pad_id,
                                  sos_id,
                                  eos_id,
                                  unk_id,
                                  self.tpm.tokenizer_sp_path,
                                  vocab_size,
                                  character_coverage,
                                  model_type)

          spm.SentencePieceTrainer.Train(config)

          tokenizer = spm.SentencePieceProcessor()
          tokenizer.Load(f'{self.tpm.tokenizer_sp_path}.model')
          return tokenizer
