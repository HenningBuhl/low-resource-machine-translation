from constants import *
from tqdm import tqdm
from data import get_src_tgt_key

import os
import sentencepiece as spm


def load_tokenizer(lang, other_lang, vocab_size=16000, character_coverage=1.0, model_type='unigram'):
    tokenizer_path = os.path.join('./tokenizers', lang)

    if os.path.exists(tokenizer_path):
        print('Tokenizer exists. Skipping training.')
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.Load(f'{os.path.join(tokenizer_path, lang)}.model')
        return tokenizer
    else:
        print('Training tokenizer...')
        os.mkdir(tokenizer_path)
        src_tgt_key = get_src_tgt_key(lang, other_lang)
        dataset_name = 'WikiMatrix'
        src_file = os.path.join('data', src_tgt_key, f'{dataset_name}.{src_tgt_key}.{lang}')

        template = "--input={} \
                --pad_id={} \
                --bos_id={} \
                --eos_id={} \
                --unk_id={} \
                --model_prefix={} \
                --vocab_size={} \
                --character_coverage={} \
                --model_type={}"

        config = template.format(src_file,
                                pad_id,
                                sos_id,
                                eos_id,
                                unk_id,
                                os.path.join(tokenizer_path, lang),
                                vocab_size,
                                character_coverage,
                                model_type)

        spm.SentencePieceTrainer.Train(config)

        tokenizer = spm.SentencePieceProcessor()
        tokenizer.Load(f'{os.path.join(tokenizer_path, lang)}.model')
        return tokenizer
