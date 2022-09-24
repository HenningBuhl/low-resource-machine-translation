from data import *

import os
import sys
import gzip


class ModelConfig():
    def __init__(self,
                 type, # single, cascaded
                 langs, # [src, tgt], [src, pvt, tgt]
                 paths, # model_path or paths
                 name=None,
                ):
        self.type = type
        self.langs = langs
        self.paths = paths
        if name is None:
            self.name = paths.split('/')[-1].replace('.pt', '')
        else:
            self.name = name


class BenchmarkConfig():
    def __init__(self,
                 name,
                 collate_fn, # benchmark data collate function (download and unzip).
                 pp_fn, # benchmark data preprocess function (get data for language pair).
                 lang_keys=None, # language key to benchmark language value (currently unused).
                 ):
        self.name = name
        self.collate_fn = collate_fn
        self.pp_fn = pp_fn
        self.lang_keys = lang_keys


def src_tgt_lists_to_dataset(src_text_list, tgt_text_list,
                             src_tokenizer, tgt_tokenizer):
    src_inputs = process_src(src_text_list, src_tokenizer)
    target_inputs, target_output = process_tgt(tgt_text_list, tgt_tokenizer)
    dataset = TensorDataset(torch.LongTensor(src_inputs), torch.LongTensor(target_inputs), torch.LongTensor(target_output))
    return dataset


lang_to_flores_key = {
    'de': 'deu_Latn',
    'en': 'eng_Latn',
    'nl': 'nld_Latn',
}

def flores_collate_fn(data_dir):
    benchmark_name = 'flores'
    zip_file = os.path.join(data_dir, f'{benchmark_name}.tar.gz')
    unzip_dir = os.path.join(data_dir, benchmark_name)
    
    benchmark_url = 'https://tinyurl.com/flores200dataset'
    download_file(benchmark_url, zip_file)
    unzip_targz(zip_file, unzip_dir)

def flores_pp_fn(data_path, src_lang, tgt_lang, src_tokenizer, tgt_tokenizer):
    data_path += '/flores/flores200_dataset'
    file_pairs = [
        #(os.path.join(data_path, 'dev', f'{lang_to_flores_key[src_lang]}.dev'), os.path.join(data_path, 'dev', f'{lang_to_flores_key[tgt_lang]}.dev')),
        (os.path.join(data_path, 'devtest', f'{lang_to_flores_key[src_lang]}.devtest'), os.path.join(data_path, 'devtest', f'{lang_to_flores_key[tgt_lang]}.devtest')),
    ]

    src_text_list = []
    tgt_text_list = []

    for src_file, tgt_file in file_pairs:
        with open(src_file, 'r', encoding='utf8') as s_f:
            src_text_list.extend(s_f.readlines())

        with open(tgt_file, 'r', encoding='utf8') as t_f:
            tgt_text_list.extend(t_f.readlines())

    return src_tgt_lists_to_dataset(src_text_list, tgt_text_list, src_tokenizer, tgt_tokenizer)


lang_to_tatoeba_key = {
    'de': 'deu',
    'en': 'eng',
    'nl': 'nld',
}

def get_tatoeba_key_pair(src_lang, tgt_lang, path_template):
    src_key = lang_to_tatoeba_key[src_lang]
    tgt_key = lang_to_tatoeba_key[tgt_lang]

    key_pair = f'{src_key}-{tgt_key}'
    path = path_template.format(key_pair)
    if os.path.exists(path):
        return key_pair

    key_pair = f'{tgt_key}-{src_key}'
    path = path_template.format(key_pair)
    if os.path.exists(path):
        return key_pair
    
    raise ValueError('No key combination found.')

def tatoeba_collate_fn(data_dir):
    benchmark_name = 'tatoeba'
    zip_file = os.path.join(data_dir, f'{benchmark_name}.tar')
    unzip_dir = os.path.join(data_dir, benchmark_name)
    
    benchmark_url = 'https://object.pouta.csc.fi/Tatoeba-Challenge-devtest/test.tar'
    download_file(benchmark_url, zip_file)
    unzip_tar(zip_file, unzip_dir)

def tatoeba_pp_fn(data_path, src_lang, tgt_lang, src_tokenizer, tgt_tokenizer):
    data_path += '/tatoeba/data/release/test/v2021-08-07'
    path_template = data_path + '/tatoeba-test-v2021-08-07.{}.txt.gz'
    lang_pair_key = get_tatoeba_key_pair(src_lang, tgt_lang, path_template)

    file_pairs = [
        path_template.format(lang_pair_key),
    ]

    src_text_list = []
    tgt_text_list = []

    for gz_file in file_pairs:
        lines = []
        with gzip.open(gz_file, 'r') as s_f:
            lines.extend(s_f.readlines())

        for line in lines:
            _, _, src_text, tgt_text = line.decode().split('\t')
            src_text_list.append(src_text)
            tgt_text_list.append(tgt_text)

    return src_tgt_lists_to_dataset(src_text_list, tgt_text_list, src_tokenizer, tgt_tokenizer)
