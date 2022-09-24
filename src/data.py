from tqdm import tqdm
from constants import *
from util import *
from torch.utils.data import TensorDataset

import torch
import sentencepiece as spm
import numpy as np
import os
import requests
import gzip
import shutil
import zipfile
import tarfile


def download_data(src_lang, tgt_lang):
    src_tgt_key = get_src_tgt_key(src_lang, tgt_lang)

    # Download data.
    data_zip_file = os.path.join('data', f'{src_tgt_key}.zip')
    download_file(wiki_matrix_data_urls[src_tgt_key], data_zip_file)

    # Unpack data.
    data_unzip_dir = os.path.join('data', src_tgt_key)
    unzip_zip(data_zip_file, data_unzip_dir)

def load_data(src_lang, tgt_lang, src_tokenizer, tgt_tokenizer, max_examples=-1):
    src_tgt_key = get_src_tgt_key(src_lang, tgt_lang)
    preprocessed_data_file = os.path.join('data', src_tgt_key, 'preprocessed.pt')
    
    if os.path.exists(preprocessed_data_file):
        print('Preprocessed data exists, loading from disk...')
        dataset = torch.load(preprocessed_data_file)
    else:
        dataset_name = 'WikiMatrix'
        src_file = os.path.join('data', src_tgt_key, f'{dataset_name}.{src_tgt_key}.{src_lang}')
        tgt_file = os.path.join('data', src_tgt_key, f'{dataset_name}.{src_tgt_key}.{tgt_lang}')

        print("Tokenizing & Padding source data...")
        with open(src_file, 'r', encoding='utf8') as f:
            src_text_list = f.readlines()
        src_inputs = process_src(src_text_list, src_tokenizer) # (sample_num, L)

        print("Tokenizing & Padding target data...")
        with open(tgt_file, 'r', encoding='utf8') as f:
            tgt_text_list = f.readlines()
        target_inputs, target_output = process_tgt(tgt_text_list, tgt_tokenizer) # (sample_num, L)

        print('Saving data to disk...')
        dataset = TensorDataset(torch.LongTensor(src_inputs), torch.LongTensor(target_inputs), torch.LongTensor(target_output))
        torch.save(dataset, preprocessed_data_file)    

    train_dataset, val_dataset, test_dataset = split_dataset(dataset, src_tgt_key, max_examples=max_examples)
    return train_dataset, val_dataset, test_dataset

def download_file(url, file_name):
    if os.path.exists(file_name):
        print(f'File "{file_name}" already exists. Skipping download.')
        return

    print(f'Downloading from {url}...')
    response = requests.get(url)
    open(file_name, "wb").write(response.content)
    print(f'Saved file to {file_name}.')

def unzip_gz(gz_file, unzip_file):
    if os.path.exists(unzip_file):
        print(f'File {unzip_file} already exists. Skipping unzipping.')
        return

    print(f'Unzipping {gz_file} to {unzip_file}...')
    with gzip.open(gz_file, 'rb') as f_in:
        with open(unzip_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    print(f'Unzipped {gz_file} to {unzip_file}.')

def unzip_zip(zip_file, unzip_dir):
    if os.path.exists(unzip_dir):
        print(f'Directory {unzip_dir} already exists. Skipping unzipping.')
        return
    print(f'Unzipping {zip_file} to {unzip_dir}...')
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(unzip_dir)
    print(f'Unzipped {zip_file} to {unzip_dir}.')

def unzip_tar(zip_file, unzip_dir):
    if os.path.exists(unzip_dir):
        print(f'Directory {unzip_dir} already exists. Skipping unzipping.')
        return
    print(f'Unzipping {zip_file} to {unzip_dir}...')
    with tarfile.open(zip_file, 'r') as zip_ref:
        zip_ref.extractall(unzip_dir)
    print(f'Unzipped {zip_file} to {unzip_dir}.')

def unzip_targz(zip_file, unzip_dir):
    if os.path.exists(unzip_dir):
        print(f'Directory {unzip_dir} already exists. Skipping unzipping.')
        return
    print(f'Unzipping {zip_file} to {unzip_dir}...')
    tar = tarfile.open(zip_file, "r:gz")
    tar.extractall(unzip_dir)
    tar.close()
    print(f'Unzipped {zip_file} to {unzip_dir}.')

def get_src_tgt_key(src_lang, tgt_lang):
    pseudo_key = f'{src_lang}-{tgt_lang}'
    if not pseudo_key in wiki_matrix_data_urls.keys():
        pseudo_key = f'{tgt_lang}-{src_lang}'
    return pseudo_key

wiki_matrix_data_urls = {
    'de-en': 'https://opus.nlpl.eu/download.php?f=WikiMatrix/v1/moses/de-en.txt.zip',
    'de-nl': 'https://opus.nlpl.eu/download.php?f=WikiMatrix/v1/moses/de-nl.txt.zip',
    'en-nl': 'https://opus.nlpl.eu/download.php?f=WikiMatrix/v1/moses/en-nl.txt.zip',
}

def process_src(text_list, src_tokenizer):
    tokenized_list = []
    for text in tqdm(text_list):
        tokenized = src_tokenizer.EncodeAsIds(text.strip())
        tokenized_list.append(pad_or_truncate(tokenized + [eos_id]))
    return tokenized_list

def process_tgt(text_list, tgt_tokenizer):
    input_tokenized_list = []
    output_tokenized_list = []
    for text in tqdm(text_list):
        tokenized = tgt_tokenizer.EncodeAsIds(text.strip())
        tgt_input = [sos_id] + tokenized
        tgt_output = tokenized + [eos_id]
        input_tokenized_list.append(pad_or_truncate(tgt_input))
        output_tokenized_list.append(pad_or_truncate(tgt_output))
    return input_tokenized_list, output_tokenized_list

def pad_or_truncate(tokens):
    if len(tokens) < max_len:
        left = max_len - len(tokens)
        padding = [pad_id] * left
        tokens += padding
    else:
        tokens = tokens[:max_len]
    return tokens

def split_dataset(dataset, src_tgt_key, val_ratio=0.05, test_ratio=0.05, max_examples=-1):
    print(f'Splitting {src_tgt_key} data...')
    max_examples = len(dataset) if max_examples == -1 else max_examples      
    train_size = np.minimum(int((1 - val_ratio - test_ratio) * len(dataset)), max_examples)
    val_size = int(val_ratio * len(dataset))
    test_size = int(test_ratio * len(dataset))
    residue_size = len(dataset) - train_size - val_size - test_size
    train_dataset, val_dataset, test_dataset, _ = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size, residue_size])

    print(f'Data ({src_tgt_key}) split.')
    return train_dataset, val_dataset, test_dataset
