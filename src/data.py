from tqdm import tqdm
from constants import *
from path_management import CONST_DATA_DIR, get_files, get_parallel_data_dir
from util import *
from torch.utils.data import DataLoader
from functools import partial

import torch
import random
import os
import pickle


class ParallelDataPreProcessor():
    '''A class handling preprocessing of a parallel data corpus.'''

    def __init__(self, src_lang, tgt_lang):
        # Parallel corpus data dir.
        data_dir = os.path.join(get_parallel_data_dir(CONST_DATA_DIR, src_lang, tgt_lang))

        # Language directories.
        self.src_lang_dir = os.path.join(data_dir, src_lang)
        self.tgt_lang_dir = os.path.join(data_dir, tgt_lang)

        # Split data files.
        self.src_train_file = os.path.join(self.src_lang_dir, 'train.txt')
        self.tgt_train_file = os.path.join(self.tgt_lang_dir, 'train.txt')
        self.src_val_file = os.path.join(self.src_lang_dir, 'val.txt')
        self.tgt_val_file = os.path.join(self.tgt_lang_dir, 'val.txt')
        self.src_test_file = os.path.join(self.src_lang_dir, 'test.txt')
        self.tgt_test_file = os.path.join(self.tgt_lang_dir, 'test.txt')

        # Tokenized data files.
        self.src_tokenized_train_file = os.path.join(self.src_lang_dir, 'train-tokenized.pickle')
        self.tgt_tokenized_train_file = os.path.join(self.tgt_lang_dir, 'train-tokenized.pickle')
        self.src_tokenized_val_file = os.path.join(self.src_lang_dir, 'val-tokenized.pickle')
        self.tgt_tokenized_val_file = os.path.join(self.tgt_lang_dir, 'val-tokenized.pickle')
        self.src_tokenized_test_file = os.path.join(self.src_lang_dir, 'test-tokenized.pickle')
        self.tgt_tokenized_test_file = os.path.join(self.tgt_lang_dir, 'test-tokenized.pickle')

        # Raw language files.
        self.src_files = self.get_raw_files(self.src_lang_dir)
        self.tgt_files = self.get_raw_files(self.tgt_lang_dir)

    def split_data(self, shuffle, num_val_examples, num_test_examples, fresh_run):
        if self.is_data_split() and not fresh_run:
            print('Data is already split.')
            return
        elif self.is_data_tokenized() and not fresh_run:
            print('Data is already split and tokenized.')
            return

        # Gather sentences.
        print('Gathering data from src files.')
        src_sentences = []
        for src_file in self.src_files:
            with open(src_file, 'r', encoding='utf8') as f:
                src_sentences.extend(f.readlines())

        print('Gathering data from tgt files.')
        tgt_sentences = []
        for tgt_file in self.tgt_files:
            with open(tgt_file, 'r', encoding='utf8') as f:
                tgt_sentences.extend(f.readlines())

        # Zip data into list of sentence pairs.
        pairs = list(zip(src_sentences, tgt_sentences))

        # Shuffle data.
        if shuffle:
            print('Shuffling data.')
            random.shuffle(pairs)
        
        # Split data.
        num_examples = len(src_sentences)
        num_train_examples = num_examples - num_val_examples - num_test_examples

        print(f'Splitting data into ({num_train_examples}, {num_val_examples}, {num_test_examples}) (train, val, test).')
        train_examples = pairs[0:num_train_examples]
        val_examples = pairs[num_train_examples:num_train_examples+num_val_examples]
        test_examples = pairs[num_train_examples+num_val_examples:]

        # Save split data to disk.
        print('Saving split data to disk.')
        src_train_examples, tgt_train_examples = zip(*train_examples)
        src_val_examples, tgt_val_examples = zip(*val_examples)
        src_test_examples, tgt_test_examples = zip(*test_examples)

        with open(self.src_train_file, 'w') as f: f.write(''.join(src_train_examples))
        with open(self.src_val_file, 'w') as f: f.write(''.join(src_val_examples))
        with open(self.src_test_file, 'w') as f: f.write(''.join(src_test_examples))
        with open(self.tgt_train_file, 'w') as f: f.write(''.join(tgt_train_examples))
        with open(self.tgt_val_file, 'w') as f: f.write(''.join(tgt_val_examples))
        with open(self.tgt_test_file, 'w') as f: f.write(''.join(tgt_test_examples))

    def pre_process(self, src_tokenizer, tgt_tokenizer, batch_size, shuffle, max_examples, max_len, fresh_run=False):
        if self.is_data_tokenized() and not fresh_run:
            # Load tokenized data.
            print('Loading tokenized data from disk.')
            with open(self.src_tokenized_train_file, 'rb') as f: src_train_tokenized = pickle.load(f)
            with open(self.src_tokenized_val_file, 'rb') as f: src_val_tokenized = pickle.load(f)
            with open(self.src_tokenized_test_file, 'rb') as f: src_test_tokenized = pickle.load(f)
            with open(self.tgt_tokenized_train_file, 'rb') as f: tgt_train_tokenized = pickle.load(f)
            with open(self.tgt_tokenized_val_file, 'rb') as f: tgt_val_tokenized = pickle.load(f)
            with open(self.tgt_tokenized_test_file, 'rb') as f: tgt_test_tokenized = pickle.load(f)
        else:
            # Load (train, val, test) sets.
            print('Loading split data from disk.')
            with open(self.src_train_file, 'r', encoding='utf8') as f: src_train_examples = f.readlines()
            with open(self.src_val_file, 'r', encoding='utf8') as f: src_val_examples = f.readlines()
            with open(self.src_test_file, 'r', encoding='utf8') as f: src_test_examples = f.readlines()
            with open(self.tgt_train_file, 'r', encoding='utf8') as f: tgt_train_examples = f.readlines()
            with open(self.tgt_val_file, 'r', encoding='utf8') as f: tgt_val_examples = f.readlines()
            with open(self.tgt_test_file, 'r', encoding='utf8') as f: tgt_test_examples = f.readlines()

            # Tokenize data.
            print('Tokenizing data.')
            src_train_tokenized = self.tokenize(src_train_examples, src_tokenizer)
            src_val_tokenized = self.tokenize(src_val_examples, src_tokenizer)
            src_test_tokenized = self.tokenize(src_test_examples, src_tokenizer)
            tgt_train_tokenized = self.tokenize(tgt_train_examples, tgt_tokenizer)
            tgt_val_tokenized = self.tokenize(tgt_val_examples, tgt_tokenizer)
            tgt_test_tokenized = self.tokenize(tgt_test_examples, tgt_tokenizer)

            # Save tokenized data to disk.
            print('Saving tokenized data to disk.')
            with open(self.src_tokenized_train_file, 'wb') as f: pickle.dump(src_train_tokenized, f)
            with open(self.src_tokenized_val_file, 'wb') as f: pickle.dump(src_val_tokenized, f)
            with open(self.src_tokenized_test_file, 'wb') as f: pickle.dump(src_test_tokenized, f)
            with open(self.tgt_tokenized_train_file, 'wb') as f: pickle.dump(tgt_train_tokenized, f)
            with open(self.tgt_tokenized_val_file, 'wb') as f: pickle.dump(tgt_val_tokenized, f)
            with open(self.tgt_tokenized_test_file, 'wb') as f: pickle.dump(tgt_test_tokenized, f)

        # Zip src-tgt pairs.
        train_tokenized = list(zip(src_train_tokenized, tgt_train_tokenized))
        val_tokenized = list(zip(src_val_tokenized, tgt_val_tokenized))
        test_tokenized = list(zip(src_test_tokenized, tgt_test_tokenized))

        # Limit to max examples.
        if max_examples != -1:
            print(f'Limiting training data to {max_examples}')
            train_tokenized = train_tokenized[:max_examples]

        # Create data loaders.
        fn = partial(self.collate_fn, max_len=max_len)
        train_datalaoder = DataLoader(train_tokenized, batch_size=batch_size, shuffle=shuffle, collate_fn=fn)
        val_dataloader = DataLoader(val_tokenized, batch_size=batch_size, shuffle=False, collate_fn=fn)
        test_dataloader = DataLoader(test_tokenized, batch_size=batch_size, shuffle=False, collate_fn=fn)

        return train_datalaoder, val_dataloader, test_dataloader

    def is_data_split(self):
        return os.path.exists(self.src_train_file)

    def is_data_tokenized(self):
        return os.path.exists(self.src_tokenized_train_file)

    def collate_fn(self, batch, max_len):
        src_inputs, tgt_inputs, tgt_output = [], [], []
        for src, tgt in batch:
            src_inputs.append(pad_or_truncate(src + [eos_id], max_len))
            tgt_inputs.append(pad_or_truncate([sos_id] + tgt, max_len))
            tgt_output.append(pad_or_truncate(tgt + [eos_id], max_len))
        return torch.LongTensor(src_inputs), torch.LongTensor(tgt_inputs), torch.LongTensor(tgt_output)

    def tokenize(self, text_list, tokenizer):
        tokenized_list = []
        for text in tqdm(text_list):
            tokenized = tokenizer.EncodeAsIds(text.strip())
            tokenized_list.append(tokenized)
        return tokenized_list

    def get_raw_files(self, dir):
        files = get_files(dir)
        for f in ['train.txt', 'val.txt', 'test.txt', 'train-tokenized.pickle', 'val-tokenized.pickle', 'test-tokenized.pickle']:
            if f in files:
              files.remove(f)
        return sorted([os.path.join(dir, f) for f in files])


class BenchmarkDataPreProcessor:
    # TODO
    pass


def pad_or_truncate(tokens, max_len):
    if len(tokens) < max_len:
        left = max_len - len(tokens)
        padding = [pad_id] * left
        tokens += padding
    else:
        tokens = tokens[:max_len]
    return tokens
