from tqdm import tqdm
from constants import *
from path_management import DataPathManager
from util import *
from torch.utils.data import DataLoader

import torch
import random
import os


class PreProcessor():
    '''A class handling preprocessing of a parallel data corpus.'''

    def __init__(self, src_lang, tgt_lang, data_dir):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.data_dir = data_dir

        self.dpm = DataPathManager(self.src_lang, self.tgt_lang, self.data_dir)

        # Determine state of data in data dir.
        self.data_already_split = os.path.exists(self.dpm.src_train_file)

    def split_data(self, shuffle, num_val_examples, num_test_examples, fresh_run):
        if self.data_already_split and not fresh_run:
            print('Data is already split.')
            return

        # Gather sentences.
        print('Gathering data from src files.')
        src_sentences = []
        for src_file in self.dpm.src_files:
            with open(src_file, 'r', encoding='utf8') as f:
                src_sentences.extend(f.readlines())

        print('Gathering data from tgt files.')
        tgt_sentences = []
        for tgt_file in self.dpm.tgt_files:
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

        with open(self.dpm.src_train_file, 'w') as f: f.write(''.join(src_train_examples))
        with open(self.dpm.src_val_file, 'w') as f: f.write(''.join(src_val_examples))
        with open(self.dpm.src_test_file, 'w') as f: f.write(''.join(src_test_examples))
        with open(self.dpm.tgt_train_file, 'w') as f: f.write(''.join(tgt_train_examples))
        with open(self.dpm.tgt_val_file, 'w') as f: f.write(''.join(tgt_val_examples))
        with open(self.dpm.tgt_test_file, 'w') as f: f.write(''.join(tgt_test_examples))

    def pre_process(self, src_tokenizer, tgt_tokenizer, batch_size, shuffle, max_examples):
        # Load (train, val, test) sets.
        print('Loading split dat from disk.')
        with open(self.dpm.src_train_file, 'r', encoding='utf8') as f: src_train_examples = f.readlines()
        with open(self.dpm.src_val_file, 'r', encoding='utf8') as f: src_val_examples = f.readlines()
        with open(self.dpm.src_test_file, 'r', encoding='utf8') as f: src_test_examples = f.readlines()
        with open(self.dpm.tgt_train_file, 'r', encoding='utf8') as f: tgt_train_examples = f.readlines()
        with open(self.dpm.tgt_val_file, 'r', encoding='utf8') as f: tgt_val_examples = f.readlines()
        with open(self.dpm.tgt_test_file, 'r', encoding='utf8') as f: tgt_test_examples = f.readlines()

        # Tokenize data.
        print('Tokenizing data.')
        src_train_tokenized = self.tokenize(src_train_examples, src_tokenizer)
        src_val_tokenized = self.tokenize(src_val_examples, src_tokenizer)
        src_test_tokenized = self.tokenize(src_test_examples, src_tokenizer)
        tgt_train_tokenized = self.tokenize(tgt_train_examples, tgt_tokenizer)
        tgt_val_tokenized = self.tokenize(tgt_val_examples, tgt_tokenizer)
        tgt_test_tokenized = self.tokenize(tgt_test_examples, tgt_tokenizer)

        # Zip src-tgt pairs.
        train_tokenized = list(zip(src_train_tokenized, tgt_train_tokenized))
        val_tokenized = list(zip(src_val_tokenized, tgt_val_tokenized))
        test_tokenized = list(zip(src_test_tokenized, tgt_test_tokenized))

        # Limit to max examples.
        if max_examples != -1:
            print(f'Limiting training data to {max_examples}')
            train_tokenized = train_tokenized[:max_examples]

        # Create data loaders.
        train_datalaoder = DataLoader(train_tokenized, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)
        val_dataloader = DataLoader(val_tokenized, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)
        test_dataloader = DataLoader(test_tokenized, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)

        return train_datalaoder, val_dataloader, test_dataloader

    def collate_fn(self, batch):
        src_inputs, tgt_inputs, tgt_output = [], [], []
        for src, tgt in batch:
            src_inputs.append(pad_or_truncate(src + [eos_id]))
            tgt_inputs.append(pad_or_truncate([sos_id] + tgt))
            tgt_output.append(pad_or_truncate(tgt + [eos_id]))
        return torch.LongTensor(src_inputs), torch.LongTensor(tgt_inputs), torch.LongTensor(tgt_output)

    def tokenize(self, text_list, tokenizer):
        tokenized_list = []
        for text in tqdm(text_list):
            tokenized = tokenizer.EncodeAsIds(text.strip())
            tokenized_list.append(tokenized)
        return tokenized_list


def pad_or_truncate( tokens):
    if len(tokens) < max_len:
        left = max_len - len(tokens)
        padding = [pad_id] * left
        tokens += padding
    else:
        tokens = tokens[:max_len]
    return tokens
