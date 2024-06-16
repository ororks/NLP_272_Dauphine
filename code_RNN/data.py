# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 15:58:45 2024

@author: aikan
"""

import time
import os
import torch
from collections import defaultdict
import logging
from utils import batchify


class Data():

    def __init__(self, path2data, batch_size, eval_batch_size, cuda):
        self.path2data = path2data
        corpus = Corpus(path2data)
        self.train_data = batchify(corpus.train, batch_size, cuda)
        self.val_data = batchify(corpus.valid, eval_batch_size, cuda)
        self.test_data = batchify(corpus.test, eval_batch_size, cuda)
        self.ntokens = len(corpus.dictionary)


class Dictionary(object):
    def __init__(self, path):
        self.word2idx = {}
        self.idx2word = []
        self.word2freq = defaultdict(int)

        vocab_path = os.path.join(path, 'vocab.txt')
        try:
            vocab = open(vocab_path, encoding="utf8").read()
            self.word2idx = {w: i for i, w in enumerate(vocab.split())}
            self.idx2word = [w for w in vocab.split()]
            self.vocab_file_exists = True
        except FileNotFoundError:
            logging.info("Vocab file not found, creating new vocab file.")
            self.create_vocab(os.path.join(path, 'train.txt'))
            open(vocab_path,"w").write("\n".join([w for w in self.idx2word]))

    def add_word(self, word):
        self.word2freq[word] += 1
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        #return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def create_vocab(self, path):
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split()
                for word in words:
                    self.add_word(word)



class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary(path)
        self.train = tokenize(self.dictionary, os.path.join(path, 'train.txt'))
        self.valid = tokenize(self.dictionary, os.path.join(path, 'valid.txt'))
        self.test = tokenize(self.dictionary, os.path.join(path, 'test.txt'))


def tokenize(dictionary, path):
    """Tokenizes a text file for training or testing to a sequence of indices format
       We assume that training and test data has <eos> symbols """
    assert os.path.exists(path)
    with open(path, 'r', encoding="utf8") as f:
        ntokens = 0
        for line in f:
            words = line.split()
            ntokens += len(words)

    # Tokenize file content
    with open(path, 'r', encoding="utf8") as f:
        ids = torch.LongTensor(ntokens)
        token = 0
        for line in f:
            words = line.split()
            for word in words:
                if word in dictionary.word2idx:
                    ids[token] = dictionary.word2idx[word]
                else:
                    ids[token] = dictionary.word2idx["<unk>"]
                token += 1

    return ids


