from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import json
from Voc import Voc


MAX_LENGTH = 10

class Proc:
 
    def __init__(self, MAX_LENGTH, MIN_COUNT):
        self.MAX_LENGTH = MAX_LENGTH
        self.MIN_COUNT = MIN_COUNT
        self.PAD_token = 0
        self.EOS_token = 2

    def unicodeToAscii(self, s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    # Lowercase, trim, and remove non-letter characters
    def normalizeString(self, s):
        s = self.unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^а-яА-Я.!?]+", r" ", s)
        s = re.sub(r"\s+", r" ", s).strip()
        return s

    # Read query/response pairs and return a voc object
    def readVocs(self, datafile, corpus_name):
        print("Reading lines...")
        # Read the file and split into lines
        lines = open(datafile, encoding='utf-8').\
        read().strip().split('\n')
        # Split every line into pairs and normalize
        pairs = [[self.normalizeString(s) for s in l.split('\t')] for l in lines]
        voc = Voc(corpus_name)
        return voc, pairs

    def filterPair(self, p):
        return len(p[0].split(' ')) < self.MAX_LENGTH and len(p[1].split(' ')) < self.MAX_LENGTH

    def filterPairs(self, pairs):
        print(pairs)
        return [pair for pair in pairs if self.filterPair(pair)]

    def loadPrepareData(self, corpus, corpus_name, datafile, save_dir):
        print("Start preparing training data ...")
        voc, pairs = self.readVocs(datafile, corpus_name)

        print("Read {!s} sentence pairs".format(len(pairs)))
        pairs = self.filterPairs(pairs)
        print("Trimmed to {!s} sentence pairs".format(len(pairs)))
        print("Counting words...")
        for pair in pairs:
            voc.addSentence(pair[0])
            voc.addSentence(pair[1])
        print("Counted words:", voc.num_words)
        return voc, pairs

    def trimRareWords(self, voc, pairs):
        voc.trim(self.MIN_COUNT)
        keep_pairs = []
        for pair in pairs:
            input_sentence = pair[0]
            output_sentence = pair[1]
            keep_input = True
            keep_output = True
            for word in input_sentence.split(' '):
                if word not in voc.word2index:
                    keep_input = False
                    break
            for word in output_sentence.split(' '):
                if word not in voc.word2index:
                    keep_output = False
                    break

            if keep_input and keep_output:
                keep_pairs.append(pair)

        print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
        return keep_pairs

    def indexesFromSentence(self, voc, sentence):
        return [voc.word2index[word] for word in sentence.split(' ')] + [self.EOS_token]


    def zeroPadding(self, l, fillvalue=0):
        return list(itertools.zip_longest(*l, fillvalue=fillvalue))

    def binaryMatrix(self, l, value=0):
        m = []
        for i, seq in enumerate(l):
            m.append([])
            for token in seq:
                if token == self.PAD_token:
                    m[i].append(0)
                else:
                    m[i].append(1)
        return m

    def inputVar(self, l, voc):
        indexes_batch = [self.indexesFromSentence(voc, sentence) for sentence in l]
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        padList = self.zeroPadding(indexes_batch)
        padVar = torch.LongTensor(padList)
        return padVar, lengths

    def outputVar(self, l, voc):
        indexes_batch = [self.indexesFromSentence(voc, sentence) for sentence in l]
        max_target_len = max([len(indexes) for indexes in indexes_batch])
        padList = self.zeroPadding(indexes_batch)
        mask = self.binaryMatrix(padList)
        mask = torch.BoolTensor(mask)
        padVar = torch.LongTensor(padList)
        return padVar, mask, max_target_len

    def batch2TrainData(self, voc, pair_batch):
        pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
        input_batch, output_batch = [], []
        for pair in pair_batch:
            input_batch.append(pair[0])
            output_batch.append(pair[1])
        inp, lengths = self.inputVar(input_batch, voc)
        output, mask, max_target_len = self.outputVar(output_batch, voc)
        return inp, lengths, output, mask, max_target_len
