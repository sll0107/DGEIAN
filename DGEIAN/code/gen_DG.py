# -*- coding: utf-8 -*-
import json

import numpy as np
import spacy
import torch
import pickle

from spacy.tokens import Doc

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

def dependency_adj_matrix(text):
    tokens = nlp(text)
    words = text.split()
    matrix = np.zeros((len(words), len(words))).astype('float32')
    assert len(words) == len(list(tokens))

    for token in tokens:
        matrix[token.i][token.i] = 1
        for child in token.children:
            matrix[token.i][child.i] = 1
            matrix[child.i][token.i] = 1

    return matrix

def process(filename):
    idx2graph = {}
    fout = open(filename + '.graph', 'wb')
    sentence_packs = json.load(open(filename))
    for sentence_pack in sentence_packs:
        adj_matrix = dependency_adj_matrix(sentence_pack['sentence'])
        idx2graph[sentence_pack['id']] = adj_matrix
    pickle.dump(idx2graph, fout)
    fout.close()

if __name__ == '__main__':
    process('../../data/lap14/train.json')
    process('../../data/lap14/test.json')
    process('../../data/lap14/dev.json')
    process('../../data/res14/train.json')
    process('../../data/res14/test.json')
    process('../../data/res14/dev.json')
    process('../../data/res15/train.json')
    process('../../data/res15/test.json')
    process('../../data/res15/dev.json')
    process('../../data/res16/train.json')
    process('../../data/res16/test.json')
    process('../../data/res16/dev.json')