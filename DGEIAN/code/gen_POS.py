# -*- coding: utf-8 -*-
import json
import pickle

import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize


def poslable(text):
    text = nltk.word_tokenize(text)
    poslable = nltk.pos_tag(text)
    return poslable

def process(filename):
    idx2pos = {}
    fout = open(filename + '.testpos', 'wb')
    sentence_packs = json.load(open(filename))
    for sentence_pack in sentence_packs:
        pos = poslable(sentence_pack['sentence'])
        idx2pos[sentence_pack['id']] = pos

    pickle.dump(idx2pos, fout)
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