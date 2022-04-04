import math
import pickle

import numpy as np
import torch
import spacy
nlp = spacy.load('en_core_web_sm')

sentiment2id = {'negative': 3, 'neutral': 4, 'positive': 5}

def get_spans(tags):
    '''for BIO tag'''
    tags = tags.strip().split()
    length = len(tags)
    spans = []
    start = -1
    for i in range(length):
        if tags[i].endswith('B'):
            if start != -1:
                spans.append([start, i - 1])
            start = i
        elif tags[i].endswith('O'):
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, length - 1])
    return spans

def getpos(tags,args):
    pos = torch.zeros(args.max_sequence_len).long()
    for i, tags in enumerate(tags):
        word, tag = tags
        if tag.startswith('NN'):
            pos[i] = 1
        elif tag.startswith('VB'):
            pos[i] = 2
        elif tag.startswith('JJ'):
            pos[i] = 3
        elif tag.startswith('RB'):
            pos[i] = 4
        else:
            pos[i] = 0
    return pos


class Instance(object):
    def __init__(self, sentence_pack, word2index, args, fname):
        self.id = sentence_pack['id']
        self.sentence = sentence_pack['sentence']
        self.sentence_tokens = torch.zeros(args.max_sequence_len).long()
        words = self.sentence.split()
        self.length = len(words)
        for i, w in enumerate(words):
            word = w
            if word in word2index:
                self.sentence_tokens[i] = word2index[word]
            else:
                self.sentence_tokens[i] = word2index['<unk>']

        f = open(fname + '.pos', 'rb')
        idx2pos = pickle.load(f)
        f.close()
        for key in idx2pos.keys():
            if key == self.id:
                self.sentence_pos = getpos(idx2pos[key],args)
                break


        self.sentence_adj = torch.zeros(self.length, self.length).long()
        fin = open(fname + '.graph', 'rb')
        idx2graph = pickle.load(fin)
        fin.close()
        for key in idx2graph.keys():
            if key == self.id:
                self.sentence_adj = idx2graph[key]
                break

        self.aspect_tags = torch.zeros(args.max_sequence_len).long()
        self.opinion_tags = torch.zeros(args.max_sequence_len).long()
        self.aspect_tags[self.length:] = -1
        self.opinion_tags[self.length:] = -1
        self.tags = torch.zeros(args.max_sequence_len, args.max_sequence_len).long()
        self.tags[:, :] = -1


        for i in range(self.length):
            for j in range(i, self.length):
                self.tags[i][j] = 0
        for pair in sentence_pack['triples']:
            aspect = pair['target_tags']
            opinion = pair['opinion_tags']
            aspect_span = get_spans(aspect)
            opinion_span = get_spans(opinion)

            for l, r in aspect_span:
                for i in range(l, r+1):
                    self.aspect_tags[i] = 1 if i == l else 2
                    self.tags[i][i] = 1
                    if i > l: self.tags[i-1][i] = 1
                    for j in range(i, r+1):
                        self.tags[i][j] = 1
            for l, r in opinion_span:
                for i in range(l, r+1):
                    self.opinion_tags[i] = 1 if i == l else 2
                    self.tags[i][i] = 2
                    if i > l: self.tags[i-1][i] = 2
                    for j in range(i, r+1):
                        self.tags[i][j] = 2

            for al, ar in aspect_span:
                for pl, pr in opinion_span:
                    for i in range(al, ar+1):
                        for j in range(pl, pr+1):
                            if args.task == 'pair':
                                if i > j: self.tags[j][i] = 3
                                else: self.tags[i][j] = 3
                            elif args.task == 'triplet':
                                if i > j: self.tags[j][i] = sentiment2id[pair['sentiment']]
                                else: self.tags[i][j] = sentiment2id[pair['sentiment']]

        '''generate mask of the sentence'''
        self.mask = torch.zeros(args.max_sequence_len)
        self.mask[:self.length] = 1


def load_data_instances(sentence_packs, word2index, args, fname):
    instances = list()
    for sentence_pack in sentence_packs:
        instances.append(Instance(sentence_pack, word2index, args, fname))
    return instances


class DataIterator(object):
    def __init__(self, instances, args):
        self.instances = instances
        self.args = args
        self.batch_count = math.ceil(len(instances)/args.batch_size)

    def get_batch(self, index):
        sentence_ids = []
        sentence_tokens = []
        sentence_poses = []
        sentence_adjs = []
        lengths = []
        masks = []
        aspect_tags = []
        opinion_tags = []
        tags = []

        for i in range(index * self.args.batch_size,
                       min((index + 1) * self.args.batch_size, len(self.instances))):


            sentence_ids.append(self.instances[i].id)
            sentence_tokens.append(self.instances[i].sentence_tokens)
            lengths.append(self.instances[i].length)
            masks.append(self.instances[i].mask)
            aspect_tags.append(self.instances[i].aspect_tags)
            opinion_tags.append(self.instances[i].opinion_tags)
            tags.append(self.instances[i].tags)

        max_len = max(lengths)

        for i in range(index * self.args.batch_size,
                       min((index + 1) * self.args.batch_size, len(self.instances))):
            pos = self.instances[i].sentence_pos
            sentence_poses.append(pos)

        for i in range(index * self.args.batch_size,
                       min((index + 1) * self.args.batch_size, len(self.instances))):
            len_s = self.instances[i].length
            a = self.instances[i].sentence_adj
            adj = np.pad(a, ((0,max_len-len_s),(0,max_len-len_s)), 'constant')
            adj = torch.from_numpy(adj)
            sentence_adjs.append(adj)

        indexes = list(range(len(sentence_tokens)))
        indexes = sorted(indexes, key=lambda x: lengths[x], reverse=True)


        sentence_ids = [sentence_ids[i] for i in indexes]
        sentence_tokens = torch.stack(sentence_tokens).to(self.args.device)[indexes]
        sentence_poses = torch.stack(sentence_poses).to(self.args.device)[indexes]
        sentence_adjs = torch.stack(sentence_adjs).to(self.args.device)[indexes]
        lengths = torch.tensor(lengths).to(self.args.device)[indexes]
        masks = torch.stack(masks).to(self.args.device)[indexes]
        aspect_tags = torch.stack(aspect_tags).to(self.args.device)[indexes]
        opinion_tags = torch.stack(opinion_tags).to(self.args.device)[indexes]
        tags = torch.stack(tags).to(self.args.device)[indexes]

        return sentence_ids, sentence_tokens, sentence_poses, sentence_adjs, lengths, masks, aspect_tags, opinion_tags, tags
