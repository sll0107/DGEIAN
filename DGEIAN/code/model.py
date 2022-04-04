import torch
import torch.nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

from attention_module import MultiHeadedAttention, SelfAttention
from jiaohu_attention import MultiHeadedAttention, jiaohu_SelfAttention
from gcn import GCNModel, GraphConvolution

class MultiInferRNNModel(torch.nn.Module):
    def __init__(self, gen_emb, domain_emb, args):
        '''double embedding + lstm encoder + dot self attention'''
        super(MultiInferRNNModel, self).__init__()
        self.args = args
        self.gen_embedding = torch.nn.Embedding(gen_emb.shape[0], gen_emb.shape[1])
        self.gen_embedding.weight.data.copy_(gen_emb)
        self.gen_embedding.weight.requires_grad = False
        self.domain_embedding = torch.nn.Embedding(domain_emb.shape[0], domain_emb.shape[1])
        self.domain_embedding.weight.data.copy_(domain_emb)
        self.domain_embedding.weight.requires_grad = False

        self.pos_embedding1 = torch.nn.Embedding(args.max_sequence_len, args.pos_dim)
        self.pos_embedding2 = torch.nn.Embedding(5, args.pos_dim)

        self.dropout1 = torch.nn.Dropout(0.3)
        self.dropout2 = torch.nn.Dropout(0.1)

        self.bilstm = torch.nn.LSTM(300+100+100, args.lstm_dim,
                                    num_layers=1, batch_first=True, bidirectional=True)
        self.bilstm1 = torch.nn.LSTM(args.lstm_dim * 2, args.lstm_dim,
                                    num_layers=1, batch_first=True, bidirectional=True)
        self.gcn_layer = GCNModel(args)
        self.attention_layer = SelfAttention(args)
        self.jiaohu_attention = jiaohu_SelfAttention(args)

        self.feature_linear = torch.nn.Linear(args.lstm_dim*4 + args.class_num*3, args.lstm_dim*4)
        self.cls_linear = torch.nn.Linear(args.lstm_dim*4, args.class_num)

    def _get_embedding1(self, sentence_tokens, mask):
        gen_embed = self.gen_embedding(sentence_tokens)
        domain_embed = self.domain_embedding(sentence_tokens)
        embedding = torch.cat([gen_embed, domain_embed], dim=2)
        embedding = self.dropout1(embedding)
        embedding = embedding * mask.unsqueeze(2).float().expand_as(embedding)
        return embedding

    def _get_embedding2(self, sentence_poses, mask):
        pos_embed = self.pos_embedding2(sentence_poses)
        pos_embed = self.dropout1(pos_embed)
        embedding = pos_embed * mask.unsqueeze(2).float().expand_as(pos_embed)
        return embedding

    def _lstm_feature(self, embedding, lengths):
        embedding = pack_padded_sequence(embedding, lengths.cpu(), batch_first=True)
        context, _ = self.bilstm(embedding)
        context, _ = pad_packed_sequence(context, batch_first=True)
        return context

    def _lstm_feature1(self, embedding, lengths):
        embedding = pack_padded_sequence(embedding, lengths.cpu(), batch_first=True)
        context, _ = self.bilstm1(embedding)
        context, _ = pad_packed_sequence(context, batch_first=True)

        return context

    def _cls_logits(self, features):
        tags = self.cls_linear(features)
        return tags

    def final_logits(self, features, lengths, mask):
        max_length = features.shape[1]
        mask = mask[:, :max_length]
        mask_a = mask.unsqueeze(1).expand([-1, max_length, -1])
        mask_b = mask.unsqueeze(2).expand([-1, -1, max_length])
        mask = mask_a * mask_b
        mask = torch.triu(mask).unsqueeze(3).expand([-1, -1, -1, self.args.class_num])
        logits_list = []
        logits = self._cls_logits(features)
        logits_list.append(logits)
        return logits_list



    def forward(self, sentence_tokens, sentence_poses, sentence_adjs, lengths, mask):
        embedding1 = self._get_embedding1(sentence_tokens, mask)
        embedding2 = self._get_embedding2(sentence_poses, mask)
        embedding = torch.cat([embedding1, embedding2], dim=2)
        lstm_feature = self._lstm_feature(embedding, lengths)
        a = lstm_feature
        lstm_feature_gcn = self.gcn_layer(lstm_feature, sentence_adjs, mask[:, :lengths[0]])
        b = lstm_feature_gcn
        jiaohu_num = 3
        for i in range(jiaohu_num):
            lstm_feature_jiaohu_a = self.jiaohu_attention(lstm_feature, lstm_feature, mask[:, :lengths[0]])
            lstm_feature_gcn_jiaohu_a = self.jiaohu_attention(lstm_feature_gcn, lstm_feature_gcn, mask[:, :lengths[0]])

            lstm_feature_jiaohu = torch.bmm(lstm_feature_gcn_jiaohu_a, lstm_feature)
            lstm_feature_jiaohu = lstm_feature_jiaohu * mask[:, :lengths[0]].unsqueeze(2).float().expand_as(lstm_feature_jiaohu)

            lstm_feature_gcn_jiaohu = torch.bmm(lstm_feature_jiaohu_a, lstm_feature_gcn)
            lstm_feature_gcn_jiaohu = lstm_feature_gcn_jiaohu * mask[:, :lengths[0]].unsqueeze(2).float().expand_as(lstm_feature_gcn_jiaohu)

            lstm_feature_gcn_jiaohu_drop2 = self.dropout2(lstm_feature_gcn_jiaohu) + b
            lstm_feature_jiaohu_drop2 = self.dropout2(lstm_feature_jiaohu) + a

            lstm_feature = self._lstm_feature1(lstm_feature_jiaohu_drop2, lengths)
            lstm_feature_gcn = self.gcn_layer(lstm_feature_gcn_jiaohu_drop2, sentence_adjs, mask[:, :lengths[0]])
        lstm_feature = a + lstm_feature
        lstm_feature_attention = self.attention_layer(lstm_feature, lstm_feature, mask[:, :lengths[0]])
        lstm_feature = a + lstm_feature + lstm_feature_attention
        lstm_feature = lstm_feature.unsqueeze(2).expand([-1, -1, lengths[0], -1])
        lstm_feature_T = lstm_feature.transpose(1, 2)
        features = torch.cat([lstm_feature, lstm_feature_T], dim=3)
        logits = self.final_logits(features, lengths, mask)
        return [logits[-1]]


