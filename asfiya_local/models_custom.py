#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
import time
import torch
from torch import nn
import torch.utils.data as td
import torchvision as tv
import bcolz
import numpy as np
import pickle

def glo2vec():
    words = []
    idx = 0
    word2idx = {}
    vectors = bcolz.carray(np.zeros(1), rootdir='6B.300.dat', mode='w')

    with open('glove.6B.300d.txt', 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)

    vectors = bcolz.carray(vectors[1:].reshape((400000, 300)), rootdir='6B.300.dat', mode='w')
    vectors.flush()
    pickle.dump(words, open('6B.300_words.pkl', 'wb'))
    pickle.dump(word2idx, open('6B.300_idx.pkl', 'wb'))


    


# In[11]:


class CNN(nn.Module):
    
    def __init__(self,out_classes,fine_tuning = False):
        super().__init__()
        inception = tv.models.inception_v3(pretrained = True)
        inception.aux_logits=False
        for param in inception.parameters():
            param.requires_grad = fine_tuning
        self.net = inception
        self.net.fc = nn.Linear(inception.fc.in_features,out_classes)
    
    def forward(self, x):
        return self.net(x)


# In[12]:


def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


# In[15]:


# c = CNN(10)
# print(c)

class LSTM_custom(nn.Module):
    def __init__(self, weights_matrix, hidden_size, vocab_size, num_layers):
        super(self, LSTM_custom).__init__()
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
    
    def forward(self, features, captions, lengths):
        embeddings = self.embeddings(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
        
        


# In[26]:


# inception = tv.models.inception_v3(pretrained = True)


# In[25]:


# fine_tuning = False
# print(inception)
# for param in inception.parameters():
#     param.requires_grad = fine_tuning


