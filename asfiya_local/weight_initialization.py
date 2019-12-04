#!/usr/bin/env python
# coding: utf-8

# In[3]:


from vocabbuild import Vocabulary
import pickle
import numpy as np
import bcolz


# In[10]:


def get_weight_matrix():
    vectors = bcolz.open('../6B.300.dat')[:]
    words = pickle.load(open('../6B.300_words.pkl', 'rb'))
    word2idx = pickle.load(open('../6B.300_idx.pkl', 'rb'))

    glove = {w: vectors[word2idx[w]] for w in words}

    with open("../vocab.pkl", 'rb') as fi:
        vocabulary = pickle.load(fi)
        
    target_vocab = []
    for i in range(len(vocabulary)):
        target_vocab.append(vocabulary.idx2word[i])
        
    matrix_len = len(target_vocab)
    weights_matrix = np.zeros((matrix_len, 300))
    words_found = 0

    for i, word in enumerate(target_vocab):
        try: 
            weights_matrix[i] = glove[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(300, ))
            
    return weights_matrix


# In[11]:


wmat = get_weight_matrix()
print(wmat.shape)


# In[4]:





# In[ ]:





