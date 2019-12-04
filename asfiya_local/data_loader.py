#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torchvision.models as models


# In[2]:


#inception = models.inception_v3(pretrained=True)


# In[3]:


#googlenet = models.googlenet(pretrained=True)


# In[4]:


from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import pylab
import os
import json
import torchvision.datasets as dset
import torchvision.datasets.coco as co
#from dest1 import CocoCaptions
import torch.utils.data as td
import torchvision.transforms as transforms
import nltk
import pickle
import torch
from vocabbuild import Vocabulary

nltk.download('punkt')

class COCODataset(co.CocoCaptions):
    
    def __init__(self, root, annFile,vocabulary, transform=None, target_transform=None, transforms=None):
        
        super().__init__(root, annFile, transform, target_transform)
        self.vocabulary = vocabulary
    
    def __getitem__(self, index):
        img,caption = super().__getitem__(index)
        print(caption)
        coco = COCO(json)
        caption_test = coco.anns[ann_id]['caption']
        print(caption_test)
        
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        #print(tokens)
        # append start and end 
        captions =[]
        captions.append(self.vocabulary('<<start>>'))
        for i in tokens:
            captions.append(self.vocabulary(i))
        captions.append(self.vocabulary('<<end>>'))
        captions = torch.Tensor(captions)
        return img,captions
    
    def __len__(self):
        return super().__len__()

def coco_batch(coco_data):
    '''
    create mini_batch tensors from the list of tuples, this is to match the output of __getitem__()
    coco_data: list of tuples of length 2:
        coco_data[0]: image, shape of (3, 256, 256)
        coco_data[1]: caption, shape of length of the caption;
    '''
    # Sort Descending by caption length
    coco_data.sort(key=lambda x: len(x[1]), reverse = True)
    images, captions = zip(*coco_data)
    
    # turn images to a 4D tensor with specified batch_size
    images = torch.stack(images, 0)
    
    # do the same thing for caption. -> 2D tensor with equal length of max lengths in batch, padded with 0
    cap_length = [len(cap) for cap in captions]
    targets = torch.LongTensor(np.zeros((len(captions), max(cap_length))))
    for i, cap in enumerate(captions):
        length = cap_length[i]
        targets[i, :length] = cap[:length]
    
    return images, targets, cap_length

