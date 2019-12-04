import nltk
from collections import Counter
from pycocotools.coco import COCO
import logging
import numpy as np
import pickle

class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<<unknown>>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


if __name__ == "__main__":

    json = "annotations/captions_train2014.json"
    portion = 0.995 
    # Manually setting threshold
    threshold = 4
    save_path = "vocab.pkl"
    # construct coco instance
    coco = COCO(json)
    ids = coco.anns.keys()
    counter = Counter()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)
        if (i+1) % 5000 == 0:
            print("Tokenization Process: {0:.2f}%.".format((i+1)*100/len(ids)))
            
    words = []
    for word, count in counter.items():
        if count >= threshold:
            words.append(word)

    vocab = Vocabulary()
    # for padding purpose
    vocab.add_word('<<padding>>')
    vocab.add_word('<<start>>')
    vocab.add_word('<<end>>')
    vocab.add_word('<<unknown>>')

    # Add the words to the vocabulary.
    for word in words:
        vocab.add_word(word)

    with open(save_path, 'wb') as f:
        pickle.dump(vocab, f)





