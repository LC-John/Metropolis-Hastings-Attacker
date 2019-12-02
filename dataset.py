#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import random
import numpy
from util import reverse_seq

class Dictionary(object):
    
    def __init__(self, dict_path, dict_size=30000):
        
        with open(dict_path, "rb") as f:
            self.__dict = pickle.load(f)
            self.__dict_size = dict_size
            self.__dict = self.__dict[:self.__dict_size]
            assert len(self.__dict) == self.__dict_size
            self.__dict_unk = self.__dict_size      # <UNK>
            self.__dict_pad = self.__dict_size + 1  # <PAD>
            self.__dict_init = self.__dict.index("<__INIT__>")
            
    def get_vocab_idx(self, s):
        
        if s in self.__dict:
            return self.__dict.index(s)
        else:
            return self.__dict_unk
        
    def get_vocab(self, idx):
        
        if idx < self.__dict_size:
            return self.__dict[idx]
        elif idx == self.__dict_unk:
            return "<__UNK__>"
        elif idx == self.__dict_pad:
            return "<__PAD__>"
        else:
            return None
        
    def get_dict_size(self):
        
        return self.__dict_size + 2
    
    def get_init_idx(self):
        
        return self.__dict_init
    
    def get_unk_idx(self):
        
        return self.__dict_unk
    
    def get_pad_idx(self):
        
        return self.__dict_pad

class SeqClassificationDataset(object):
    
    def __init__(self, input_data_path, output_data_path, dict_path,
                 max_seq_len=100, dict_size=30000, train_ratio=1.0):
        
        self.__max_seq_len = max_seq_len
        
        with open(input_data_path, "rb") as f:
            X = pickle.load(f)
        with open(output_data_path, "rb") as f:
            Y = pickle.load(f)
        with open(dict_path, "rb") as f:
            self.__dict = pickle.load(f)
            self.__dict_size = dict_size
            self.__dict = self.__dict[:self.__dict_size]
            assert len(self.__dict) == self.__dict_size
            self.__dict_unk = self.__dict_size      # <UNK>
            self.__dict_ph = self.__dict_size + 1   # <PH>
            self.__dict_pad = self.__dict_size + 2  # <PAD>
            
        print ("DATA LOADED!")
            
        assert len(X) == len(Y)
        self.__L = []
        self.__X = []
        self.__Y = []
        for i in range(len(X)):
            if len(X[i]) <= max_seq_len:
                self.__L.append(len(X[i]))
                tmp_l = len(X[i])
            else:
                self.__L.append(max_seq_len)
                tmp_l = max_seq_len
            tmp_x = []
            for j in range(max_seq_len):
                if j >= tmp_l:
                    tmp_x.append(self.__dict_pad)
                elif X[i][j] < self.__dict_unk:
                    tmp_x.append(X[i][j])
                else:
                    tmp_x.append(self.__dict_unk)
            self.__X.append(tmp_x)
            self.__Y.append(Y[i])
            if ((i+1) % (len(X)/100) == 0):
                print ("\rPROCESSING... %d/%d" % (i+1, len(X)), end="")
        
        split_idx = int(train_ratio * len(self.__X))
        self.__X_te = self.__X[split_idx:]
        self.__X = self.__X[:split_idx]
        self.__Y_te = self.__Y[split_idx:]
        self.__Y = self.__Y[:split_idx]
        self.__L_te = self.__L[split_idx:]
        self.__L = self.__L[:split_idx]
        self.__train_size = len(self.__X)
        self.__test_size = len(self.__X_te)
        
        self.__epoch = None
        self.__epoch_te = None
        self.reset_train_epoch()
        self.reset_test_epoch()
        
        print ("\nDone!")
        
    def minibatch(self, batch_size, ph=False):
        
        new_epoch = False
        assert batch_size <= self.__train_size
        if batch_size > len(self.__epoch):
            new_epoch = True
            self.reset_train_epoch()
        
        idx = self.__epoch[:batch_size]
        self.__epoch = self.__epoch[batch_size:]
        x = []
        y = []
        l = []
        for i in idx:
            if ph:
                tmp_idx = random.randint(0, self.__L[i])
                tmp_x = self.__X[i]
                tmp_x = tmp_x[:tmp_idx] + [self.__dict_ph] + tmp_x[tmp_idx:]
                tmp_x = tmp_x[:-1]
                x.append(tmp_x)
                if self.__L[i] >= self.__max_seq_len:
                    l.append(self.__L[i])
                else:
                    l.append(self.__L[i] + 1)
            else:
                x.append(self.__X[i])
                l.append(self.__L[i])
            y.append(self.__Y[i])
        return (numpy.asarray(x, dtype=numpy.int32),
                numpy.asarray(y, dtype=numpy.int32),
                numpy.asarray(l, dtype=numpy.int32), new_epoch)
        
    def test_batch(self, batch_size):
        
        new_epoch = False
        assert batch_size <= self.__test_size
        if batch_size > len(self.__epoch):
            new_epoch = True
            self.reset_test_epoch()
        
        idx = self.__epoch_te[:batch_size]
        self.__epoch_te = self.__epoch_te[batch_size:]
        x = []
        y = []
        l = []
        for i in idx:
            x.append(self.__X_te[i])
            y.append(self.__Y_te[i])
            l.append(self.__L_te[i])
        return (numpy.asarray(x, dtype=numpy.int32),
                numpy.asarray(y, dtype=numpy.int32),
                numpy.asarray(l, dtype=numpy.int32), new_epoch)
        
    def get_vocab_idx(self, s):
        
        if s in self.__dict:
            return self.__dict.index(s)
        elif s == "<__PH__>":
            return self.__dict_ph
        else:
            return self.__dict_unk
        
    def get_vocab(self, idx):
        
        if idx < self.__dict_size:
            return self.__dict[idx]
        elif idx == self.__dict_unk:
            return "<__UNK__>"
        elif idx == self.__dict_pad:
            return "<__PAD__>"
        elif idx == self.__dict_ph:
            return "<__PH__>"
        else:
            return None
        
    def get_whole_dataset(self):
        
        return (self.__X, self.__Y, self.__X_te, self.__Y_te)
        
    def get_dict_size(self):
        
        return self.__dict_size + 3
    
    def get_train_size(self):
        
        return self.__train_size
    
    def get_test_size(self):
        
        return self.__test_size
    
    def reset_train_epoch(self):
        
        self.__epoch = random.sample(list(range(self.__train_size)),
                                     self.__train_size)
        
    def reset_test_epoch(self):
        
        self.__epoch_te = random.sample(list(range(self.__test_size)),
                                        self.__test_size)

class Dataset(object):
    
    def __init__(self, input_data_path, output_data_path, dict_path,
                 max_seq_len=100, dict_size=30000, train_ratio=0.8, processed=False):
        
        self.__max_seq_len = max_seq_len
        
        with open(input_data_path, "rb") as f:
            X = pickle.load(f)
        with open(output_data_path, "rb") as f:
            Y = pickle.load(f)
        with open(dict_path, "rb") as f:
            self.__dict = pickle.load(f)
            self.__dict_size = dict_size
            self.__dict = self.__dict[:self.__dict_size]
            assert len(self.__dict) == self.__dict_size
            self.__dict_unk = self.__dict_size      # <UNK>
            self.__dict_pad = self.__dict_size + 1  # <PAD>
            self.__dict_init = self.__dict.index("<__INIT__>")
            
        print ("DATA LOADED!")
            
        assert len(X) == len(Y)
        self.__L = []
        self.__X = []
        self.__Y = []
        for i in range(len(X)):
            assert len(X[i]) == len(Y[i])
            if len(X[i]) <= max_seq_len:
                self.__L.append(len(X[i]))
                tmp_l = len(X[i])
            else:
                self.__L.append(max_seq_len)
                tmp_l = max_seq_len
            tmp_x = []
            tmp_y = []
            for j in range(max_seq_len):
                if processed:
                    if j >= tmp_l:
                        tmp_x.append(self.__dict_pad)
                    elif X[i][j] in ["<__INIT__>"]:
                        tmp_x.append(self.__dict_init)
                    elif X[i][j] < self.__dict_unk:
                        tmp_x.append(X[i][j])
                    else:
                        tmp_x.append(self.__dict_unk)
                    if j >= tmp_l:
                        tmp_y.append(self.__dict_pad)
                    elif Y[i][j] in ["<__INIT__>"]:
                        tmp_y.append(self.__dict_init)
                    elif Y[i][j] < self.__dict_unk:
                        tmp_y.append(Y[i][j])
                    else:
                        tmp_y.append(self.__dict_unk)
                else:
                    if j >= tmp_l:
                        tmp_x.append(self.__dict_pad)
                    elif X[i][j] in self.__dict:
                        tmp_x.append(self.__dict.index(X[i][j]))
                    else:
                        tmp_x.append(self.__dict_unk)
                    if j >= tmp_l:
                        tmp_y.append(self.__dict_pad)
                    elif Y[i][j] in self.__dict:
                        tmp_y.append(self.__dict.index(Y[i][j]))
                    else:
                        tmp_y.append(self.__dict_unk)
            self.__X.append(tmp_x)
            self.__Y.append(tmp_y)
            if ((i+1) % (len(X)/100) == 0):
                print ("\rPROCESSING... %d/%d" % (i+1, len(X)), end="")
            
        split_idx = int(train_ratio * len(self.__X))
        self.__X_te = self.__X[split_idx:]
        self.__X = self.__X[:split_idx]
        self.__Y_te = self.__Y[split_idx:]
        self.__Y = self.__Y[:split_idx]
        self.__L_te = self.__L[split_idx:]
        self.__L = self.__L[:split_idx]
        self.__train_size = len(self.__X)
        self.__test_size = len(self.__X_te)
        
        self.__epoch = None
        self.__epoch_te = None
        self.reset_train_epoch()
        self.reset_test_epoch()
        
        print ("\nDone!")
        
    def minibatch(self, batch_size):
        
        new_epoch = False
        assert batch_size <= self.__train_size
        if batch_size > len(self.__epoch):
            new_epoch = True
            self.reset_train_epoch()
        
        idx = self.__epoch[:batch_size]
        self.__epoch = self.__epoch[batch_size:]
        x = []
        y = []
        l = []
        for i in idx:
            x.append(self.__X[i])
            y.append(self.__Y[i])
            l.append(self.__L[i])
        return (numpy.asarray(x, dtype=numpy.int32),
                numpy.asarray(y, dtype=numpy.int32),
                numpy.asarray(l, dtype=numpy.int32), new_epoch)
        
    def minibatch_rev(self, batch_size):
        
        x, y, l, n = self.minibatch(batch_size)
        x_rev, y_rev = reverse_seq(x, y, l, self.__dict_init, self.__dict_pad)
        return (numpy.asarray(x_rev, dtype=numpy.int32),
                numpy.asarray(y_rev, dtype=numpy.int32),
                numpy.asarray(l, dtype=numpy.int32), n)
        
    def test_batch(self, batch_size):
        
        new_epoch = False
        assert batch_size <= self.__test_size
        if batch_size > len(self.__epoch):
            new_epoch = True
            self.reset_test_epoch()
        
        idx = self.__epoch_te[:batch_size]
        self.__epoch_te = self.__epoch_te[batch_size:]
        x = []
        y = []
        l = []
        for i in idx:
            x.append(self.__X_te[i])
            y.append(self.__Y_te[i])
            l.append(self.__L_te[i])
        return (numpy.asarray(x, dtype=numpy.int32),
                numpy.asarray(y, dtype=numpy.int32),
                numpy.asarray(l, dtype=numpy.int32), new_epoch)
        
    def test_batch_rev(self, batch_size):
        
        x, y, l, n = self.test_batch(batch_size)
        x_rev, y_rev = reverse_seq(x, y, l, self.__dict_init, self.__dict_pad)
        return (numpy.asarray(x_rev, dtype=numpy.int32),
                numpy.asarray(y_rev, dtype=numpy.int32),
                numpy.asarray(l, dtype=numpy.int32), n)
        
    def get_vocab_idx(self, s):
        
        if s in self.__dict:
            return self.__dict.index(s)
        else:
            return self.__dict_unk
        
    def get_vocab(self, idx):
        
        if idx < self.__dict_size:
            return self.__dict[idx]
        elif idx == self.__dict_unk:
            return "<__UNK__>"
        elif idx == self.__dict_pad:
            return "<__PAD__>"
        else:
            return None
        
    def get_whole_dataset(self):
        
        return (self.__X, self.__Y, self.__X_te, self.__Y_te)
        
    def get_dict_size(self):
        
        return self.__dict_size + 2
    
    def get_train_size(self):
        
        return self.__train_size
    
    def get_test_size(self):
        
        return self.__test_size
    
    def reset_train_epoch(self):
        
        self.__epoch = random.sample(list(range(self.__train_size)),
                                     self.__train_size)
        
    def reset_test_epoch(self):
        
        self.__epoch_te = random.sample(list(range(self.__test_size)),
                                        self.__test_size)
        
if __name__ == "__main__":
    
    seq_max_len = 200
    vocab_size = 30000
    tr = SeqClassificationDataset("./imdb/train_input.pkl",
                                  "./imdb/train_output.pkl",
                                  "./imdb/vocab.pkl",
                                  seq_max_len, vocab_size, 1.0)
    te = SeqClassificationDataset("./imdb/test_input.pkl",
                                  "./imdb/test_output.pkl",
                                  "./imdb/vocab.pkl",
                                  seq_max_len, vocab_size, 0.0)