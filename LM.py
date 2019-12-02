#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy

class LanguageModel(object):
    
    def __init__(self, seq_max_len=20, embed_w=300, vocab_size=30000,
                 n_layer=2, n_hidden=300, keep_prob=0.9, lr=5e-4,
                 n_gpu=[0], grad_clip=1, scope="", is_training=True):
        
        assert len(n_gpu) > 0
        
        self.__is_training = is_training
        if is_training == False:
            is_training = True
            keep_prob = 1.0
        
        # placeholders for the language model -- input, output & valid seq length
        self.__X = tf.placeholder(shape=[None, seq_max_len],
                                  dtype=tf.int32, name="input")
        self.__Y = tf.placeholder(shape=[None, seq_max_len],
                                  dtype=tf.int32, name="target")
        self.__L = tf.placeholder(shape=[None], dtype=tf.int32, name="valid_len")
        
        # embedding matrix
        with tf.device("/cpu:0"):
            self.__embed_matrix = tf.get_variable("embedding_martix",
                                                  [vocab_size, embed_w],
                                                  dtype=tf.float32)
        
        # rnn architecture
        self.__cell_list_fw = [self.__make_cell(is_training, n_hidden, keep_prob, i)
                                for i in range(n_layer)]
        self.__cell_fw = tf.contrib.rnn.MultiRNNCell(self.__cell_list_fw)
        
        self.__dense_W = tf.Variable(tf.random_normal([n_hidden, vocab_size]),
                                     name="dense_w")
        self.__dense_b = tf.Variable(tf.constant(0, dtype=tf.float32,
                                                 shape=[vocab_size]),
                                     name="dense_b")
        
        if is_training:
            self.__opt = tf.train.AdamOptimizer(lr, name="optimizer")
        
        # embedding
        with tf.device("/cpu:0"):
            self.__embed = tf.nn.embedding_lookup(self.__embed_matrix, self.__X,
                                                  name="embedding")
            
        # split the minibatch
        self.__embed_list = tf.split(self.__embed, num_or_size_splits=len(n_gpu),
                                     axis=0, name="embedding_list")
        self.__y_list = tf.split(self.__Y, num_or_size_splits=len(n_gpu),
                                 axis=0, name="y_list")
        self.__l_list = tf.split(self.__L, num_or_size_splits=len(n_gpu),
                                 axis=0, name="l_list")
        
        self.__loss_list = []
        self.__prob_list = []
        self.__grad_and_var_list = []
        
        for i in range(len(n_gpu)):
            with tf.device("/device:GPU:"+str(n_gpu[i])):
        
                tmp_embed = self.__embed_list[i]
                tmp_y = self.__y_list[i]
                tmp_l = self.__l_list[i]
                
                # input dropout
                if is_training and keep_prob < 1:
                    self.__rnn_in = tf.nn.dropout(tmp_embed, keep_prob,
                                                  name="input_dropout")
                else:
                    self.__rnn_in = tmp_embed
                
                self.__rnn_out, self.__rnn_state = tf.nn.dynamic_rnn(self.__cell_fw,
                                                                     self.__rnn_in,
                                                                     tmp_l,
                                                                     dtype=tf.float32)
                
                # softmax output
                self.__rnn_out_flatten = tf.reshape(self.__rnn_out, [-1, n_hidden])
                self.__logit_flatten = tf.matmul(self.__rnn_out_flatten,
                                                 self.__dense_W) + self.__dense_b
                self.__logit = tf.reshape(self.__logit_flatten,
                                          [-1, seq_max_len, vocab_size])
                self.__prob_list.append(tf.nn.softmax(self.__logit,
                                                      name="probability"))
        
                # cross entropy loss
                self.__seq_mask = tf.sequence_mask(tmp_l, seq_max_len,
                                                   dtype=tf.float32, name="sequence_mask")
                self.__loss_list.append(tf.contrib.seq2seq.sequence_loss(self.__logit,
                                                                         tmp_y,
                                                                         self.__seq_mask,
                                                                         average_across_timesteps=True,
                                                                         average_across_batch=True,
                                                                         name="loss"))
                
                # compute gradients, if training
                if is_training:
                    trainable = [x for x in tf.trainable_variables()
                                    if scope in x.name]
                    tmp_grads = self.__opt.compute_gradients(self.__loss_list[-1],
                                                             var_list=trainable)
                    self.__grad_and_var_list.append(tmp_grads)
        
        # merge the results from multiplt gpu's
        self.__loss = tf.reduce_mean(self.__loss_list, 0, name="final_loss")
        self.__prob = tf.concat(self.__prob_list, 0, name="final_prob")
        
        # training operation
        if is_training:
            self.__final_grads_and_vars = []
            for grads_and_vars in zip(*self.__grad_and_var_list):
                grads = []
                var = None
                for tmp_grad, tmp_var in grads_and_vars:
                    grads.append(tf.expand_dims(tmp_grad, 0))
                    var = tmp_var
                tmp_grad = tf.reduce_mean(tf.concat(grads, 0), 0,
                                          name="gradient")
                tmp_grad = tf.clip_by_value(tmp_grad, -grad_clip, grad_clip,
                                            name="gradient_clipping")
                self.__final_grads_and_vars.append((tmp_grad, var))
            self.__train_op = self.__opt.apply_gradients(self.__final_grads_and_vars,
                                                         name="train_op")

    def train_op(self, sess, X, Y, L):
        
        if self.__is_training:
            _, l = sess.run((self.__train_op, self.__loss), feed_dict={self.__X: X,
                                                                       self.__Y: Y,
                                                                       self.__L: L})
            return l
        else:
            return None
    
    def test_op(self, sess, X, Y, L):
        
        l = sess.run(self.__loss, feed_dict={self.__X: X,
                                             self.__Y: Y,
                                             self.__L: L})
        return l
    
    def prob_op(self, sess, X, L):
        
        p = sess.run(self.__prob, feed_dict={self.__X: X, self.__L: L})
        return p

    def __make_cell(self, is_training, n_hidden, keep_prob, layer=0):
        
        # make the basic LSTM cells
        tmp_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=0.0,
                                                reuse=not is_training)
        # dropout if needed
        if is_training and keep_prob < 1:
            cell = tf.contrib.rnn.DropoutWrapper(tmp_cell,
                                                 output_keep_prob=keep_prob)
        else:
            cell = tmp_cell
        return cell

if __name__ == "__main__":
    
    m = LanguageModel(is_training=True, keep_prob=1, n_gpu=2)
    
    cfg = tf.ConfigProto(allow_soft_placement=True)
    cfg.gpu_options.allow_growth = True
    sess = tf.Session(config=cfg)
    sess.run(tf.global_variables_initializer())
    
    arr = numpy.asarray(numpy.random.uniform(low=0,
                                             high=30000,
                                             size=(32, 20)),
                        dtype=numpy.int32)
    arr_l = numpy.asarray(numpy.random.uniform(low=1,
                                               high=20,
                                               size=(32)),
                          dtype=numpy.int32)
                
    res_tr = m.train_op(sess, arr, arr, arr_l)
    print(res_tr)
    res_te = m.test_op(sess, arr, arr, arr_l)
    print(res_te)
    res_pr = m.prob_op(sess, arr, arr_l)
    print(res_pr.shape)