#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy
import pickle
import string
import os
import copy

import LM
import dataset as Dataset

from util import write_log
from util import reverse_seq
from util import target_seq
from util import original_seq_prob
from util import find_k_largest
from util import random_pick_idx_with_unnormalized_prob
from util import just_acc
from util import prob_len_modify
from util import cos_sim

class MetropolisHastings(object):
    
    def __init__(self, seq_max_len, embed_w, vocab_size, n_layer, n_hidden,
                 keep_prob, lr, n_gpu, grad_clip, init_idx=0, punkt_idx=[],
                 is_training=True, scope_name=""):
        
        self.__seq_max_len = seq_max_len
        self.__vocab_size = vocab_size
        self.__vocab_init = init_idx
        self.__vocab_pad = self.__vocab_size - 1
        self.__vocab_unk = self.__vocab_size - 2
        
        self.__vocab_punkt = punkt_idx
        
        with tf.variable_scope("forward", reuse=None):
            self.__forward_tr = LM.LanguageModel(seq_max_len=seq_max_len,
                                                 embed_w=embed_w,
                                                 vocab_size=vocab_size, 
                                                 n_layer=n_layer,
                                                 n_hidden=n_hidden,
                                                 keep_prob=keep_prob,
                                                 lr=lr,
                                                 n_gpu=n_gpu,
                                                 grad_clip=grad_clip,
                                                 scope=scope_name+"/forward",
                                                 is_training=is_training)
        
        if scope_name == "":
            self.__forward_var = [x for x in tf.trainable_variables()
                                if x.name.startswith('forward')]
        else:
            self.__forward_var = {}
            for v in tf.trainable_variables():
                if v.name.startswith(scope_name):
                    tmp_name = v.name
                    while scope_name in tmp_name:
                        tmp_name = tmp_name.strip(scope_name+"/")
                    if "forward" in tmp_name:
                        self.__forward_var[tmp_name.split(":")[0]] = v
        self.__forward_saver = tf.train.Saver(self.__forward_var, max_to_keep=1)
        
        with tf.variable_scope("backward", reuse=None):
            self.__backward_tr = LM.LanguageModel(seq_max_len=seq_max_len,
                                                 embed_w=embed_w,
                                                 vocab_size=vocab_size, 
                                                 n_layer=n_layer,
                                                 n_hidden=n_hidden,
                                                 keep_prob=keep_prob,
                                                 lr=lr,
                                                 n_gpu=n_gpu,
                                                 grad_clip=grad_clip,
                                                 scope=scope_name+"/backward",
                                                 is_training=is_training)
        if scope_name == "":
            self.__backward_var = [x for x in tf.trainable_variables()
                                    if x.name.startswith('backward')]
        else:
            self.__backward_var = {}
            for v in tf.trainable_variables():
                if v.name.startswith(scope_name):
                    tmp_name = v.name
                    while scope_name in tmp_name:
                        tmp_name = tmp_name.strip(scope_name+"/")
                    if "backward" in tmp_name:
                        self.__backward_var[tmp_name.split(":")[0]] = v
        self.__backward_saver = tf.train.Saver(self.__backward_var, max_to_keep=1)
    
    def train(self, sess, max_epoch, batch_size, dataset,
              forward_model_save_path, backward_model_save_path,
              forward_log_save_path, backward_log_save_path):
        
        n_train_batch = dataset.get_train_size() // batch_size
        n_test_batch = dataset.get_test_size() // batch_size
        
        print ("TRAINING MODEL, BEGIN!")
        write_log("===== FORWARD MODEL =====\n", forward_log_save_path)
        write_log("===== BACKWARD MODEL =====\n", backward_log_save_path)
        test_ppl_mean_best_fw = 20.0
        test_ppl_mean_best_bw = 20.0
        
        for epoch in range(max_epoch):
            # Forward model
            train_ppl_list=[]
            test_ppl_list=[]
            dataset.reset_train_epoch()
            dataset.reset_test_epoch()
            for i in range(n_train_batch):
                X, Y, L, _ = dataset.minibatch(batch_size)
                train_ppl = self.__forward_tr.train_op(sess, X, Y, L)
                train_ppl_list.append(train_ppl)
                print("FEpoch = %d\t iter = %d/%d\tTrain PPL = %.3f"
                      % (epoch+1, i+1, n_train_batch, train_ppl))
            for i in range(n_test_batch):
                X, Y, L, _ = dataset.test_batch(batch_size)
                test_ppl = self.__forward_tr.test_op(sess, X, Y, L)
                test_ppl_list.append(test_ppl)
                print("FEpoch = %d\t iter = %d/%d\tTest PPL = %.3f"
                      % (epoch+1, i+1, n_test_batch, test_ppl))
            test_ppl_mean = numpy.mean(test_ppl_list)
            train_ppl_mean = numpy.mean(train_ppl_list)
            if test_ppl_mean < test_ppl_mean_best_fw:
                test_ppl_mean_best_fw = test_ppl_mean
                self.__forward_saver.save(sess, forward_model_save_path)
            write_log('Epoch '+str(epoch+1)+'\ttrain ppl = '+str(train_ppl_mean)
                        +'\t'+'test ppl:'+str(test_ppl_mean)+"\n",
                      forward_log_save_path)
            # Backward model
            train_ppl_list=[]
            test_ppl_list=[]
            dataset.reset_train_epoch()
            dataset.reset_test_epoch()
            for i in range(n_train_batch):
                X, Y, L, _ = dataset.minibatch_rev(batch_size)
                train_ppl = self.__backward_tr.train_op(sess, X, Y, L)
                train_ppl_list.append(train_ppl)
                print("BEpoch = %d\t iter = %d/%d\tTrain PPL = %.3f"
                      % (epoch+1, i+1, n_train_batch, train_ppl))
            for i in range(n_test_batch):
                X, Y, L, _ = dataset.test_batch_rev(batch_size)
                test_ppl = self.__backward_tr.test_op(sess, X, Y, L)
                test_ppl_list.append(test_ppl)
                print("BEpoch = %d\t iter = %d/%d\tTest PPL = %.3f"
                      % (epoch+1, i+1, n_test_batch, test_ppl))
            test_ppl_mean = numpy.mean(test_ppl_list)
            train_ppl_mean = numpy.mean(train_ppl_list)
            if test_ppl_mean < test_ppl_mean_best_bw:
                test_ppl_mean_best_bw = test_ppl_mean
                self.__backward_saver.save(sess, backward_model_save_path)
            write_log('Epoch '+str(epoch+1)+'\ttrain ppl = '+str(train_ppl_mean)
                        +'\t'+'test ppl:'+str(test_ppl_mean)+"\n",
                      backward_log_save_path)
            dataset.reset_test_epoch()
        print ("TRAINING MODEL, END!")
    
    def load(self, sess, forward_model_save_path, backward_model_save_path):
        
        self.__forward_saver.restore(sess, forward_model_save_path)
        self.__backward_saver.restore(sess, backward_model_save_path)

    def __compute_seq_probability(self, sess, X, L):
        
        # X is a sequence batch, and L are the lengths of each sequence
        x_fw, y_fw, l = target_seq(X, L, self.__vocab_init, self.__vocab_pad)
        prob_fw = self.__forward_tr.prob_op(sess, x_fw, l)
        prob = original_seq_prob(prob_fw, y_fw, l)
        x_bw, y_bw = reverse_seq(x_fw, y_fw, l, self.__vocab_init, self.__vocab_pad)
        prob_bw = self.__backward_tr.prob_op(sess, x_bw, l)
        prob += original_seq_prob(prob_bw, y_bw, l)
        #print (prob)
        return prob, prob_fw, prob_bw

    def op_replace(self, sess, X, L, cl_x, cl_l, cl_y, idx, n_candidate=100, prob=[1/3, 1/3, 1/3],
                   lm_vocab=None, cl_vocab=None, cl=None):
        
        assert ((lm_vocab is None) and (cl_vocab is None) and (cl is None)) \
            or ((lm_vocab is not None) and (cl_vocab is not None) and (cl is not None))
        
        # Compute probability for the original sequence
        log_prob_old, prob_old_fw, prob_old_bw = self.__compute_seq_probability(sess,
                                                                                [X],
                                                                                [L])
        
        # Generate candidate sequences
        prob_candidate = prob_old_fw[0, idx-1, :] * prob_old_bw[0, L-1-idx, :]
        prob_candidate, idx_candidate = find_k_largest(prob_candidate, n_candidate)
        candidate = []
        candidate_l = []
        if cl is not None:
            cl_candidate = []
            cl_candidate_l = []
        if not X[idx] in self.__vocab_punkt:
            for i in range(n_candidate):
                if idx_candidate[i] in ([X[idx], self.__vocab_init, self.__vocab_pad,
                                        self.__vocab_unk] + self.__vocab_punkt):
                    continue
                candidate.append(copy.deepcopy(numpy.asarray(X)))
                candidate[-1][idx] = idx_candidate[i]
                candidate_l.append(L)
                if cl is not None:
                    cl_candidate.append(numpy.asarray(cl_x))
                    tmp_str= lm_vocab.get_vocab(copy.deepcopy(idx_candidate[i]))
                    if tmp_str in cl_vocab.keys():
                        cl_candidate[-1][idx-1] = cl_vocab[tmp_str]
                    else:
                        cl_candidate[-1][idx-1] = cl_vocab["<unk>"]
                    cl_candidate_l.append(cl_l)
        else:
            for i in range(n_candidate):
                if idx_candidate[i] in [X[idx], self.__vocab_init, self.__vocab_pad,
                                        self.__vocab_unk]:
                    continue
                candidate.append(copy.deepcopy(numpy.asarray(X)))
                candidate[-1][idx] = idx_candidate[i]
                candidate_l.append(L)
                if cl is not None:
                    cl_candidate.append(copy.deepcopy(numpy.asarray(cl_x)))
                    tmp_str= lm_vocab.get_vocab(idx_candidate[i])
                    if tmp_str in cl_vocab.keys():
                        cl_candidate[-1][idx-1] = cl_vocab[tmp_str]
                    else:
                        cl_candidate[-1][idx-1] = cl_vocab["<unk>"]
                    cl_candidate_l.append(cl_l)
        
        # Compute probability for each candidate sequence
        log_prob_new, _, _ = self.__compute_seq_probability(sess,
                                                            candidate,
                                                            candidate_l)
        
        prob_new = numpy.float_power(numpy.e, log_prob_new-numpy.min(log_prob_new))
        prob_old = numpy.float_power(numpy.e, log_prob_old-numpy.min(log_prob_new))
        
        # Compute alpha
        if cl is None:
            # Sample from candidate, and decide whether to accept it
            new_idx = random_pick_idx_with_unnormalized_prob(prob_new)
            alpha = 1
        else:
            embedding = sess.run(cl.embed_matrix)
            delta = []
            for i in cl_candidate:
                delta.append(embedding[i[idx-1]]-embedding[cl_x[idx-1]])
            delta = numpy.asarray(delta)
            grad = sess.run(cl.embed_grad, feed_dict={cl.X: [cl_x],
                                                      cl.Y: [cl_y],
                                                      cl.L: [cl_l]})[0]
            grad = numpy.asarray([grad[0, idx-1, :] for i in range(len(delta))])
            grad = numpy.random.uniform(-1e-100, 1e-100, grad.shape) + grad
            delta = numpy.random.uniform(-1e-100, 1e-100, delta.shape) + delta
            sim = cos_sim(grad, delta)
            old2new = numpy.asarray(prob_new * sim)
            normalized_old2new = old2new / numpy.sum(old2new)
            # Sample from candidate, and decide whether to accept it
            new_idx = random_pick_idx_with_unnormalized_prob(old2new)
            grad_rev = sess.run(cl.embed_grad, feed_dict={cl.X: cl_candidate,
                                                          cl.Y: [cl_y for i in range(len(cl_candidate))],
                                                          cl.L: cl_candidate_l})[0][:, idx-1, :]
            delta_rev = []
            for i in range(len(cl_candidate)):
                delta_rev.append(embedding[cl_candidate[i][idx-1]]-embedding[cl_candidate[new_idx][idx-1]])
            delta_rev[new_idx] = embedding[cl_x[idx-1]] - embedding[cl_candidate[new_idx][idx-1]]
            delta_rev = numpy.asarray(delta_rev)
            delta_rev = numpy.random.uniform(-1e-100, 1e-100, delta_rev.shape) + delta_rev
            grad_rev = numpy.random.uniform(-1e-100, 1e-100, grad_rev.shape) + grad_rev
            sim_rev = cos_sim(grad_rev, delta_rev)
            new2old = numpy.asarray(prob_new * sim_rev)
            new2old[new_idx] = prob_old[0] * sim_rev[new_idx]
            normalized_new2old = new2old / numpy.sum(new2old)
            
            old2new_ = normalized_old2new[new_idx]
            new2old_ = normalized_new2old[new_idx]
            alpha = (prob_new[new_idx] * new2old_) / (prob_old[0] * old2new_)
        
        prob_old = prob_old[0]
        prob_new = prob_new[new_idx]
		
        return {"proposal": candidate[new_idx],
                "candidate": (idx_candidate, prob_candidate),
                "alpha": alpha,
                "old_prob": prob_old,
                "new_prob": prob_new}
        
    def op_insert(self, sess, X, L, cl_x, cl_l, cl_y, idx, n_candidate=100, prob=[1/3, 1/3, 1/3],
                   lm_vocab=None, cl_vocab=None, cl=None):
        
        assert ((lm_vocab is None) and (cl_vocab is None) and (cl is None)) \
            or ((lm_vocab is not None) and (cl_vocab is not None) and (cl is not None))
        
        # Compute probability for the original sequence
        log_prob_old, prob_old_fw, prob_old_bw = self.__compute_seq_probability(sess,
                                                                                [X],
                                                                                [L])
        
        prob_candidate = prob_old_fw[0, idx, :] * prob_old_bw[0, L-1-idx, :]
        prob_candidate, idx_candidate = find_k_largest(prob_candidate, n_candidate)

        # Generate candidate sequences
        candidate = []
        candidate_l = []
        X = numpy.asarray(X).tolist()
        X = X[:idx+1] + [self.__vocab_unk] + X[idx+1:]
        X = X[:-1]
        L = L + 1
        if L >= len(X):
            L = len(X)-1
        idx = idx + 1
        for i in range(n_candidate):
            if idx_candidate[i] in [self.__vocab_init, self.__vocab_pad, self.__vocab_unk]:
                continue
            candidate.append(numpy.asarray(X))
            candidate[-1][idx] = idx_candidate[i]
            candidate_l.append(L)
        
        # Compute probability for each candidate sequence
        log_prob_new, prob_new_fw, prob_new_bw = self.__compute_seq_probability(sess,
                                                                                candidate,
                                                                                candidate_l)
        
        prob_new = numpy.float_power(numpy.e, log_prob_new-numpy.min(log_prob_new))
        prob_old = numpy.float_power(numpy.e, log_prob_old-numpy.min(log_prob_new))
        prob_old_modified = prob_len_modify(prob_old, L)
        prob_new_modified = prob_len_modify(prob_new, L)
        normalized_prob_new = prob_new_modified / numpy.sum(prob_new_modified)
        
        # Sample from candidate, and decide whether to accept it
        new_idx = random_pick_idx_with_unnormalized_prob(prob_new)
        prob_new = prob_new_modified[new_idx]
        prob_old = prob_old_modified[0]
        prob_old2new_insert = normalized_prob_new[new_idx]
        
        return {"proposal": candidate[new_idx],
                "alpha": (prob_new * prob[2]) / (prob_old * prob_old2new_insert *prob[1]),
                "old_prob": prob_old,
                "new_prob": prob_new}
        
    def op_delete(self, sess, X, L, cl_x, cl_l, cl_y, idx, n_candidate=100, prob=[1/3, 1/3, 1/3],
                   lm_vocab=None, cl_vocab=None, cl=None):
        
        assert ((lm_vocab is None) and (cl_vocab is None) and (cl is None)) \
            or ((lm_vocab is not None) and (cl_vocab is not None) and (cl is not None))
        
        # Compute probability for the original sequence
        log_prob_old, prob_old_fw, prob_old_bw = self.__compute_seq_probability(sess,
                                                                                [X],
                                                                                [L])
        
        # Generate candidate sequences
        prob_candidate = prob_old_fw[0, idx-1, :] * prob_old_bw[0, L-1-idx, :]
        prob_candidate, idx_candidate = find_k_largest(prob_candidate, n_candidate)
        candidate = []
        candidate_l = []
        if X[idx] not in idx_candidate:
            tmp_prob = 0
        else:
            for i in range(n_candidate):
                if idx_candidate[i] in [self.__vocab_init, self.__vocab_pad, self.__vocab_unk]:
                    continue
                candidate.append(numpy.asarray(X))
                candidate[-1][idx] = idx_candidate[i]
                candidate_l.append(L)
            
            # Compute probability for each candidate sequence
            log_tmp_prob, tmp_prob_fw, tmp_prob_bw = self.__compute_seq_probability(sess,
                                                                                    candidate,
                                                                                    candidate_l)
            tmp_prob = numpy.float_power(numpy.e, log_tmp_prob-numpy.min(log_tmp_prob))
            tmp_prob_modified = prob_len_modify(tmp_prob, L)
            normalized_tmp_prob = tmp_prob_modified / numpy.sum(tmp_prob_modified)
            tmp_prob = 0
            ii = 0
            for i in range(n_candidate):
                if idx_candidate[i] in [self.__vocab_init, self.__vocab_pad, self.__vocab_unk]:
                    continue
                if idx_candidate[i] == X[idx]:
                    tmp_prob = normalized_tmp_prob[ii]
                ii += 1
        
        # Compute probability for the sequence after deletion
        X = numpy.asarray(X).tolist()
        X = X[:idx] + X[idx+1:] + [self.__vocab_pad]
        L -= 1
        log_prob_new, _, _ = self.__compute_seq_probability(sess,
                                                            [X],
                                                            [L])
        
        prob_new = numpy.float_power(numpy.e, log_prob_new-numpy.min(log_prob_new))
        prob_old = numpy.float_power(numpy.e, log_prob_old-numpy.min(log_prob_new))
        prob_old_modified = prob_len_modify(prob_old, L)
        prob_new_modified = prob_len_modify(prob_new, L)

        prob_new = prob_new_modified[0]
        prob_old = prob_old_modified[0]

        return {"proposal": X,
                "alpha": (prob_new * tmp_prob * prob[1]) / (prob_old * prob[2]),
                "old_prob": prob_old,
                "new_prob": prob_new}
        
def report_op(seq, l, vocab, op, acc=True):
    
    if op == 0:
        print ("REPLACE", end="\t")
    elif op == 1:
        print ("INSERT", end="\t")
    elif op == 2:
        print ("DELETE", end="\t")
    else:
        print ("PASS", end="\t")
    if (acc):
        print ("ACC", end="\t")
    else:
        print ("REJ", end="\t\t")
    for t in seq[1:l]:
        print (vocab.get_vocab(t), end=" ")
    print ()
    

if __name__ == "__main__":
    
    max_seq_len = 200
    vocab_size = 50000
    
    n_candidate = 100
    n_acc_sample = 100
    just_acc_rate = 0.01
    prob_threshold = 0.5
    op_prob = [1/3, 1/3, 1/3, 0.]
    
    threshold = 0.9

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    n_gpu = 1

    vocab = Dataset.Dictionary("./one_billion_word/vocab_1M_processed.pkl", vocab_size)
    m = MetropolisHastings(max_seq_len, 300, vocab.get_dict_size(), 2, 300,
                           1.0, 1e-5, n_gpu, 1, vocab.get_init_idx(), [], False)
    
    print ("MODEL BUILT")
    
    init_op = tf.global_variables_initializer()
    configs = tf.ConfigProto(allow_soft_placement=True)
    configs.gpu_options.allow_growth = True
    sess = tf.Session(config=configs)
    sess.run(init_op)
    
    print ("MODEL INITIALIZED")
    
    m.load(sess, "./model/forward_1M_processed/model.ckpt",
           "./model/backward_1M_processed/model.ckpt")
    
    print ("MODEL LOADED")

    key_words = ["it", "was", "the", "best", "of", "times", ".",
                 "it", "was", "the", "worst", "of", "times", "."]
    mask = [True, False, False, False, False, False, False, False,
                  False, False, False, False, False, False, False]
    
    assert len(mask) - 1 == len(key_words)
    
    seq = [vocab.get_init_idx()]
    for i in key_words:
        seq.append(vocab.get_vocab_idx(i.lower()))
    l = len(seq)
    while len(seq) < max_seq_len:
        seq.append(vocab.get_pad_idx())

    print ("\noriginal sentences:", end="\t")
    for t in key_words:
        print (t, end=" ")
    print ("\nkey words:", end="\t")
    for i in range(len(mask[1:])):
        if mask[i+1]:
            print (key_words[i], end=" ")
    print ("\n")
    
    n_acc = 0
    idx = 0
    while n_acc < n_acc_sample:
        print (n_acc, end="\t")
        idx = (idx + 1) % l
        op = random_pick_idx_with_unnormalized_prob(op_prob)
        if op == 3:
            n_acc += 1
            report_op(seq, l, vocab, op, True)
            continue
        elif op == 2:
            if mask[idx]:
                report_op([], 0, vocab, op, False)
                continue
            if l - 1 <= 0:
                report_op([], 0, vocab, op, False)
                continue
            proposal = m.op_delete(sess, seq, l, idx, n_candidate, op_prob, n_gpu)
            acc = False
            old_l = l
            if (just_acc(just_acc_rate)
                or numpy.random.uniform(0,1) <= proposal["alpha"]
                or proposal["old_prob"] * threshold <= proposal["new_prob"]):
                n_acc += 1
                acc = True
                mask = mask[:idx] + mask[idx+1:]
                l -= 1
                seq = list(proposal["proposal"])
                #print ([vocab.get_vocab(seq[i]) for i in range(len(mask)) if mask[i]])
            report_op(proposal["proposal"], old_l-1, vocab, op, acc)
        elif op == 1:
            if l + 1 >= max_seq_len:
                report_op([], 0, vocab, op, False)
                continue
            proposal = m.op_insert(sess, seq, l, idx, n_candidate, op_prob, n_gpu)
            acc = False
            old_l = l
            if (just_acc(just_acc_rate)
                or numpy.random.uniform(0,1) <= proposal["alpha"]
                or proposal["old_prob"] * threshold <= proposal["new_prob"]):
                n_acc += 1
                acc = True
                mask = mask[:idx+1] + [False] + mask[idx+1:]
                l += 1
                seq = list(proposal["proposal"])
                #print ([vocab.get_vocab(seq[i]) for i in range(len(mask)) if mask[i]])
            report_op(proposal["proposal"], old_l+1, vocab, op, acc)
        elif op == 0:
            if mask[idx]:
                report_op([], 0, vocab, op, False)
                continue
            proposal = m.op_replace(sess, seq, l, idx, n_candidate, op_prob, n_gpu)
            acc = False
            if (just_acc(just_acc_rate)
                or numpy.random.uniform(0,1) <= proposal["alpha"]
                or proposal["old_prob"] * threshold <= proposal["new_prob"]):
                n_acc += 1
                acc = True
                seq = list(proposal["proposal"])
                #print ([vocab.get_vocab(seq[i]) for i in range(len(mask)) if mask[i]])
            tmp_candi = sorted([(proposal["candidate"][0][i], proposal["candidate"][1][i]) \
                                for i in range(len(proposal["candidate"][0]))],
                               key=lambda item: item[1])
            for ii in range(len(proposal["candidate"][0])):
                print("%s(%.2e)" % (vocab.get_vocab(tmp_candi[ii][0]),
                      tmp_candi[ii][1]), end=" ")
            print("\n", end="\t")
            report_op(proposal["proposal"], l, vocab, op, acc)
        
        assert len(mask) == l
        
        #print ("\t%d %d\t%.2e\t%.2e\t%.2f" % (op, idx, proposal['old_prob'],
        #                                      proposal['new_prob'], proposal["alpha"]))
                        
    print ("\noriginal sentences:", end="\t")
    for t in key_words:
        print (t, end=" ")
    print ("\nkey words:", end="\t")
    for i in range(len(mask[1:])):
        if mask[i]:
            print (vocab.get_vocab(seq[i]), end=" ")