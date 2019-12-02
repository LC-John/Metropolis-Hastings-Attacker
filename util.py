#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import numpy
import scipy.spatial
from nltk.corpus import wordnet as wn

def get_part_of_speech(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return wn.NOUN

def write_log(s, path):
    
    with open(path, 'a') as log:
        log.write(s+'\n')
        
def cos_sim(grad, vec):
    
    grad = numpy.asarray(grad)
    vec = numpy.asarray(vec)
    assert grad.shape == vec.shape, str(grad.shape)+" "+str(vec.shape)
    cos = []
    for i in range(len(grad)):
        cos.append(1 - scipy.spatial.distance.cosine(grad[i], vec[i]) / 2)
    return numpy.asarray(cos)
    
def prob_len_modify(probs, l):
    
    for i in range(len(probs)):
        probs[i] = numpy.power(probs[i], 1/l)
    return probs

def reverse_seq(x, y, l, init, pad):
    
    batch_size = x.shape[0]
    max_len = x.shape[1]
    x_new = []
    y_new = []
    for s_idx in range(batch_size):
        tmp_x = [init, y[s_idx][l[s_idx]-1]]
        for i in range(l[s_idx]-1, 1, -1):
            tmp_x.append(x[s_idx][i])
        while len(tmp_x) < max_len:
            tmp_x.append(pad)
        tmp_y = [y[s_idx][i] for i in range(l[s_idx]-1, -1, -1)]
        while len(tmp_y) < max_len:
            tmp_y.append(pad)
        x_new.append(tmp_x)
        y_new.append(tmp_y)
    return numpy.asarray(x_new), numpy.asarray(y_new)

def target_seq(arg_x, arg_l, init, pad):
    
    x = numpy.asarray(arg_x)
    l = numpy.asarray(arg_l)
    batch_size = x.shape[0]
    max_len = x.shape[1]
    y_new = []
    x_new = []
    l_new = []
    for s in range(batch_size):
        tmp_y = []
        for i in range(1, l[s], 1):
            tmp_y.append(x[s][i])
        while len(tmp_y) < max_len:
            tmp_y.append(pad)
        y_new.append(tmp_y)
        x_new.append(x[s])
        x_new[-1][l[s]-1] = pad
        l_new.append(l[s]-1)
    return (numpy.asarray(x_new, dtype=numpy.int32),
            numpy.asarray(y_new, dtype=numpy.int32),
            numpy.asarray(l_new, dtype=numpy.int32))

def original_seq_prob(prob, y, l):
    
    ret = [0 for i in l]
    prob_ = numpy.log(prob)
    for s in range(len(l)):
        for t in range(l[s]):
            ret[s] += prob_[s][t][y[s][t]]
    return numpy.asarray(ret)

def find_k_largest(x, k):
    
    idx = numpy.argpartition(x, -k)[-k:]
    ret = numpy.asarray([x[i] for i in idx])
    return ret, idx

def random_pick_idx_with_unnormalized_prob(prob):
    
    p = numpy.random.uniform(low=0.0, high=numpy.sum(prob))
    ret = 0
    while p >= 0 and ret < len(prob):
        p -= prob[ret]
        if p < 0:
            return ret
        ret += 1
    return len(prob)-1

def just_acc(just_acc_rate):
    
    if numpy.random.uniform() <= just_acc_rate:
        return True
    return False

# Get an fixed-learning-rate optimizer instance of, for example, tf.train.AdamOptimizer.
def get_optimizer(opt_type, lr):
	opt_type = opt_type.lower()
	if opt_type in ['sgd', 'gd', 'gradientdescent']:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
	elif opt_type=='adagrad':
		optimizer = tf.train.AdagradOptimizer(learning_rate=lr)
	elif opt_type=='adadelta':
		optimizer = tf.train.AdadeltaOptimizer(learning_rate=lr)
	elif opt_type=='adam':
		optimizer = tf.train.AdamOptimizer(learning_rate=lr)
	else:
		assert False, '<Unknown Optimizer> %s'%opt_type
	return optimizer

# Get a certain-type RNNCell instance of, for example, tf.contrib.rnn.BasicLSTMCell.
def get_rnn_cell(cell_type, hidden_size):
	cell_type = cell_type.lower()
	if cell_type in ["rnn", "basicrnn"]:	
		cell = tf.contrib.rnn.BasicRNNCell(hidden_size)
	elif cell_type in ["lstm", "basiclstm"]:
		cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
	elif cell_type == "gru":
		cell = tf.contrib.rnn.GRUCell(hidden_size)
	else:
		assert False, "<Unknown RNN Cell Type>: %s"%cell_type
	return cell

# Input tensor shaped [batch_size, max_time, input_width], return (atten_outs, alphas)
#   atten_ous: an attention tensor shaped [batch_size, input_width] 
#   alphas: an attention weights tensor shaped [batch_size, max_time]
def intra_attention(atten_inputs, input_lens, atten_size):
	## attention mechanism uses Ilya Ivanov's implementation(https://github.com/ilivans/tf-rnn-attention)
	max_time = int(atten_inputs.shape[1])
	input_width = int(atten_inputs.shape[2])
	W_omega = tf.Variable(tf.random_normal([input_width, atten_size], stddev=0.1, dtype=tf.float32), name="W_omega")
	b_omega = tf.Variable(tf.random_normal([atten_size], stddev=0.1, dtype=tf.float32), name="b_omega")
	u_omega = tf.Variable(tf.random_normal([atten_size], stddev=0.1, dtype=tf.float32), name="u_omega")
	v = tf.tanh(\
			tf.matmul(tf.reshape(atten_inputs, [-1, input_width]), W_omega) + \
			tf.reshape(b_omega, [1, -1]))
	# u_omega is the summarizing question vector
	vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
	mask = tf.sequence_mask(input_lens, maxlen=max_time, dtype=tf.float32)
	exps = tf.reshape(tf.exp(vu), [-1, max_time]) * mask + 1e-10
	alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])
	atten_outs = tf.reduce_sum(atten_inputs * tf.reshape(alphas, [-1, max_time, 1]), 1)
	return atten_outs, alphas

# Print Trainable Variables
# MUST used after sess.run(tf.global_variables_initializer()) or sever.file(sess, ckpt)
def print_variables():
	print("[*] Model Trainable Variables:")
	parm_cnt = 0
	variable = [v for v in tf.trainable_variables()]
	#variable = [v for v in tf.global_variables()]
	for v in variable:
		print("   ", v.name, v.get_shape())
		parm_cnt_v = 1
		for i in v.get_shape().as_list():
			parm_cnt_v *= i
		parm_cnt += parm_cnt_v
	print("[*] Model Param Size: %.4fM" %(parm_cnt/1024/1024))

class BucketedDataIterator():
    ## bucketed data iterator uses R2RT's implementation(https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html)
    
    def __init__(self, df, num_buckets=3):
        df = df.sort_values('text_length').reset_index(drop=True)
        # NOTE: sort, http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.reset_index.html
        self.size = int(len(df) / num_buckets)
        self.dfs = []
        for bucket in range(num_buckets):
            self.dfs.append(df.iloc[bucket*self.size: (bucket+1)*self.size])
            # NOTE: slice, http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.iloc.html
            # l = list(range(20)); l[19:22]->[19]
        self.num_buckets = num_buckets

        # cursor[i] will be the cursor for the ith bucket
        self.cursor = np.array([0] * num_buckets)
        self.shuffle()

        self.epochs = 0

    def shuffle(self):
        #sorts dataframe by sequence length, but keeps it random within the same length
        for i in range(self.num_buckets):
            self.dfs[i] = self.dfs[i].sample(frac=1).reset_index(drop=True)
            # Note: Return a random sample of items from an axis of object, frac means the ratio: |sample| / items
            self.cursor[i] = 0

    def next_batch(self, n):
        if np.any(self.cursor+n > self.size):
            self.epochs += 1
            self.shuffle()
        i = np.random.randint(0, self.num_buckets)
        res = self.dfs[i].iloc[self.cursor[i]:self.cursor[i]+n]
        self.cursor[i] += n
        if 'sents_length' in res:
            	return np.asarray(res['text'].tolist()), res['label'].tolist(), res['sents_length'].tolist()
        else:
            	return np.asarray(res['text'].tolist()), res['label'].tolist(), res['text_length'].tolist(), res["text_raw"].tolist()
