#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import tensorflow as tf
from util import *

class SequenceClassificationModel(object):

	def __init__(self, max_seqlen, n_class, vocab_size=None,
              embed_size=None, embed_matrix=None, embed_trainable=True,
              opt_type="adam", lr=1e-3):
		
		'''
			PART I. basic information for model
		'''
		self.max_seqlen = max_seqlen
		self.n_class = n_class
		# vocab_size, embed_size and embed_matrix will be handled below.
		self.embed_trainable = embed_trainable
		self.opt_type = opt_type
		self.lr = lr

		'''
			PART II. embed 
		'''
		if embed_matrix is None:
			# if no pre-trained embedding, new embedding matrix should be trainable. That is, flag embed_trainable is useless.
			assert vocab_size and embed_size
			self.vocab_size = vocab_size + 2
			self.embed_size = embed_size
			with tf.device("/cpu:0"):
				self.embed_matrix = tf.get_variable(shape=[self.vocab_size, embed_size], dtype=tf.float32, name="embed_matrix", trainable=True) 
		else:
			# otherwise, <unk> and <pad> are trainable at least.
			self.vocab_size = embed_matrix.shape[0] + 2
			self.embed_size = embed_matrix.shape[1]
			with tf.device("/cpu:0"):
				self.embed_matrix = tf.concat(
					values=(tf.get_variable(
								initializer=tf.reshape(tf.convert_to_tensor(value=embed_matrix[0], dtype=tf.float32),
                                shape=[1, self.embed_size]), 
								name="embed_matrix_unk", trainable=True),
							tf.get_variable(
								initializer=tf.convert_to_tensor(value=embed_matrix[1:-1], dtype=tf.float32), 
								name="embed_matrix_listed_words", trainable=embed_trainable),
							tf.get_variable(
								initializer=tf.reshape(tf.convert_to_tensor(value=embed_matrix[-1], dtype=tf.float32),
                                shape=[1, self.embed_size]), 
								name="embed_matrix_pad", trainable=True),
                            tf.get_variable(
								shape=[1, self.embed_size], 
								name="embed_matrix_ph", trainable=True)),
					axis=0, name="embedding_matrix")

		'''
			PART III. build whole graph 
		'''		
		self.build_graph()

	# maybe need to implement in subClass.
	def build_graph(self):
		
		self.X = tf.placeholder(dtype=tf.int64, shape=[None, self.max_seqlen], name='X')
		self.L = tf.placeholder(dtype=tf.int64, shape=[None], name='L')
		self.Y = tf.placeholder(dtype=tf.int64, shape=[None], name='Y')
		
		self.logits, _ = self.__build_core_graph()

		self.loss = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(indices=self.Y, depth=self.n_class, axis=-1), logits=self.logits), 
			name="loss")
		self.opt = get_optimizer(self.opt_type, self.lr).minimize(self.loss)
		self.prob = tf.nn.softmax(self.logits, -1, name="Y_prob")
		self.Y_pred = tf.argmax(self.logits, 1, name="Y_pred")
		self.acc = tf.reduce_mean(
			tf.cast(tf.equal(self.Y_pred, self.Y), tf.float32), 
			name="acc")	

	# need to implement in subClass.
	@ abc.abstractmethod
	def __build_core_graph(self):
		pass

	# feed a batch of training data into sess.run(), train this model and return average loss & acc.
	def train(self, sess, batch_X, batch_L, batch_Y):
		_, loss, acc = sess.run(
			[self.opt, self.loss, self.acc], 
			feed_dict={self.X: batch_X, self.L: batch_L, self.Y: batch_Y})
		return loss, acc

	# feed a batch of evaluating data into sess.run(), return average loss & acc.
	def eval(self, sess, batch_X, batch_L, batch_Y):
		loss, acc = sess.run(
			[self.loss, self.acc], 
			feed_dict={self.X: batch_X, self.L: batch_L, self.Y: batch_Y})
		return loss, acc

	# feed a batch of infering data into sess.run(), return predicted labels.
	def infer(self, sess, batch_X, batch_L):
		Y_pred = sess.run(
			[self.Y_pred], 
			feed_dict={self.X: batch_X, self.L: batch_L})
		return Y_pred

class UniRNNSequenceClassifier(SequenceClassificationModel):

	def __init__(self, cell_type="lstm", hidden_size=128, average_hidden=False, **kwargs):
		
		self.cell_type = cell_type
		self.hidden_size = hidden_size
		self.average_hidden = average_hidden
		super().__init__(**kwargs)

	def build_graph(self):
		
		self.X = tf.placeholder(dtype=tf.int64, shape=[None, self.max_seqlen], name='X')
		self.L = tf.placeholder(dtype=tf.int64, shape=[None], name='L')
		self.Y = tf.placeholder(dtype=tf.int64, shape=[None], name='Y')
		
		self.logits, _, self.embed_x = self.__build_core_graph()

		self.loss = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(indices=self.Y, depth=self.n_class, axis=-1), logits=self.logits), 
			name="loss")
		self.opt = get_optimizer(self.opt_type, self.lr).minimize(self.loss)
		self.prob = tf.nn.softmax(self.logits, -1, name="Y_prob")
		self.Y_pred = tf.argmax(self.logits, 1, name="Y_pred")
		self.acc = tf.reduce_mean(
			tf.cast(tf.equal(self.Y_pred, self.Y), tf.float32), 
			name="acc")	

		self.embed_grad = tf.gradients(self.loss, self.embed_x)

	def __build_core_graph(self):

		embed_x = tf.nn.embedding_lookup(self.embed_matrix, self.X)

		# rnn
		with tf.variable_scope("rnn"):
			rnn_inputs = embed_x
			cell = get_rnn_cell(self.cell_type, self.hidden_size)
			rnn_outputs, final_state = tf.nn.dynamic_rnn(cell=cell,
														 inputs=rnn_inputs,
														 sequence_length=self.L,
														 dtype=tf.float32)
			if self.average_hidden:
				_logits = tf.cast(tf.reduce_sum(rnn_outputs, axis=1), dtype=tf.float32) / tf.cast(tf.reshape(self.L, shape=[-1, 1]), dtype=tf.float32)
			else:
				_logits = final_state.h

		# projection layer
		with tf.variable_scope("projection_layer"):
			W_output = tf.get_variable(name="W_output", dtype=tf.float32, shape=[self.hidden_size, self.n_class])
			b_output = tf.get_variable(name="b_output", dtype=tf.float32, shape=[self.n_class])
			logits = tf.matmul(_logits, W_output) + b_output

		return logits, (rnn_outputs, final_state), embed_x

class BiRNNSequenceClassifier(SequenceClassificationModel):

	def __init__(self, cell_type="lstm", hidden_size=64, average_hidden=False, concat_fw_bw=True, **kwargs):

		self.cell_type = cell_type
		self.hidden_size = hidden_size
		self.average_hidden = average_hidden
		self.concat_fw_bw = concat_fw_bw # if False, average forward and backward RNN outputs (or final state).
		super().__init__(**kwargs)

	def build_graph(self):
		
		self.X = tf.placeholder(dtype=tf.int64, shape=[None, self.max_seqlen], name='X')
		self.L = tf.placeholder(dtype=tf.int64, shape=[None], name='L')
		self.Y = tf.placeholder(dtype=tf.int64, shape=[None], name='Y')
		
		self.logits, self.embed_x, _ = self.__build_core_graph()

		self.loss = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(indices=self.Y, depth=self.n_class, axis=-1), logits=self.logits), 
			name="loss")
		self.opt = get_optimizer(self.opt_type, self.lr).minimize(self.loss)
		self.prob = tf.nn.softmax(self.logits, -1, name="Y_prob")
		self.Y_pred = tf.argmax(self.logits, 1, name="Y_pred")
		self.acc = tf.reduce_mean(
			tf.cast(tf.equal(self.Y_pred, self.Y), tf.float32), 
			name="acc")	
        
		self.embed_grad = tf.gradients(self.loss, self.embed_x)

	def __build_core_graph(self):

		embed_x = tf.nn.embedding_lookup(self.embed_matrix, self.X)
        
		# rnn
		with tf.variable_scope("rnn"):
			rnn_inputs = embed_x
			cell_fw = get_rnn_cell(self.cell_type, self.hidden_size)
			cell_bw = get_rnn_cell(self.cell_type, self.hidden_size)
			rnn_outputs, final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
																	   cell_bw=cell_bw,
																	   inputs=rnn_inputs,
																	   sequence_length=self.L,
																	   dtype=tf.float32)
			if self.average_hidden:
				_logits_fw = tf.cast(tf.reduce_sum(rnn_outputs[0], axis=1), dtype=tf.float32) / tf.cast(tf.reshape(self.L, shape=[-1, 1]), dtype=tf.float32)
				_logits_bw = tf.cast(tf.reduce_sum(rnn_outputs[1], axis=1), dtype=tf.float32) / tf.cast(tf.reshape(self.L, shape=[-1, 1]), dtype=tf.float32)
			else:
				_logits_fw = final_state[0].h
				_logits_bw = final_state[1].h
			# concatenate or average
			if self.concat_fw_bw:
				_logits = tf.concat([_logits_fw, _logits_bw], axis=1)
			else:
				_logits = (_logits_fw + _logits_bw) / 2

		# projection layer
		with tf.variable_scope("projection_layer"):
			if self.concat_fw_bw:
				W_output = tf.get_variable(name="W_output", dtype=tf.float32, shape=[self.hidden_size*2, self.n_class])
			else:
				W_output = tf.get_variable(name="W_output", dtype=tf.float32, shape=[self.hidden_size, self.n_class])
			b_output = tf.get_variable(name="b_output", dtype=tf.float32, shape=[self.n_class])
			logits = tf.matmul(_logits, W_output) + b_output

		return logits, embed_x, (rnn_outputs, final_state)

class HierarchicalAttentionClassifier(SequenceClassificationModel):

	def __init__(self, max_textlen, cell_type="lstm", hidden_size=32, atten_size=32, **kwargs):

		self.max_textlen = max_textlen # max number of sequences or sentences in "text".
		self.cell_type = cell_type
		self.hidden_size = hidden_size
		self.atten_size = atten_size
		super().__init__(**kwargs)

	# placeholder X and L are changed.
	def build_graph(self):

		self.X = tf.placeholder(dtype=tf.int64, shape=[None, self.max_textlen, self.max_seqlen], name='X')
		self.L = tf.placeholder(dtype=tf.int64, shape=[None, self.max_textlen], name='L')
		self.Y = tf.placeholder(dtype=tf.int64, shape=[None], name='Y')
		
		self.logits, self.alpha_words, self.alpha_sents, self.embed_x = self.__build_core_graph()

		self.loss = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(indices=self.Y, depth=self.n_class, axis=-1), logits=self.logits), 
			name="loss")
		self.opt = get_optimizer(self.opt_type, self.lr).minimize(self.loss)
		self.prob = tf.nn.softmax(self.logits, -1, name="Y_prob")
		self.Y_pred = tf.argmax(self.logits, 1, name="Y_pred")
		self.acc = tf.reduce_mean(
			tf.cast(tf.equal(self.Y_pred, self.Y), tf.float32), 
			name="acc")	
        
		self.embed_grad = tf.gradients(self.loss, self.embed_x)

	def __build_core_graph(self):

		embed_x = tf.nn.embedding_lookup(self.embed_matrix, self.X)
        
		# word level
		rnn_inputs = tf.reshape(embed_x, 
								shape=[-1, self.max_seqlen, self.embed_size])
		rnn_input_lens = tf.reshape(self.L, shape=[-1])
		with tf.variable_scope("word_level"):
			cell_fw = get_rnn_cell(self.cell_type, self.hidden_size)
			cell_bw = get_rnn_cell(self.cell_type, self.hidden_size)
			rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
															 cell_bw=cell_bw,
															 inputs=rnn_inputs,
															 sequence_length=rnn_input_lens,
															 dtype=tf.float32)
			attn_inputs = tf.concat(rnn_outputs, 2)
			attn_outputs, alpha_words = intra_attention(attn_inputs, rnn_input_lens, self.atten_size)

		# sent level
		rnn_inputs = tf.reshape(attn_outputs, [-1, self.max_textlen, 2*self.hidden_size])
		rnn_input_lens = tf.reduce_sum(tf.cast(self.L>0, tf.int64), axis=1)
		with tf.variable_scope("sent_level"):
			cell_fw = get_rnn_cell(self.cell_type, self.hidden_size)
			cell_bw = get_rnn_cell(self.cell_type, self.hidden_size)
			rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
															 cell_bw=cell_bw,
															 inputs=rnn_inputs,
															 sequence_length=rnn_input_lens,
															 dtype=tf.float32)
			attn_inputs = tf.concat(rnn_outputs, 2)
			attn_outputs, alpha_sents = intra_attention(attn_inputs, rnn_input_lens, self.atten_size)

		# projection layer
		with tf.variable_scope("projection_layer"):
			W_output = tf.get_variable(name="W_output", dtype=tf.float32, shape=[self.hidden_size*2, self.n_class])
			b_output = tf.get_variable(name="b_output", dtype=tf.float32, shape=[self.n_class])
			logits = tf.matmul(attn_outputs, W_output) + b_output

		return logits, alpha_words, alpha_sents, embed_x



