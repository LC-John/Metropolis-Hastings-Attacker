#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

tf.app.flags.DEFINE_integer("n_chunk", 10,
                            "number of data chunks")
tf.app.flags.DEFINE_integer("chunk_idx", 0,
                            "index of this chunk")

tf.app.flags.DEFINE_integer("models_per_gpu", 2,
                            "put how many models on one gpu")

tf.app.flags.DEFINE_integer("atk_seq_max_len", 100,
                            "max length of the sequences to be adversarial attacked")

tf.app.flags.DEFINE_float("swp_prob", 1/2,
                          "probability of SWAP operation")
tf.app.flags.DEFINE_float("ins_prob", 1/4,
                          "probability of INSERT operation")
tf.app.flags.DEFINE_float("del_prob", 1/4,
                          "probability of DELETE operation")
tf.app.flags.DEFINE_float("pass_prob", 0,
                          "probability of PASS operation")
tf.app.flags.DEFINE_integer("sample_max_n", 800,
                            "max number of trial iteration of CGMH")
tf.app.flags.DEFINE_integer("n_candidate", 40,
                            "candidate number of CGMH")
tf.app.flags.DEFINE_float("lm_swp_threshold", 0.5,
                          "ratio threshold of the language model score")
tf.app.flags.DEFINE_float("lm_ins_threshold", 0.999,
                          "ratio threshold of the language model score")
tf.app.flags.DEFINE_float("lm_del_threshold", 0.999,
                          "ratio threshold of the language model score")
tf.app.flags.DEFINE_float("swp_threshold", 0.95,
                          "ratio threshold of the classifier output probability for SWAP")
tf.app.flags.DEFINE_float("ins_threshold", 0.999,
                          "ratio threshold of the classifier output probability for INSERT")
tf.app.flags.DEFINE_float("del_threshold", 0.999,
                          "ratio threshold of the classifier output probability for DELETE")
tf.app.flags.DEFINE_float("senti_obj_threshold", 0.8,
                          "threshold of the objective score")
tf.app.flags.DEFINE_float("senti_pos_threshold", 0.1,
                          "threshold of the pos-neg score")
tf.app.flags.DEFINE_integer("seq_min_len", 10,
                            "min length of the sequence")
tf.app.flags.DEFINE_float("just_acc_rate", 0.001,
                          "accept no matter what")
tf.app.flags.DEFINE_string("index_mode", "grad",
                           "indexing mode while performing CGMH, \"traverse\" & \
                               \"random\" & \"grad\" available")

tf.app.flags.DEFINE_integer("seq_max_len", 200,
                            "max length of the sequences")
tf.app.flags.DEFINE_integer("embed_width", 300,
                            "dimension of the embedding space")
tf.app.flags.DEFINE_integer("vocab_size", 50000,
                            "vocabulary size")
tf.app.flags.DEFINE_integer("n_rnn_layer", 2,
                            "number of layers of the multi-layer rnn")
tf.app.flags.DEFINE_integer("n_rnn_cell", 300,
                            "dimension of each layer of the multi-layer rnn")

tf.app.flags.DEFINE_string("atk_data_path", "/mnt/cephfs/lab/zhanghuangzhao/cgmh_data/atk_data.pkl.gz",
                           "data samples to be attacked")
tf.app.flags.DEFINE_string("imdb_root_path", "/mnt/cephfs/lab/zhanghuangzhao/cgmh_data/imdb",
                           "root path of imdb data & model")
tf.app.flags.DEFINE_string("model_dir", "bilstm",
                           "model directory")
tf.app.flags.DEFINE_string("cgmh_output_dir", "/mnt/cephfs/lab/zhanghuangzhao/cgmh_adversarial_log",
                           "directory of the cgmh log & result")
tf.app.flags.DEFINE_string("forward_model", "/mnt/cephfs/lab/zhanghuangzhao/cgmh_model/lm_forward/model.ckpt",
                           "path of the forward model")
tf.app.flags.DEFINE_string("backward_model", "/mnt/cephfs/lab/zhanghuangzhao/cgmh_model/lm_backward/model.ckpt",
                           "path of the backward model")
tf.app.flags.DEFINE_string("lm_input_data", "one_billion_word/input_1M_processed.pkl",
                           "input of the dataset for language model")
tf.app.flags.DEFINE_string("lm_output_data", "one_billion_word/output_1M_processed.pkl",
                           "output of the dataset for language model")
tf.app.flags.DEFINE_string("lm_vocab_data", "/mnt/cephfs/lab/zhanghuangzhao/cgmh_data/vocab_1M_processed.pkl",
                           "vocabulary of the dataset for language model")

tf.app.flags.DEFINE_string("gpu", "0,1,2,3",
                           "gpu selection")
tf.app.flags.DEFINE_boolean("is_training", False,
                            "train the model, or test it")

tf.app.flags.DEFINE_float("keep_prob", 0.9,
                          "probability of keeping when performing drop out")
tf.app.flags.DEFINE_float("learning_rate", 1e-3,
                          "learning rate")
tf.app.flags.DEFINE_float("grad_clip", 1,
                          "gradient clipping")
tf.app.flags.DEFINE_string("forward_log", "./log/forward_1M_processed.log",
                           "path of the log of the forward model")
tf.app.flags.DEFINE_string("backward_log", "./log/backward_1M_processed.log",
                           "path of the log of the backward model")
tf.app.flags.DEFINE_integer("n_epoch", 100,
                            "number of epoches")
tf.app.flags.DEFINE_integer("batch_size", 128,
                            "batch size")

def print_flags():
    
    flags = tf.app.flags.FLAGS
    print ("sequence length = "+str(flags.seq_max_len))
    print ("embedding width = "+str(flags.embed_width))
    print ("vocabulary size = "+str(flags.vocab_size))
    print ("rnn layer number = "+str(flags.n_rnn_layer))
    print ("rnn layer width = "+str(flags.n_rnn_cell))
    print ()
    print ("forward model path = "+flags.forward_model)
    print ("backward model path = "+flags.backward_model)
    print ("vocabulary file path = "+flags.lm_vocab_data)
    print ("gpu selection = "+flags.gpu)
    if not flags.is_training:
        print ("\ntesting mode")
    else:
        print ("\ntraining mode")
        print ("input file path = "+flags.lm_input_data)
        print ("output file path = "+flags.lm_output_data)
        print ("drop out rate = "+str(1-flags.keep_prob))
        print ("learning rate = "+str(flags.learning_rate))
        print ("gradient clipping = "+str(flags.grad_clip))
        print ("epoch number = "+str(flags.n_epoch))
        print ("batch size = "+str(flags.batch_size))
        print ("forward log path = "+flags.forward_log)
        print ("backward log path = "+flags.backward_log)
    print ()