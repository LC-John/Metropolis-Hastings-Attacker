#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy
import random
import string
import nltk
import sys, os

nltk.download('sentiwordnet')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

def flush():
    sys.stdout.flush()
    sys.stderr.flush()
    
flush()

from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

import pickle as pkl
import gzip as gz
import pandas as pd
import copy
import threading
import time

next(swn.all_senti_synsets()) 
next(wn.words())

import config
flags = tf.app.flags.FLAGS

if flags.is_training:
    import mh
    import dataset
else:
    import mh
    import dataset
    import classifier

from util import random_pick_idx_with_unnormalized_prob
from util import just_acc
from util import get_part_of_speech

class multi_thread_cgmh(threading.Thread):
    
    def __init__(self, bb_atk_data, bb_w2i, bb_i2w, vocab, bb, m, sess, index=0, bb_max_seqlen=400, negs=[]):
        
        threading.Thread.__init__(self)
        self.__bb_atk_data = bb_atk_data
        self.__vocab = vocab
        self.__bb_word2idx = bb_w2i
        self.__bb_idx2word = bb_i2w
        self.__bb_max_seqlen = bb_max_seqlen
        self.__bb = bb
        self.__model = m
        self.__sess = sess
        self.__idx = index
        self.__negs = negs
        
    def run(self):
        
        out_dir = flags.cgmh_output_dir
        fout = open(os.path.join(out_dir,
                                 "log"+str(self.__idx)+".log"), "w")
        res_path = os.path.join(out_dir,
                                "res"+str(self.__idx)+".res")
        
        bb_atk_data = self.__bb_atk_data
        bb_atk_data_size = len(self.__bb_atk_data['raw'])
        bb_word2idx = self.__bb_word2idx
        bb_idx2word = self.__bb_idx2word
        vocab = self.__vocab
        bb = self.__bb
        m = self.__model
        bb_max_seqlen = self.__bb_max_seqlen
        sess = self.__sess
        negations = self.__negs
        
        op_prob = [flags.swp_prob,
                   flags.ins_prob,
                   flags.del_prob,
                   flags.pass_prob]
        op_prob = op_prob / numpy.sum(op_prob)
        n_sample = flags.sample_max_n
        n_candidate = flags.n_candidate
        just_acc_rate = flags.just_acc_rate
        swp_lm_threshold = flags.lm_swp_threshold
        ins_lm_threshold = flags.lm_ins_threshold
        del_lm_threshold = flags.lm_del_threshold
        swp_prob_threshold = flags.swp_threshold
        ins_prob_threshold = flags.ins_threshold
        del_prob_threshold = flags.del_threshold
        swn_obj_threshold = flags.senti_obj_threshold
        swn_pos_threshold = flags.senti_pos_threshold
        seq_min_len = flags.seq_min_len
        mode = flags.index_mode
            
        res_log = []
        sents = []
        idx = 0
        op = 3
        
        total_time = 0
        n_succ = 0
        
        lemmatzr = WordNetLemmatizer()
            
        for i in range(bb_atk_data_size):
            
            start_time = time.time()
            
            print ("===== DATA %d/%d =====" % (i+1, bb_atk_data_size),
                   file=fout, flush=True)
            print ("DATA %d/%d, id=%d" % (i+1, bb_atk_data_size, self.__idx))
            flush()
            res_log.append([])
            raw = copy.deepcopy(bb_atk_data["raw"][i])
            raw = nltk.word_tokenize(raw.lower())
            l = len(raw) + 1
            if (l > flags.seq_max_len):
                l = flags.seq_max_len
            seq = [vocab.get_init_idx()]
            for ii in range(1, l):
                seq.append(vocab.get_vocab_idx(raw[ii-1]))
            while len(seq) < flags.seq_max_len:
                seq.append(vocab.get_pad_idx())
            mask = [True]
            for ii in range(1, l):
                mask.append(False)
                
            bb_y = bb_atk_data["y"][i]
            bb_l = len(raw)
            if (bb_l > bb_max_seqlen):
                bb_l = bb_max_seqlen
            bb_seq = [] 
            for ii in range(bb_l):
                if raw[ii] in bb_word2idx.keys():
                    bb_seq.append(bb_word2idx[raw[ii]])
                else:
                    bb_seq.append(bb_word2idx["<unk>"])
            while len(bb_seq) < bb_max_seqlen:
                bb_seq.append(bb_word2idx["<pad>"])
            
            sents.append([])
            sample_cnt = 0
            sample_all = 0
            idx = 0
            sents[-1].append(copy.deepcopy(raw))
            print("%d/%d\tOriginal\tFAIL with %.5f" % (i+1, bb_atk_data_size, 1-bb_atk_data["prob"][i]),
                  end="\n\t", file=fout, flush=True)
            for ii in range(len(raw)):
                print (raw[ii], end=" ", file=fout, flush=True)
            if bb_y == 1:
                print("\t<POS>", file=fout, flush=True)
            else:
                print("\t<NEG>", file=fout, flush=True)
            
            while sample_all < n_sample:
                
                try:
                    
                    wn.ensure_loaded()
                    
                    sample_all += 1
                    op = random_pick_idx_with_unnormalized_prob(op_prob)
                    succ = False
                    if op == 3:
                        tmp_prob = sess.run(bb.prob, feed_dict={bb.X: [bb_seq],
                                                                bb.L: [bb_l]})[0][1-bb_y]
                        if tmp_prob >= 0.5:
                            res_log[i].append((sample_all, 1))
                            print ("%d/%d\t%d acc / %d all\tPASS\t SUCC with %.5f" %
                                   (i+1, bb_atk_data_size, sample_cnt+1, sample_all+1, tmp_prob),
                                   file=fout, flush=True)
                            succ = True
                        else:
                            res_log[i].append((sample_all, 0))
                            print ("%d/%d\t%d acc / %d all\tPASS\t FAIL with %.5f" %
                                   (i+1, bb_atk_data_size, sample_cnt+1, sample_all+1, tmp_prob),
                                   file=fout, flush=True)
                        sample_cnt += 1
                        sents[-1].append(copy.deepcopy(raw))
                        print("", end="\t", file=fout, flush=True)
                        for ii in range(len(raw)):
                            print (raw[ii], end=" ", file=fout, flush=True)
                        if bb_y == 1:
                            print("\t<POS>", file=fout, flush=True)
                        else:
                            print("\t<NEG>", file=fout, flush=True)
                        if succ:
                            print ("\tSUCC!")
                            flush()
                            break
                        continue
                        
                    if mode == "random":
                        idx = random.randint(0, l-1)
                    elif mode == "traverse":
                        idx = (idx + 1) % l
                    elif mode == "grad":
                        if op == 1:
                            idx = random.randint(0, l-1)
                        else:
                            grad_vecs = sess.run(bb.embed_grad, feed_dict={bb.X: [bb_seq],
                                                                           bb.L: [bb_l],
                                                                           bb.Y: [1-bb_y]})[0][0]
                            grads = numpy.linalg.norm(grad_vecs, axis=-1)
                            candidate_grads = []
                            candidate_idxs = []
                            position_tag = nltk.pos_tag(raw)
                            for pos in range(len(position_tag)):
                                tmp_tag = get_part_of_speech(position_tag[pos][1])
                                if tmp_tag is None:
                                    candidate_grads.append(grads[pos])
                                    candidate_idxs.append(pos+1)
                                    continue
                                tmp_wn = wn.synsets(lemmatzr.lemmatize(raw[pos]), pos=tmp_tag)
                                if len(tmp_wn) <= 0:
                                    candidate_grads.append(grads[pos])
                                    candidate_idxs.append(pos+1)
                                    continue
                                tmp_swn = swn.senti_synset(tmp_wn[0].name())
                                if (tmp_swn.obj_score() > swn_obj_threshold \
                                    or (tmp_swn.obj_score() <= swn_obj_threshold \
                                        and abs(tmp_swn.pos_score()-tmp_swn.neg_score()) <= swn_pos_threshold)):
                                    candidate_grads.append(grads[pos])
                                    candidate_idxs.append(pos+1)
                                    continue
                            idx_idx = random_pick_idx_with_unnormalized_prob(candidate_grads)
                            idx = candidate_idxs[idx_idx]
                    else:
                        assert False, "Invalid mode \""+mode+"\""
                            
                        
                    old_wrong_prob = sess.run(bb.prob, feed_dict={bb.X: [bb_seq],
                                                                  bb.L: [bb_l]})[0][1-bb_y]
                    if op == 0:
                        if mask[idx]:
                            continue
                        proposal = m.op_replace(sess, copy.deepcopy(seq), l,
                                                copy.deepcopy(bb_seq), bb_l, 1-bb_y,
                                                idx, n_candidate, op_prob,
                                                vocab, bb_word2idx, bb)
                        tmp_bb_seq = copy.deepcopy(bb_seq)
                        tmp_str = vocab.get_vocab(proposal['proposal'][idx])
                        if tmp_str in bb_word2idx.keys():
                            tmp_bb_seq[idx-1] = bb_word2idx[tmp_str]
                        else:
                            tmp_bb_seq[idx-1] = bb_word2idx["<unk>"]
                        new_wrong_prob = sess.run(bb.prob, feed_dict={bb.X: [tmp_bb_seq], 
                                                                      bb.L: [bb_l]})[0][1-bb_y]
                        tmp_raw = copy.deepcopy(raw)
                        tmp_raw[idx-1] = vocab.get_vocab(proposal["proposal"][idx])
                        new_tag = get_part_of_speech(nltk.pos_tag(tmp_raw)[idx-1][1])
                        if new_tag is None:
                            new_obj = 1
                            new_pos = 0
                        else:
                            new_wn = wn.synsets(lemmatzr.lemmatize(tmp_raw[idx-1]), pos=new_tag)
                            if len(new_wn) <= 0:
                                new_obj = 1
                                new_pos = 0
                            else:
                                new_swn = swn.senti_synset(new_wn[0].name())
                                new_obj = new_swn.obj_score()
                                new_pos = new_swn.pos_score() - new_swn.neg_score()
                        if (just_acc(just_acc_rate)
                            or (numpy.random.uniform(0,1) <= \
                                proposal["alpha"] * new_wrong_prob / old_wrong_prob
                            and proposal["old_prob"] * swp_lm_threshold <= proposal["new_prob"]
                            and old_wrong_prob * swp_prob_threshold <= new_wrong_prob
                            and (new_obj > swn_obj_threshold        # objective
                                      or (new_obj <= swn_obj_threshold    # neutral
                                    and abs(new_pos) <= swn_pos_threshold))
                            and (tmp_str not in negations))):
                            if new_wrong_prob >= 0.5:
                                res_log[i].append((sample_all, 1))
                                print ("%d/%d\t%d acc / %d all\tSWP\t SUCC with %.5f\t[%s](%d) => [%s](%d) (%d)" %
                                       (i+1, bb_atk_data_size, sample_cnt+1, sample_all, new_wrong_prob,
                                        vocab.get_vocab(seq[idx]), seq[idx],
                                        vocab.get_vocab(proposal["proposal"][idx]), proposal["proposal"][idx], idx),
                                       file=fout, flush=True)
                                succ = True
                            else:
                                res_log[i].append((sample_all, 0))
                                print ("%d/%d\t%d acc / %d all\tSWP\t FAIL with %.5f\t[%s](%d) => [%s](%d) (%d)" %
                                       (i+1, bb_atk_data_size, sample_cnt+1, sample_all, new_wrong_prob,
                                        vocab.get_vocab(seq[idx]), seq[idx],
                                        vocab.get_vocab(proposal["proposal"][idx]), proposal["proposal"][idx], idx),
                                       file=fout, flush=True)
                            sample_cnt += 1
                            seq = proposal["proposal"]
                            bb_seq = tmp_bb_seq
                            raw = tmp_raw
                            sents[-1].append(copy.deepcopy(raw))
                            print("", end="\t", file=fout, flush=True)
                            for ii in range(len(raw)):
                                print (raw[ii], end=" ", file=fout, flush=True)
                            if bb_y == 1:
                                print("\t<POS>", file=fout, flush=True)
                            else:
                                print("\t<NEG>", file=fout, flush=True)
                        else:
                            print ("%d/%d\t%d acc / %d all\tSWP\talpha %.2e" %
                                   (i+1, bb_atk_data_size, sample_cnt, sample_all, proposal["alpha"]),
                                   file=fout, flush=True)
                        
                    elif op == 1:
                        if idx == l-1:
                            continue
                        proposal = m.op_insert(sess, copy.deepcopy(seq), l,
                                               copy.deepcopy(bb_seq), bb_l, 1-bb_y,
                                               idx, n_candidate, op_prob,
                                               vocab, bb_word2idx, bb)
                        tmp_bb_seq = numpy.asarray(copy.deepcopy(bb_seq)).tolist()
                        tmp_str = vocab.get_vocab(proposal['proposal'][idx+1])
                        if tmp_str in bb_word2idx.keys():
                            tmp_bb_seq = tmp_bb_seq[:idx]+[bb_word2idx[tmp_str]]+tmp_bb_seq[idx:]
                        else:
                            tmp_bb_seq = tmp_bb_seq[:idx]+[bb_word2idx["<unk>"]]+tmp_bb_seq[idx:]
                        tmp_bb_seq = tmp_bb_seq[:-1]
                        tmp_bb_l = bb_l + 1
                        if tmp_bb_l >bb_max_seqlen:
                            tmp_bb_l = bb_max_seqlen
                        new_wrong_prob = sess.run(bb.prob, feed_dict={bb.X: [tmp_bb_seq], 
                                                                      bb.L: [tmp_bb_l]})[0][1-bb_y]
                        tmp_raw = copy.deepcopy(raw)
                        tmp_raw = tmp_raw[:idx]+[tmp_str]+tmp_raw[idx:]
                        new_tag = get_part_of_speech(nltk.pos_tag(tmp_raw)[idx][1])
                        if new_tag is None:
                            new_obj = 1
                            new_pos = 0
                        else:
                            new_wn = wn.synsets(lemmatzr.lemmatize(tmp_raw[idx]), pos=new_tag)
                            if len(new_wn) <= 0:
                                new_obj = 1
                                new_pos = 0
                            else:
                                new_swn = swn.senti_synset(new_wn[0].name())
                                new_obj = new_swn.obj_score()
                                new_pos = new_swn.pos_score() - new_swn.neg_score()
                        if (just_acc(just_acc_rate)
                            or (numpy.random.uniform(0,1) <= \
                                proposal["alpha"] * new_wrong_prob / old_wrong_prob
                            and proposal["old_prob"] * ins_lm_threshold <= proposal["new_prob"]
                            and old_wrong_prob * ins_prob_threshold <= new_wrong_prob
                            and (new_obj > swn_obj_threshold        # objective
                                      or (new_obj <= swn_obj_threshold   # neutral
                                     and new_pos <= swn_pos_threshold))
                            and (tmp_str not in negations))):
                            if new_wrong_prob >= 0.5:
                                res_log[i].append((sample_all, 1))
                                print ("%d/%d\t%d acc / %d all\tINS\t SUCC with %.5f\t[] => [%s](%d,%.1f,%.1f) (%d)" %
                                       (i+1, bb_atk_data_size, sample_cnt+1, sample_all, new_wrong_prob,
                                        vocab.get_vocab(proposal["proposal"][idx+1]), proposal["proposal"][idx+1],
                                        new_obj, new_pos, idx), file=fout, flush=True)
                                succ = True
                            else:
                                res_log[i].append((sample_all, 0))
                                print ("%d/%d\t%d acc / %d all\tINS\t FAIL with %.5f\t[] => [%s](%d,%.1f,%.1f) (%d)" %
                                       (i+1, bb_atk_data_size, sample_cnt+1, sample_all, new_wrong_prob,
                                        vocab.get_vocab(proposal["proposal"][idx+1]), proposal["proposal"][idx+1],
                                        new_obj, new_pos, idx), file=fout, flush=True)
                            sample_cnt += 1
                            seq = proposal["proposal"]
                            bb_seq = tmp_bb_seq
                            l += 1
                            mask = mask[:idx+1] + [False] + mask[idx+1:]
                            if l > flags.seq_max_len:
                                l = flags.seq_max_len
                            mask = mask[:l]
                            bb_l = tmp_bb_l
                            raw = raw[:idx] + [vocab.get_vocab(seq[idx+1])] + raw[idx:]
                            sents[-1].append(copy.deepcopy(raw))
                            print("", end="\t", file=fout, flush=True)
                            for ii in range(len(raw)):
                                print (raw[ii], end=" ", file=fout, flush=True)
                            if bb_y == 1:
                                print("\t<POS>", file=fout, flush=True)
                            else:
                                print("\t<NEG>", file=fout, flush=True)
                        else:
                            print ("%d/%d\t%d acc / %d all\tINS\talpha %.2e" %
                                   (i+1, bb_atk_data_size, sample_cnt, sample_all, proposal["alpha"]),
                                   file=fout, flush=True)
                        
                    elif op == 2:
                        if mask[idx] or l-1 < seq_min_len:
                            continue
                        proposal = m.op_delete(sess, copy.deepcopy(seq), l,
                                               copy.deepcopy(bb_seq), bb_l, 1-bb_y,
                                               idx, n_candidate, op_prob,
                                               vocab, bb_word2idx, bb)
                        tmp_bb_seq = numpy.asarray(copy.deepcopy(bb_seq)).tolist()
                        tmp_str = vocab.get_vocab(seq[idx])
                        tmp_bb_seq = tmp_bb_seq[:idx-1]+tmp_bb_seq[idx:]+[bb_word2idx['<pad>']]
                        tmp_bb_l = bb_l - 1
                        new_wrong_prob = sess.run(bb.prob, feed_dict={bb.X: [tmp_bb_seq], 
                                                                      bb.L: [tmp_bb_l]})[0][1-bb_y]
                        if (just_acc(just_acc_rate)
                            or (numpy.random.uniform(0,1) <= \
                                proposal["alpha"] * new_wrong_prob / old_wrong_prob
                            and proposal["old_prob"] * del_lm_threshold <= proposal["new_prob"]
                            and old_wrong_prob * del_prob_threshold <= new_wrong_prob)
                            and (tmp_str not in negations)):
                            if new_wrong_prob >= 0.5:
                                res_log[i].append((sample_all, 1))
                                print ("%d/%d\t%d acc / %d all\tDEL\t SUCC with %.5f\t[%s](%d) => [] (%d)" %
                                       (i+1, bb_atk_data_size, sample_cnt+1, sample_all, new_wrong_prob,
                                        vocab.get_vocab(seq[idx]), seq[idx], idx), file=fout, flush=True)
                                succ = True
                            else:
                                res_log[i].append((sample_all, 0))
                                print ("%d/%d\t%d acc / %d all\tDEL\t FAIL with %.5f\t[%s](%d) => [] (%d)" %
                                       (i+1, bb_atk_data_size, sample_cnt+1, sample_all, new_wrong_prob,
                                        vocab.get_vocab(seq[idx]), seq[idx], idx), file=fout, flush=True)
                            sample_cnt+= 1
                            seq = proposal["proposal"]
                            bb_seq = tmp_bb_seq
                            l -= 1
                            mask = mask[:idx] + mask[idx+1:]
                            bb_l = tmp_bb_l
                            raw = raw[:idx-1] + raw[idx:]
                            sents[-1].append(copy.deepcopy(raw))
                            print("", end="\t", file=fout, flush=True)
                            for ii in range(len(raw)):
                                print (raw[ii], end=" ", file=fout, flush=True)
                            if bb_y == 1:
                                print("\t<POS>", file=fout, flush=True)
                            else:
                                print("\t<NEG>", file=fout, flush=True)
                        else:
                            print ("%d/%d\t%d acc / %d all\tDEL\talpha %.2e" %
                                   (i+1, bb_atk_data_size, sample_cnt, sample_all, proposal["alpha"]),
                                   file=fout, flush=True)
                    
                    if succ:
                        end_time = time.time()
                        total_time += end_time - start_time
                        n_succ += 1
                        print ("\tSUCC!")
                        print ("\t\ttime =", total_time, n_succ)
                        flush()
                        break
                    
                    assert len(mask) == l
                    
                except Exception as e:
                    
                    print ("Something went wrong... Abort!", file=fout, flush=True)
                    print ("Something went wrong... Abort! -- Thread %d" % self.__idx)
                    print ("\t", e)
                    sys.stdout.flush()
                    sys.stderr.flush()
                    continue
                
            with open(res_path, "wb") as f:
                pkl.dump((res_log, sents), f)

def main(_):
    
    wn.ensure_loaded()
    
    if not os.path.isdir(flags.cgmh_output_dir):
        os.mkdir(flags.cgmh_output_dir)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = flags.gpu
    n_gpu = len(flags.gpu.split(","))
    numpy.seterr(divide='ignore', invalid='ignore')
    
    with gz.open(flags.atk_data_path, "rb") as f:
        bb_atk_data = pkl.load(f)
    bb_atk_data_list = []
    bb_data_chunk_size = int(numpy.ceil(len(bb_atk_data['raw'])/n_gpu/flags.models_per_gpu))
    for i in range(n_gpu*flags.models_per_gpu-1):
        tmp_bb_atk_data = {}
        for k in bb_atk_data.keys():
            tmp_bb_atk_data[k] = bb_atk_data[k][int(i*bb_data_chunk_size) \
                                                :int((i+1)*bb_data_chunk_size)]
        bb_atk_data_list.append(tmp_bb_atk_data)
    tmp_bb_atk_data = {}
    for k in bb_atk_data.keys():
        tmp_bb_atk_data[k] = bb_atk_data[k][int((n_gpu*flags.models_per_gpu-1)*bb_data_chunk_size):]
    bb_atk_data_list.append(tmp_bb_atk_data)
    print ("ATTACK DATA LOADED!")
    for i in range(n_gpu*flags.models_per_gpu):
        print ("\tchunk %d, size = %d" % (i, len(bb_atk_data_list[i]["raw"])),
               list(bb_atk_data_list[i].keys()))
    flush()
    
    bb_path = flags.imdb_root_path     # Root dir of the black box model
    bb_data = pd.read_pickle(os.path.join(bb_path, "data/test_df_file"))
    with open(os.path.join(bb_path, "data/emb_matrix"), "rb") as f:
        bb_embed_matrix, bb_word2idx, bb_idx2word = pkl.load(f)
    (bb_max_seqlen, ) = bb_data['text'][0].shape
    
    vocab = dataset.Dictionary(dict_path=flags.lm_vocab_data,
                               dict_size=flags.vocab_size)
    punkts = []
    for i in string.punctuation:
        if not vocab.get_vocab_idx(i) == vocab.get_unk_idx():
            punkts.append(vocab.get_vocab_idx(i))
    negations = ["no", "not", "n't", "nt"]
           
    '''
    print (bb_atk_data_list[0]['raw'][0])
    for i in bb_atk_data_list[0]["x"][0]:
        print (bb_idx2word[i], end=" ")
    print ()
    for i in bb_atk_data_list[0]["x"][0]:
        print (vocab.get_vocab(i), end=" ")
    print ()
    '''
    
    bb_saver = []
    bb_list = []
    bb_latest_ckpt_file = tf.train.latest_checkpoint(os.path.join(bb_path, flags.model_dir))
    for i in range(n_gpu*flags.models_per_gpu):
        with tf.device("/device:GPU:"+str(i%n_gpu)):
            with tf.variable_scope("IMDB"+str(i)):
                bb_list.append(classifier.BiRNNSequenceClassifier(max_seqlen=bb_max_seqlen,
                                                                  n_class=2,
                                                                  average_hidden=True,
                                                                  concat_fw_bw=True,
                                                                  embed_matrix=bb_embed_matrix,
                                                                  embed_trainable=False, 
                                                                  opt_type="adam",
                                                                  lr=1e-3,
                                                                  cell_type="lstm",
                                                                  hidden_size=128))
                tmp_var_list = {}
                for v in tf.global_variables():
                    if v.name.startswith('IMDB'+str(i)):
                        tmp_name = v.name
                        while "IMDB"+str(i) in tmp_name:
                            tmp_name = tmp_name.strip("IMDB"+str(i)+"/")
                        tmp_var_list[tmp_name.split(":")[0]] = v
                bb_saver.append(tf.train.Saver(var_list=tmp_var_list))    
    
    m_list = []
    for i in range(n_gpu*flags.models_per_gpu):
        with tf.device("/device:GPU:"+str(i)):
            with tf.variable_scope("MH"+str(i)):
                m_list.append(mh.MetropolisHastings(seq_max_len=flags.seq_max_len,
                                                    embed_w=flags.embed_width,
                                                    vocab_size=vocab.get_dict_size(),
                                                    n_layer=flags.n_rnn_layer,
                                                    n_hidden=flags.n_rnn_cell,
                                                    keep_prob=flags.keep_prob,
                                                    lr=flags.learning_rate,
                                                    n_gpu=[i%n_gpu],
                                                    grad_clip=flags.grad_clip,
                                                    init_idx=vocab.get_init_idx(),
                                                    is_training=False,
                                                    punkt_idx=punkts,
                                                    scope_name="MH"+str(i)))
    
    configs = tf.ConfigProto(allow_soft_placement=True)
    configs.gpu_options.allow_growth = True
    sess = tf.Session(config=configs)
        
    for tmp_m in m_list:
        tmp_m.load(sess,
                   forward_model_save_path=flags.forward_model,
                   backward_model_save_path=flags.backward_model)
    for i in bb_saver:
        i.restore(sess, bb_latest_ckpt_file)
        
    print ("MODELS LOADED!")
    for i in range(n_gpu*flags.models_per_gpu):
        print ("\t", m_list[i])
        print ("\t", bb_list[i])
    flush()
    threads = []
    for i in range(n_gpu*flags.models_per_gpu):
        threads.append(multi_thread_cgmh(bb_atk_data=bb_atk_data_list[i],
                                         bb_w2i=bb_word2idx,
                                         bb_i2w=bb_idx2word,
                                         vocab=vocab,
                                         bb=bb_list[i],
                                         m=m_list[i],
                                         sess=sess,
                                         index=i,
                                         bb_max_seqlen=400,
                                         negs=negations))
    print ("THREADS BUILT!")
    flush()
    for i in range(len(threads)):
        print ("\t%d"%i, threads[i])
    for i in threads:
        i.start()
    flush()
    for i in threads:
        i.join()
        
if __name__ == "__main__":
    
    tf.app.run(main=main)