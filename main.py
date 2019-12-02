#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy
import random
import string
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import sys, os
import pickle as pkl
import pandas as pd
import copy

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
from util import BucketedDataIterator
from util import get_part_of_speech

def main(_):
    
    os.environ['CUDA_VISIBLE_DEVICES'] = flags.gpu
    numpy.seterr(divide='ignore', invalid='ignore')
    
    if flags.is_training:
    
        config.print_flags()
        d = dataset.Dataset(input_data_path=flags.lm_input_data,
                            output_data_path=flags.lm_output_data,
                            dict_path=flags.lm_vocab_data,
                            max_seq_len=flags.seq_max_len,
                            dict_size=flags.vocab_size,
                            train_ratio=0.8,
                            processed=True)
        m = mh.MetropolisHastings(seq_max_len=flags.seq_max_len,
                                  embed_w=flags.embed_width,
                                  vocab_size=d.get_dict_size(),
                                  n_layer=flags.n_rnn_layer,
                                  n_hidden=flags.n_rnn_cell,
                                  keep_prob=flags.keep_prob,
                                  lr=flags.learning_rate,
                                  init_idx=d.get_vocab_idx("<__INIT__>"),
                                  n_gpu=len(flags.gpu.split(",")),
                                  grad_clip=flags.grad_clip,
                                  is_training=True)
        
        init_op = tf.global_variables_initializer()
        configs = tf.ConfigProto(allow_soft_placement=True)
        configs.gpu_options.allow_growth = True
        sess = tf.Session(config=configs)
        sess.run(init_op)
        
        os.system("rm -rf "+flags.forward_log)
        os.system("rm -rf "+flags.backward_log)
        
        m.train(sess=sess,
                max_epoch=flags.n_epoch,
                batch_size=flags.batch_size,
                dataset=d,
                forward_model_save_path=flags.forward_model,
                backward_model_save_path=flags.backward_model,
                forward_log_save_path=flags.forward_log,
                backward_log_save_path=flags.backward_log)
        
    else:
        bb_path = "imdb_wb"     # Root dir of the black box model
        bb_data = pd.read_pickle(os.path.join(bb_path,
                                                 "data/test_df_file"))
        bb_dataset = BucketedDataIterator(bb_data, 4)
        with open(os.path.join(bb_path, "data/emb_matrix"), "rb") as f:
            bb_embed_matrix, bb_word2idx, bb_idx2word = pkl.load(f)
        (bb_max_seqlen, ) = bb_data['text'][0].shape
        bb = classifier.BiRNNSequenceClassifier(max_seqlen=bb_max_seqlen,
                                                n_class=2,
                                                average_hidden=True,
                                                concat_fw_bw=True,
                                                embed_matrix=bb_embed_matrix,
                                                embed_trainable=False, 
                                                opt_type="adam",
                                                lr=1e-3,
                                                cell_type="lstm",
                                                hidden_size=128)
        bb_saver = tf.train.Saver(var_list=tf.trainable_variables())
        bb_latest_ckpt_file = tf.train.latest_checkpoint(os.path.join(bb_path,
                                                                      "bilstm"))
        
        vocab = dataset.Dictionary(dict_path=flags.lm_vocab_data,
                                   dict_size=flags.vocab_size)
        punkts = []
        for i in string.punctuation:
            if not vocab.get_vocab_idx(i) == vocab.get_unk_idx():
                punkts.append(vocab.get_vocab_idx(i))
        m = mh.MetropolisHastings(seq_max_len=flags.seq_max_len,
                                  embed_w=flags.embed_width,
                                  vocab_size=vocab.get_dict_size(),
                                  n_layer=flags.n_rnn_layer,
                                  n_hidden=flags.n_rnn_cell,
                                  keep_prob=flags.keep_prob,
                                  lr=flags.learning_rate,
                                  n_gpu=[3],
                                  grad_clip=flags.grad_clip,
                                  init_idx=vocab.get_init_idx(),
                                  is_training=False,
                                  punkt_idx=punkts)
        
        init_op = tf.global_variables_initializer()
        configs = tf.ConfigProto(allow_soft_placement=True)
        configs.gpu_options.allow_growth = True
        sess = tf.Session(config=configs)
        sess.run(init_op)
        
        m.load(sess, forward_model_save_path=flags.forward_model,
               backward_model_save_path=flags.backward_model)
        bb_saver.restore(sess, bb_latest_ckpt_file)
        
        print ("MODELS LOADED!")
        
        bb_atk_data = {"raw": [], "label": [], "prob": [], "bb_x": [], "bb_l": []}
        cnt = 0
        for i in range(len(bb_data['text'])):
            x, y, l, seq = bb_dataset.next_batch(1)
            if bb.infer(sess, x, l)[0] == y and l[0] <= flags.atk_seq_max_len:
                bb_atk_data["raw"].append(nltk.word_tokenize(seq[0].lower()))
                bb_atk_data["label"].append(y[0])
                bb_atk_data["bb_x"].append(x[0])
                bb_atk_data["bb_l"].append(l[0])
                bb_atk_data["prob"].append(sess.run(bb.prob, feed_dict={bb.X: x,
                                                                        bb.L: l})[0][y[0]])
                cnt += 1
                break
            print("\r%d/%d processing..." % (i, len(bb_data['text'])), end="")
        print()
        bb_atk_data_size = cnt
        
        print ("DATA GENERATED! DATA SIZE = %d" % bb_atk_data_size)
        '''
        for i in range(bb_atk_data_size):
            print ("%d/%d" % (i+1, bb_atk_data_size), end="\t")
            if bb_atk_data['label'][i] == 1:
                print("POS with prob = %.3f" % bb_atk_data["prob"][i])
            else:
                print("NEG with prob = %.3f" % bb_atk_data["prob"][i])
            for j in range(len(bb_atk_data["raw"][i])):
                print(bb_atk_data["raw"][i][j], end=" ")
            print ()
            for j in range(bb_atk_data['bb_l'][i]):
                print(bb_idx2word[bb_atk_data['bb_x'][i][j]], end=" ")
            print("\n")
        '''
        
        op_prob = [flags.swp_prob,
                   flags.ins_prob,
                   flags.del_prob,
                   flags.pass_prob]
        op_prob = op_prob / numpy.sum(op_prob)
        n_sample = flags.sample_max_n
        n_candidate = flags.n_candidate
        just_acc_rate = flags.just_acc_rate
        threshold = flags.lm_threshold
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
        lemmatzr = WordNetLemmatizer()
        
        for i in range(bb_atk_data_size):
            print ("===== DATA %d/%d =====" % (i+1, bb_atk_data_size))
            res_log.append([])
            raw = copy.deepcopy(bb_atk_data["raw"][i])
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
                
            bb_y = bb_atk_data["label"][i]
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
                  end="\n\t")
            for ii in range(len(raw)):
                print (raw[ii], end=" ")
            if bb_y == 1:
                print("\t<POS>")
            else:
                print("\t<NEG>")
            while sample_all < n_sample:
                sample_all += 1
                op = random_pick_idx_with_unnormalized_prob(op_prob)
                if op == 3:
                    tmp_prob = sess.run(bb.prob, feed_dict={bb.X: [bb_seq],
                                                            bb.L: [bb_l]})[0][1-bb_y]
                    if tmp_prob >= 0.5:
                        res_log[i].append((sample_all, 1))
                        print ("%d/%d\t%d acc / %d all\tPASS\t SUCC with %.5f" %
                               (i+1, bb_atk_data_size, sample_cnt+1, sample_all+1, tmp_prob))
                    else:
                        res_log[i].append((sample_all, 0))
                        print ("%d/%d\t%d acc / %d all\tPASS\t FAIL with %.5f" %
                               (i+1, bb_atk_data_size, sample_cnt+1, sample_all+1, tmp_prob))
                    sample_cnt += 1
                    sents[-1].append(copy.deepcopy(raw))
                    print("", end="\t")
                    for ii in range(len(raw)):
                        print (raw[ii], end=" ")
                    if bb_y == 1:
                        print("\t<POS>")
                    else:
                        print("\t<NEG>")
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
                        #print ("%s (%d, %.3e)" % (raw[idx-1], idx, candidate_grads[idx_idx]))
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
                    if mode == "grad":
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
                        and proposal["old_prob"] * threshold <= proposal["new_prob"]
                        and old_wrong_prob * swp_prob_threshold <= new_wrong_prob
                        and (mode == "grad"
                            and (new_obj > swn_obj_threshold        # objective
                                or (new_obj <= swn_obj_threshold    # neutral
                                and abs(new_pos) <= swn_pos_threshold))))):
                        if new_wrong_prob >= 0.5:
                            res_log[i].append((sample_all, 1))
                            print ("%d/%d\t%d acc / %d all\tSWP\t SUCC with %.5f\t[%s](%d) => [%s](%d) (%d)" %
                                   (i+1, bb_atk_data_size, sample_cnt+1, sample_all, new_wrong_prob,
                                    vocab.get_vocab(seq[idx]), seq[idx],
                                    vocab.get_vocab(proposal["proposal"][idx]), proposal["proposal"][idx], idx))
                        else:
                            res_log[i].append((sample_all, 0))
                            print ("%d/%d\t%d acc / %d all\tSWP\t FAIL with %.5f\t[%s](%d) => [%s](%d) (%d)" %
                                   (i+1, bb_atk_data_size, sample_cnt+1, sample_all, new_wrong_prob,
                                    vocab.get_vocab(seq[idx]), seq[idx],
                                    vocab.get_vocab(proposal["proposal"][idx]), proposal["proposal"][idx], idx))
                        sample_cnt += 1
                        seq = proposal["proposal"]
                        bb_seq = tmp_bb_seq
                        raw = tmp_raw
                        sents[-1].append(copy.deepcopy(raw))
                        print("", end="\t")
                        for ii in range(len(raw)):
                            print (raw[ii], end=" ")
                        if bb_y == 1:
                            print("\t<POS>")
                        else:
                            print("\t<NEG>")
                    else:
                        print ("%d/%d\t%d acc / %d all\tSWP\talpha %.2e" %
                                   (i+1, bb_atk_data_size, sample_cnt, sample_all, proposal["alpha"]))
                
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
                    if mode == "grad":
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
                        and proposal["old_prob"] * threshold <= proposal["new_prob"]
                        and old_wrong_prob * ins_prob_threshold <= new_wrong_prob
                        and (mode == "grad"
                             and (new_obj > swn_obj_threshold        # objective
                                  or (new_obj <= swn_obj_threshold   # neutral
                                 and new_pos <= swn_pos_threshold))))):
                        if new_wrong_prob >= 0.5:
                            res_log[i].append((sample_all, 1))
                            print ("%d/%d\t%d acc / %d all\tINS\t SUCC with %.5f\t[] => [%s](%d,%.1f,%.1f) (%d)" %
                                   (i+1, bb_atk_data_size, sample_cnt+1, sample_all, new_wrong_prob,
                                    vocab.get_vocab(proposal["proposal"][idx+1]), proposal["proposal"][idx+1],
                                    new_obj, new_pos, idx))
                        else:
                            res_log[i].append((sample_all, 0))
                            print ("%d/%d\t%d acc / %d all\tINS\t FAIL with %.5f\t[] => [%s](%d,%.1f,%.1f) (%d)" %
                                   (i+1, bb_atk_data_size, sample_cnt+1, sample_all, new_wrong_prob,
                                    vocab.get_vocab(proposal["proposal"][idx+1]), proposal["proposal"][idx+1],
                                    new_obj, new_pos, idx))
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
                        print("", end="\t")
                        for ii in range(len(raw)):
                            print (raw[ii], end=" ")
                        if bb_y == 1:
                            print("\t<POS>")
                        else:
                            print("\t<NEG>")
                    else:
                        print ("%d/%d\t%d acc / %d all\tINS\talpha %.2e" %
                                   (i+1, bb_atk_data_size, sample_cnt, sample_all, proposal["alpha"]))
                
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
                        and proposal["old_prob"] * threshold <= proposal["new_prob"]
                        and old_wrong_prob * del_prob_threshold <= new_wrong_prob)):
                        if new_wrong_prob >= 0.5:
                            res_log[i].append((sample_all, 1))
                            print ("%d/%d\t%d acc / %d all\tDEL\t SUCC with %.5f\t[%s](%d) => [] (%d)" %
                                   (i+1, bb_atk_data_size, sample_cnt+1, sample_all, new_wrong_prob,
                                    vocab.get_vocab(seq[idx]), seq[idx], idx))
                        else:
                            res_log[i].append((sample_all, 0))
                            print ("%d/%d\t%d acc / %d all\tDEL\t FAIL with %.5f\t[%s](%d) => [] (%d)" %
                                   (i+1, bb_atk_data_size, sample_cnt+1, sample_all, new_wrong_prob,
                                    vocab.get_vocab(seq[idx]), seq[idx], idx))
                        sample_cnt+= 1
                        seq = proposal["proposal"]
                        bb_seq = tmp_bb_seq
                        l -= 1
                        mask = mask[:idx] + mask[idx+1:]
                        bb_l = tmp_bb_l
                        raw = raw[:idx-1] + raw[idx:]
                        sents[-1].append(copy.deepcopy(raw))
                        print("", end="\t")
                        for ii in range(len(raw)):
                            print (raw[ii], end=" ")
                        if bb_y == 1:
                            print("\t<POS>")
                        else:
                            print("\t<NEG>")
                    else:
                        print ("%d/%d\t%d acc / %d all\tDEL\talpha %.2e" %
                                   (i+1, bb_atk_data_size, sample_cnt, sample_all, proposal["alpha"]))
                
                assert len(mask) == l
                
            with open("log/res_log.log", "wb") as f:
                pkl.dump((res_log, sents), f)
        
if __name__ == "__main__":
    
    tf.app.run(main=main)