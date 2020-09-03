import numpy as np
import datetime
import json
import argparse

from process_data import save_pickle, load_pickle, load_kv_pairs, lower_list, vectorize, vectorize_kv, load_kv_dataset, filter_data
from torch.autograd import Variable
import torch
import pdb

class DataLoader():
    def __init__(self, max_mem_size, batch_size, device):
        '''max_mem_size means how many memories can be visited for one query
        '''
        self.device = device

        train_data  = load_pickle('pickle/mov_task1_qa_pipe_train.pickle')
        test_data   = load_pickle('pickle/mov_task1_qa_pipe_test.pickle')
        dev_data   = load_pickle('pickle/mov_task1_qa_pipe_dev.pickle')
        kv_pairs    = load_pickle('pickle/mov_kv_pairs.pickle')
        train_k     = np.array(load_pickle('pickle/mov_train_k.pickle'))
        train_v     = np.array(load_pickle('pickle/mov_train_v.pickle'))
        test_k      = np.array(load_pickle('pickle/mov_test_k.pickle'))
        test_v      = np.array(load_pickle('pickle/mov_test_v.pickle'))
        dev_k      = np.array(load_pickle('pickle/mov_dev_k.pickle'))
        dev_v      = np.array(load_pickle('pickle/mov_dev_v.pickle'))
        entities    = load_pickle('pickle/mov_entities.pickle')
        entity_size = len(entities)

        # TODO
        vocab = load_pickle('pickle/mov_vocab.pickle')
        self.vocab_size = len(vocab)

        stopwords = load_pickle('pickle/mov_stopwords.pickle')
        w2i = load_pickle('pickle/mov_w2i.pickle')
        i2w = load_pickle('pickle/mov_i2w.pickle')

        w2i_label = load_pickle('pickle/mov_w2i_label.pickle')
        i2w_label = load_pickle('pickle/mov_i2w_label.pickle')

        print('before filter:', len(train_data), len(test_data))
        train_data, train_k, train_v = filter_data(train_data, train_k, train_v, 0, 100)
        pdb.set_trace()
        test_data, test_k, test_v = filter_data(test_data, test_k, test_v, 0, 100)
        dev_data, dev_k, dev_v = filter_data(dev_data, dev_k, dev_v, 0, 100)
        print('after filter:', len(train_data), len(test_data))

        query_maxlen = max(map(len, (x for x, _ in train_data + test_data)))

        print('-')
        print('Vocab size:', self.vocab_size, 'unique words')
        print('Query max length:', query_maxlen, 'words')
        print('Number of training data:', len(train_data))
        print('Number of test data:', len(test_data))
        print('-')
        print('Here\'s what a "data" tuple looks like (query, answer):')
        print(train_data[0])
        print('-')
        print('Vectorizing the word sequences...')

        print('Number of entities', len(entities))

        # TODO, change the vectorize function
        self.queries_train, self.answers_train = vectorize(train_data, w2i, query_maxlen, w2i_label)
        self.queries_test, self.answers_test = vectorize(test_data, w2i, query_maxlen, w2i_label)
        # queries_dev, answers_dev = vectorize(dev_data, w2i, query_maxlen, w2i_label)

        print('-')
        print('queries: integer tensor of shape (samples, max_length)')
        print('queries_train shape:', self.queries_train.shape)
        print('queries_test shape:', self.queries_test.shape)
        print('-')
        print('answers: binary (1 or 0) tensor of shape (samples, len(w2i_label))')
        print('answers_train shape:', self.answers_train.shape)
        print('answers_test shape:', self.answers_test.shape)


        # max_mem_len = 3
        self.query_max_len = query_maxlen
        self.max_mem_size = max_mem_size
        self.mem_key_len = 2 # ['Blade Runner', 'directed_by']
        self.mem_val_len = 1 # ['Ridley Scott']
        self.vec_train_k = vectorize_kv(train_k, self.mem_key_len, self.max_mem_size, w2i)
        self.vec_train_v = vectorize_kv(train_v, self.mem_val_len, self.max_mem_size, w2i)
        self.vec_test_k = vectorize_kv(test_k, self.mem_key_len, self.max_mem_size, w2i)
        self.vec_test_v = vectorize_kv(test_v, self.mem_val_len, self.max_mem_size, w2i)
        print('vec_k', self.vec_train_k.shape)
        print('vec_v', self.vec_train_v.shape)

        assert len(self.vec_train_k) == len(self.queries_train)
        assert len(self.vec_test_k) == len(self.queries_test)

        print('vec_k', self.vec_train_k.shape)
        print('vec_v', self.vec_train_v.shape)

        self.batch_size = batch_size
        print("The batch size is %d." % self.batch_size)
        self.num_qa_pairs_train = len(self.vec_train_k)
        self.num_steps = int(self.num_qa_pairs_train/self.batch_size) # drop the tail
        self.current_step = -1 # current_step should be in [0, num_steps-1]
        print("There's %d steps in one epoch" % self.num_steps)
        self.answer_set_size = len(w2i_label)


    def turnToTensorVariable(self, var):
        return Variable(torch.LongTensor(var)).to(self.device)

    def next_chunk_train(self):
        if(self.current_step == (self.num_steps - 1)):
            self.current_step = 0
        else:
            self.current_step += 1
        print(self.current_step)                

        # vec_train_k : (num_qa_pairs_train, max_mem_size, mem_key_len) => (batch_size, max_mem_size, mem_key_len)
        batch_train_k = self.vec_train_k[self.current_step*self.batch_size: (self.current_step+1)*self.batch_size, :, :]
        batch_train_k = self.turnToTensorVariable(batch_train_k)

        # vec_train_v: (num_qa_pairs_train, max_mem_size, mem_val_len) => (batch_size, max_mem_size, mem_val_len)
        batch_train_v = self.vec_train_v[self.current_step*self.batch_size: (self.current_step+1)*self.batch_size, :, :]
        batch_train_v = self.turnToTensorVariable(batch_train_v)

        # queries_train is in shape (batch_size, query_max_len - 1)
        batch_queries_train = self.queries_train[self.current_step*self.batch_size: (self.current_step+1)*self.batch_size, :]
        batch_queries_train = self.turnToTensorVariable(batch_queries_train) 

        # answers_train is in shape (batch_size, len(w2i_label))
        batch_answers_train = self.answers_train[self.current_step*self.batch_size: (self.current_step+1)*self.batch_size]
        batch_answers_train = self.turnToTensorVariable(batch_answers_train) 

        return (batch_train_k, batch_train_v, batch_queries_train, batch_answers_train)

