import torch
import torch.nn as nn
import pdb

class KVMemNN(nn.Module):
    #vocab_size:1024,mem_size:100,query_max_length:18,answer_set_size:200,n_embd:500
    def __init__(self, vocab_size, mem_size, query_max_length, answer_set_size, n_embd):
        super(KVMemNN, self).__init__()
        self.vocab_size = vocab_size
        self.answer_set_size = answer_set_size
        self.n_embd = n_embd
        self.embedding_a = nn.Embedding(self.vocab_size, self.n_embd)
        self.mem_size = mem_size
        self.query_max_length = query_max_length
        self.batchnorm_kv = nn.BatchNorm2d(num_features = self.mem_size) #TODO, why mem_size?

        self.batchnorm_query = nn.BatchNorm1d(num_features = query_max_length)
        self.batchnorm_query_updated = nn.BatchNorm1d(num_features = self.n_embd)

        self.softmax = nn.Softmax(dim=1)
        self.R = nn.Linear(self.n_embd, self.n_embd)

        self.lastDense = nn.Linear(self.n_embd, self.answer_set_size)
        self.lastSoftmax = nn.LogSoftmax(dim = -1)

    def forward(self, input):
        key, value, query, answers = input

        key_encoded = self.embedding_a(key)
        key_encoded = self.batchnorm_kv(key_encoded)
        # turn (batch_size, mem_size, mem_len, embd_size) to (batch_size, mem_size, embd_size)
        summed_key_encoded = key_encoded.sum(dim=(2))
        
        val_encoded = self.embedding_a(value)
        val_encoded = self.batchnorm_kv(val_encoded)
        # turn (batch_size, mem_size, mem_len, embd_size) to (batch_size, mem_size, embd_size)
        summed_val_encoded = val_encoded.sum(dim=(2)) 

        query_encoded = self.embedding_a(query) # (batch_size, query_max_length, embd_size)
        query_encoded = self.batchnorm_query(query_encoded) #BatchNorm1d
        # turn (batch_size, query_max_length, embd_size) to (batch_size, embd_size)
        summed_query_encoded = query_encoded.sum(dim=(1))

        # do the attention, summed_key_encoded is in shape (batch_size, mem_size, embd_size),
        # and summed_query_encoded is in shape (batch_size, embd_size)
        ph = torch.matmul(summed_key_encoded, summed_query_encoded.unsqueeze(-1)) # (batch_size, mem_size, 1)
        # ph is in shape (batch_size, mem_size, 1)
        ph = self.softmax(ph)

        # ph is in shape (batch_size, mem_size, 1) and summed_val_encoded is in shape (batch_size, mem_size, embd_size)
        # , so o is in shape (batch_size, embd_size)
        o = torch.matmul(torch.transpose(ph, 1, 2), summed_val_encoded).squeeze(1)

        updated_q = self.R(summed_query_encoded + o)
        updated_q = self.batchnorm_query_updated(updated_q)

        # do the analysis on answer
        logits = self.lastDense(updated_q)
        pred = self.lastSoftmax(logits)
        return pred
