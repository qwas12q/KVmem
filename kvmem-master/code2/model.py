import torch
import torch.nn as nn
import pdb
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import math

class KeyEncoder(nn.Module):
    def __init__(self, vocab_size, n_embd, pe, A, embedding):
        super(KeyEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        #self.embedding = nn.Embedding(self.vocab_size, self.n_embd, padding_idx=0)
        self.embedding = embedding 
        self.position_encoding = pe
        self.A = A

    def forward(self, key):
        # embedded is in the shape (batch_size, mem_size, sentence_size, n_embd)
        embedded = self.embedding(key)

        # torch.mul is the element-wise product
        # embedded_with_position is also in the shape (batch_size, mem_size, sentence_size, n_embd),
        # and the self.position_encoding (sentence_size, n_embd) is broadcasted.
        embedded_with_position = torch.mul(embedded, self.position_encoding)

        # summed_embedded_with_position is in the shape (batch_size, mem_size, n_embd)
        summed_embedded_with_position = torch.sum(embedded_with_position, dim = 2)
        return self.A(summed_embedded_with_position)


class ValueEncoder(nn.Module):
    def __init__(self, vocab_size, n_embd, pe, A, embedding):
        super(ValueEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        #self.embedding = nn.Embedding(self.vocab_size, self.n_embd, padding_idx = 0)
        self.embedding = embedding 
        self.position_encoding = pe
        self.A = A

    def forward(self, value):
        embedded = self.embedding(value)
        embedded_with_position = torch.mul(embedded, self.position_encoding)
        summed_embedded_with_position = torch.sum(embedded_with_position, dim = 2)
        return self.A(summed_embedded_with_position) 



class QuestionEncoder(nn.Module):
    def __init__(self, vocab_size, n_embd, pe, A, embedding):
        super(QuestionEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        #self.embedding = nn.Embedding(self.vocab_size, self.n_embd, padding_idx = 0)
        self.embedding = embedding 
        self.position_encoding = pe
        self.A = A

    def forward(self, question):
        # embedded is in the shape (batch_size, sentence_size, n_embd)
        embedded = self.embedding(question)
        # embedded_with_position is in the shape (batch_size, sentence_size, n_embd)
        embedded_with_position = torch.mul(embedded, self.position_encoding)
        # summed_embedded_with_position is in the shape (batch_size, n_embd)
        summed_embedded_with_position = torch.sum(embedded_with_position, dim = 1)
        return self.A(summed_embedded_with_position) 


class AnswerEncoder(nn.Module):
    def __init__(self, answer_set_size, n_embd, pe, B):
        super(AnswerEncoder, self).__init__()
        self.answer_set_size = answer_set_size 
        self.n_embd = n_embd
        self.embedding = nn.Embedding(self.answer_set_size, self.n_embd)
        self.position_encoding = pe
        self.B = B 

    def forward(self, all_answers):
        if(len(all_answers.size()) == 1):
            # in case that the answer is in the shape (answer_set_size)
            # embedded is in the shape (answer_set_size, n_embd)
            embedded = self.embedding(all_answers)

            # the returned variable's shape is (answer_set_size, hidden_size)
            return self.B(embedded) 
        else:
            raise Exception("The answer's shape is not expected")


class AttentionLayer(nn.Module):
    def __init__(self, hops, hidden_size):
        super(AttentionLayer, self).__init__()

        self.hidden_size = hidden_size
        self.hops = hops
        self.Rs= nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size, bias=False) for i in range(self.hops)])

    def forward(self, encoded_key, encoded_value, encoded_question, mask):
        # encoded_question is in the shape (batch_size, hidden_size), and encoded_key is 
        # in the shape (batch_size, mem_size, hidden_size)
        batch_size = encoded_key.size(0)
        query = encoded_question.view(batch_size, 1, self.hidden_size)
        key_t = encoded_key.transpose(-2, -1)

        # scores is in shape of (batch_size, 1, mem_size)
        mask = mask.unsqueeze(1)
        scores = torch.matmul(query, key_t) #/ math.sqrt(self.hidden_size), overidding on hidden_size**0.5 didn't spped up
        masked_scores = scores.masked_fill(mask, -1e9)
        attention_scores = F.softmax(masked_scores, dim=-1)

        # (bath_size, 1, mem_size)*(batch_size, mem_size, hidden_size) 
        # so o is in the shape (batch_size, 1, hidden_size)
        o = torch.matmul(attention_scores, encoded_value) 

        for i in range(self.hops):
            # query and o are both in the shape (batch_size, 1, hidden_size).
            # and R maps hidden_size to hidden_size,
            # so the updated query is still in the shape (batch_size, 1, hidden_size) 
            query = self.Rs[i](query+o)
            scores = torch.matmul(query, key_t) #/ math.sqrt(self.hidden_size), overidding on hidden_size**0.5 didn't spped up 
            masked_scores = scores.masked_fill(mask, -1e9)
            attention_scores = F.softmax(masked_scores, dim = -1)
            o = torch.matmul(attention_scores, encoded_value)

        # return the final updated query
        return query


class PredictionLayer(nn.Module):
    def __init__(self):
        super(PredictionLayer, self).__init__()
        self.lastDense = nn.Linear(40, 20)

    def forward(self, updated_query, encoded_all_answers):
        # the logits' shape is (batch_size, answer_set_size)
        #logits = torch.matmul(updated_query.squeeze(1), encoded_all_answers.transpose(0,1))
        #pdb.set_trace()
        #probs = F.softmax(logits, dim=-1)
        #return probs

        return F.log_softmax(self.lastDense(updated_query.squeeze(1)), dim=-1)


class KVMemNN(nn.Module):
    def __init__(self, vocab_size, mem_size, sentence_size, answer_set_size, n_embd, hidden_size, hops, device):
        super(KVMemNN, self).__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.mem_size = mem_size # the max number of sentences in stories
        self.sentence_size = sentence_size # the longest length of sentences in the stories and questiones
        self.answer_set_size = answer_set_size 
        self.hidden_size = hidden_size
        self.hops = hops

        self.A = nn.Linear(n_embd, hidden_size, bias = False)
        self.B = nn.Linear(n_embd, hidden_size, bias = False) # TODO

        self.embedding = nn.Embedding(self.vocab_size, self.n_embd, padding_idx=0)
        self.device = device

        self.pe = self.positionEncoding(self.sentence_size, self.n_embd)
        self.keyEncoder = KeyEncoder(self.vocab_size, self.n_embd, self.pe, self.A, self.embedding)
        self.valueEncoder = ValueEncoder(self.vocab_size, self.n_embd, self.pe, self.A, self.embedding)
        self.questionEncoder = QuestionEncoder(self.vocab_size, self.n_embd, self.pe, self.A, self.embedding)
        self.attentionLayer = AttentionLayer(self.hops, self.hidden_size)
        self.answerEncoder = AnswerEncoder(self.answer_set_size, self.n_embd, self.pe, self.B) 
        self.predictionLayer = PredictionLayer()

    def forward(self, input):
        #story is in the shape (batch_size, mem_size, sentence_size)
        story, question, all_answers = input 
        key = story
        value = story

        mask = (key[:,:,0]==0) # 1 means the memory slot is empty and should be masked

        # the encoded key's shape is (batch_size, mem_size, sentence_size, n_embd).
        # Empty slots are filled with 0 as there are fewer sentences in some stories than self.mem_size.
        encoded_key = self.keyEncoder(key)
        encoded_value = self.valueEncoder(value)
        encoded_quesiton = self.questionEncoder(question)

        # The updated_query is in the shape (batch_size, 1, hidden_size).
        updated_query = self.attentionLayer(encoded_key, encoded_value, encoded_quesiton, mask)
        # the encoded_all_answers is in the shape (self.answer_set_size, hidden_size) 
        encoded_all_answers = self.answerEncoder(all_answers)
        
        probs = self.predictionLayer(updated_query, encoded_all_answers) 
        return probs 


    def positionEncoding(self, sentence_size, n_embd):
        '''
        '''
        encoding = np.ones((sentence_size, n_embd), dtype = np.float32)
        for k in range(1, n_embd + 1):
            for j in range(1, sentence_size + 1):
                encoding[j-1, k-1] = (1 - j/sentence_size) - (k/n_embd) * (1 - 2 * j / sentence_size)
        return Variable(torch.FloatTensor(encoding)).to(self.device)
