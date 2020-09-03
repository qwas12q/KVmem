import torch
import torch.nn as nn
from model import KVMemNN
import pdb
import time
from torch.autograd import Variable
from utils import Utils

class Trainer():
    def __init__(self, embedding_size, mem_size, query_max_length, batch_size, n_epochs, device):
        self.device = device 
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = 1e-3
        self.embd_size = embedding_size
        self.mem_size = mem_size
        self.query_max_length = query_max_length

    def __create_model(self, vocab_size, answer_set_size):
        model = KVMemNN(vocab_size, self.mem_size, self.query_max_length, answer_set_size, self.embd_size)
        return model
    def train_within_step(self, model, optimizer, criterion, inp, target):
        model.zero_grad()
        pred = model(inp)
        loss = criterion(pred, target)
        #pdb.set_trace()

        loss.backward()
        optimizer.step()
        
        return loss.data.item() 


    def train_model(self, dataloader):
        vocab_size = dataloader.vocab_size
        answer_set_size = dataloader.answer_set_size
        model = self.__create_model(vocab_size, answer_set_size)

        optimizer = torch.optim.Adam(model.parameters(), lr = self.learning_rate, weight_decay = 1e-5)
        criterion = nn.NLLLoss()

        start = time.time()
        for epoch in range(self.n_epochs):
            print("epoch %d" % epoch)
            for step in range(0, dataloader.num_steps):
                # batch_train_k, batch_train_v, batch_queries_train, batch_answers_train = input
                input = dataloader.next_chunk_train()
                target = input[-1]
                loss = self.train_within_step(model, optimizer, criterion, input, target)
                print('[%s (%d %d%%) %.4f]' % (Utils.time_since(start), step, step / dataloader.num_steps * 100, loss))

if __name__=="__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(embedding_size=500, mem_size=100, query_max_length=18, batch_size=64, n_epochs=10, device = device)
    trainer.create_model(vocab_size=1024, answer_set_size = 200)
