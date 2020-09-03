import argparse
from dataloader import DataLoader
import pdb
import itertools
import functools
import numpy as np 
from sklearn.model_selection import train_test_split
from model import KVMemNN
import torch
import torch.nn as nn
from torch.autograd import Variable

def turnToTensorVariable(var, device):
    return Variable(torch.LongTensor(var)).to(device)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_dir',
                    default="data/tasks_1-20_v1-2/en/",
                    type=str,
                    help='Directory containing bAbI tasks')
    parser.add_argument('--memory_size',
                    default="20",
                    type=int,
                    help='Maximum size of memory.')
    parser.add_argument('--batch_size',
                    default="32",
                    type=int,
                    help='Batch size for training.')
    parser.add_argument('--embedding_size',
                    default="30",
                    type=int,
                    help='Embedding size for embedding matrices.')
    parser.add_argument('--hidden_size',
                    default="40",
                    type=int,
                    help='Hidden size, a.k.a., Feature size.')
    parser.add_argument('--hops',
                    default="3",
                    type=int,
                    help='How many hops to pay attention on key-value pairs')
    args = parser.parse_args()
    print(args.data_dir)
    dataloader = DataLoader()

    # training is a list of samples, training[0] = 
    # ([['mary', 'moved', 'to', 'the', 'bathroom'], ['john', 'went', 'to', 'the', 'hallway']], ['where', 'is', 'mary'], ['bathroom'])
    training, testing = dataloader.load_task(args.data_dir, 1)
    data = training + testing
    vocab = sorted(functools.reduce(lambda x, y: x | y, (set(list(itertools.chain.from_iterable(s)) + q + a) for s, q, a in data)))
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab)) 
    max_story_size = max(map(len, (s for s, _, _ in data)))
    mean_story_size = int(np.mean(list(map(len, (s for s, _, _ in data)))))
    sentence_size = max(map(len, itertools.chain.from_iterable(s for s, _, _ in data)))
    query_size = max(map(len, (q for _, q, _ in data)))
    memory_size = min(args.memory_size, max_story_size)
    vocab_size = len(word_idx) + 1 # +1 for nil word
    sentence_size = max(query_size, sentence_size) # for the position
    print("Longest sentence length", sentence_size)
    print("Longest story length", max_story_size)
    print("Average story length", mean_story_size)

    # train/validation/test sets
    S, Q, A = dataloader.vectorize_data(training, word_idx, sentence_size, memory_size)
    trainS, valS, trainQ, valQ, trainA, valA = train_test_split(S, Q, A, test_size=.1)
    testS, testQ, testA = dataloader.vectorize_data(testing, word_idx, sentence_size, memory_size)

    print("Training set shape", trainS.shape)

    # params
    n_train = trainS.shape[0]
    n_test = testS.shape[0]
    n_val = valS.shape[0]

    print("Training Size", n_train)
    print("Validation Size", n_val)
    print("Testing Size", n_test)

    # trainA is in the shape (n_train, vocab_size)
    # train_labels is in the shape (n_train,)
    train_labels = np.argmax(trainA, axis=1)
    test_labels = np.argmax(testA, axis=1)
    val_labels = np.argmax(valA, axis=1)

    batch_size = args.batch_size
    batches = list(zip(range(0, n_train - batch_size, batch_size), range(batch_size, n_train, batch_size)))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = KVMemNN(vocab_size, memory_size, sentence_size, vocab_size, args.embedding_size, args.hidden_size, args.hops, device)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    all_answers = turnToTensorVariable(np.arange(0, vocab_size), device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-5)
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()

    for epoch in range(1000):
        np.random.shuffle(batches)            
        cnt = 0
        total_loss = 0.0
        total_acc = 0.0
        for start, end in batches: 
            model.zero_grad()
            # story is in the shape (batch_size, memory_size, sentence_size) 
            s = turnToTensorVariable(trainS[start:end], device)
            q = turnToTensorVariable(trainQ[start:end], device)
            target = turnToTensorVariable(train_labels[start:end], device)
            pred = model((s, q, all_answers))
            loss = criterion(pred, target)
            total_loss += loss.data.item()
            cnt+=1
            loss.backward()
            optimizer.step()

            model.zero_grad()
            test_s = turnToTensorVariable(testS[start:end], device)
            test_q = turnToTensorVariable(testQ[start:end], device)
            test_target = turnToTensorVariable(test_labels[start:end], device)
            pred_test = model((test_s, test_q, all_answers))
            pred_test_labels= torch.argmax(pred_test, dim=1)
            acc_on_testing = torch.sum(torch.eq(pred_test_labels, test_target)).data.item() / batch_size
            total_acc += acc_on_testing

        print("epoch=%d avg NLL loss = %s, avg acc on testing=%s" % (epoch, total_loss/cnt, total_acc/cnt))
