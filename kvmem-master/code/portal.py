import argparse
from dataloader import DataLoader
from trainer import Trainer
import torch
import pdb

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-m', '--model', help='begin training from saved keras model')
    parser.add_argument('--max_mem_size',
                        default=100,
                        type=int,
                        help='max the number of memories related one episode')
    parser.add_argument('--embedding_size',
                        default=500,
                        type=int,
                        help='embedding dimension')
    parser.add_argument('--n_epochs',
                        default=30,
                        type=int,
                        help='number of epoch')
    parser.add_argument('--batch_size',
                        default=64,
                        type=int,
                        help='batch size when training')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloader = DataLoader(args.max_mem_size, args.batch_size, device)
    pdb.set_trace()
    query_max_length = dataloader.query_max_len 
    trainer = Trainer(args.embedding_size, args.max_mem_size, query_max_length, args.batch_size, args.n_epochs, device)
    trainer.train_model(dataloader)
