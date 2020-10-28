import argparse
import os
import shutil
import time
import random
import numpy as np
import math
import sys
sys.path.append('../..')
from utils import *
from NeuralNet import NeuralNet

#import argparse

from .OthelloNNet import OthelloNNet as onnet
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': True,
    'num_channels': 256,
#    'num_channels': 448,
#    'num_channels': 512,
})

class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = onnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

#        stack = inspect.stack()
#        the_class = stack[1][0].f_locals["self"].__class__.__name__
#        the_method = stack[1][0].f_code.co_name
#
#        print("I was called by {}.{}()".format(the_class, the_method))


    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs, input_komi = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        input_komi = np.asarray(input_komi)
        self.nnet.model.fit(x = [input_boards, input_komi], y = [target_pis, target_vs], batch_size = args.batch_size, epochs = args.epochs)
        
        # Attempt to plot performance over time
        #history = self.nnet.model.fit(x = input_boards, y = [target_pis, target_vs], batch_size = args.batch_size, epochs = args.epochs)
        #return history

    def predict(self, board, komi):
        """
        board: np array with board
        """
        # timing
        start = time.time()
        
        # preparing input
        board = board[np.newaxis, :, :]
        #komi = komi[np.newaxis, :]
        nn_input = [board, komi]
        
        # run
        pi, v, alpha = self.nnet.model.predict(nn_input)
        #print(v)

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi[0], v[0]

    def predict_alpha(self, board, komi):
        board = board[np.newaxis, :, :]
        nn_input = [board, komi]
        alpha = self.nnet.model_alpha.predict(nn_input)
        return alpha[0]
        
    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        self.nnet.model.load_weights(filepath)
