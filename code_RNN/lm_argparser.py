# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 19:51:26 2024

@author: aikan
"""

#######################################################
# Settings and hyper-parameteres of the Lanuage Model #
# #####################################################

import argparse

lm_parser = argparse.ArgumentParser(add_help=False)

lm_parser.add_argument('--path2data', type=str,
                       help='location of the data corpus', default='../data/cleaned_reports')

lm_parser.add_argument('--model', type=str, default='LSTM',
                       help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
lm_parser.add_argument('--emsize', type=int, default=200,
                       help='size of word embeddings')
lm_parser.add_argument('--nhid', type=int, default=200,
                       help='number of hidden units per layer')
lm_parser.add_argument('--nlayers', type=int, default=2,
                       help='number of layers')
lm_parser.add_argument('--dropout', type=float, default=0.2,
                       help='dropout applied to layers (0 = no dropout)')
lm_parser.add_argument('--tied', action='store_true',
                       help='tie the word embedding and softmax weights')

lm_parser.add_argument('--lr', type=float, default=20,
                       help='initial learning rate')
lm_parser.add_argument('--clip', type=float, default=0.25,
                       help='gradient clipping')
lm_parser.add_argument('--epochs', type=int, default=10,
                       help='upper epoch limit')
lm_parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                       help='batch size')
lm_parser.add_argument('--eval-batch-size', type=int, default=10, 
                       help='evaluation batch size')

lm_parser.add_argument('--bptt', type=int, default=35,
                       help='sequence length')


lm_parser.add_argument('--seed', type=int, default=1111,
                       help='random seed')
lm_parser.add_argument('--cuda', action='store_true',
                       help='use CUDA')
lm_parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                       help='report interval')
lm_parser.add_argument('--save', type=str, default='../output/model.pt',
                       help='path to save the final model')
lm_parser.add_argument('--log', type=str, default='log.txt',
                       help='path to logging file')
