# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 19:55:52 2024

@author: aikan
"""

import argparse
import math
import time

import torch
import torch.nn as nn

from lm_argparser import lm_parser

from data import Data
from model import RNNModel, eval_model
from utils import repackage_hidden, get_batch

parser = argparse.ArgumentParser(parents=[lm_parser], description="Basic training and evaluation for RNN LM")

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

# LOAD DATA
data = Data(path2data=args.path2data, batch_size=args.batch_size, eval_batch_size=args.eval_batch_size, cuda=args.cuda)

# DEFINE MODEL
criterion = nn.CrossEntropyLoss()
model = RNNModel(rnn_type=args.model, ntoken=data.ntokens, ninp=args.nhid, nhid=args.nhid, nlayers=args.nlayers, dropout=0.5, tie_weights=False)
if args.cuda:
    model.cuda()

###############
# TRAIN MODEL #
###############
lr = args.lr
best_val_loss = None
for epoch in range(1, args.epochs + 1):
    epoch_start_time = time.time()

    model.train()
    total_loss = 0
    start_time = time.time()

    hidden = model.init_hidden(args.batch_size)

    for batch, i in enumerate(range(0, data.train_data.size(0) - 1, args.bptt)):
        data_batch, targets = get_batch(data.train_data, i, args.bptt)
        # truncated BPP
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data_batch, hidden)

        loss = criterion(output.view(-1, data.ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print(f'| epoch {epoch} | {batch}/{len(data.train_data)} batches | lr {lr} | loss {cur_loss} | ppl {math.exp(cur_loss)}')
            total_loss = 0
            start_time = time.time()
    # END OF EPOCH

    val_loss = eval_model(model, data.val_data, criterion, data.ntokens, args.eval_batch_size, args.bptt)
    # Save the model if the validation loss is the best we've seen so far.
    if not best_val_loss or val_loss < best_val_loss:
        with open(args.save, 'wb') as f:
            torch.save(model, f)
        best_val_loss = val_loss
    else:
        # Anneal the learning rate if no improvement has been seen in the validation dataset.
        lr /= 4.0


# EVAL on test data.
test_loss = eval_model(model, data.test_data, criterion, data.ntokens, args.eval_batch_size, args.bptt)
print(f'Test loss: {math.exp(test_loss)}')
