# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 19:55:35 2024

@author: aikan
"""

import torch.nn as nn
import torch.utils.data.dataloader
from utils import get_batch, repackage_hidden

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder.
        ntoken: vocab size
        nip: embedding size
    """
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(num_embeddings=ntoken, embedding_dim=ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(in_features=nhid, out_features=ntoken)

        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        #print(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (weight.new(self.nlayers, bsz, self.nhid).zero_(),
                    weight.new(self.nlayers, bsz, self.nhid).zero_())
        else:
            return weight.new(self.nlayers, bsz, self.nhid).zero_()


def eval_model(model, data_source, criterion, ntokens, eval_batch_size, bptt):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(eval_batch_size)

    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i, bptt)
            #> output has size seq_length x batch_size x vocab_size
            output, hidden = model(data, hidden)
            #> output_flat has size num_targets x vocab_size (batches are stacked together)
            #> ! important, otherwise softmax computation (e.g. with F.softmax()) is incorrect
            output_flat = output.view(-1, ntokens)
            #output_candidates_info(output_flat.data, targets.data)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / (len(data_source) - 1)