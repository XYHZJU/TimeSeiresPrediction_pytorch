#!/usr/bin/env python
# coding:utf8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from PyEMD import EMD

torch.manual_seed(123456)


class Model(nn.Module):
    """
        Implementation of BLSTM Concatenation for sentiment classification task
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        # self.emb = nn.Embedding(num_embeddings=embeddings.size(0),
        #                         embedding_dim=embeddings.size(1),
        #                         padding_idx=0)
        # self.emb.weight = nn.Parameter(embeddings)

        self.input_dim = configs.enc_in
        #self.input_dim = 1
        self.hidden_dim = configs.d_model
        self.dec_out = configs.c_out
        self.pred_len = configs.pred_len
        self.totrain = configs.Seq2SeqtrainState

        # sen encoder
        self.sen_len = configs.seq_len
        self.encoder = nn.LSTM(input_size=self.input_dim,
                               hidden_size=configs.d_model,
                               num_layers=configs.rnn_layers,
                               dropout=configs.dropout,
                               batch_first=True,
                               bidirectional=True)
        self.decoder = nn.LSTM(input_size=self.input_dim,
                               hidden_size=configs.d_model,
                               num_layers=configs.rnn_layers,
                               dropout=configs.dropout,
                               batch_first=True,
                               bidirectional=True)

        self.outputlinear = nn.Linear(2 * configs.d_model, configs.c_out)

    def bi_fetch(self, rnn_outs, seq_lengths, batch_size, max_len):
        rnn_outs = rnn_outs.view(batch_size, max_len, 2, -1)

        # (batch_size, max_len, 1, -1)
        fw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([0])).cuda())
        fw_out = fw_out.view(batch_size * max_len, -1)
        bw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([1])).cuda())
        bw_out = bw_out.view(batch_size * max_len, -1)

        batch_range = Variable(torch.LongTensor(range(batch_size))).cuda() * max_len
        batch_zeros = Variable(torch.zeros(batch_size).long()).cuda()
        print("fw ",fw_out.shape)
        print("batch range: ",batch_range.shape, batch_zeros.shape)

        fw_index = batch_range + batch_size - 1
        fw_out = torch.index_select(fw_out, 0, fw_index)  # (batch_size, hid)

        bw_index = batch_range + batch_zeros
        bw_out = torch.index_select(bw_out, 0, bw_index)
        print("cat ",fw_out.shape, bw_out.shape)
        outs = torch.cat([fw_out, bw_out], dim=1)
        return outs

    def emd_decompose(self,input):
        emd = EMD()
        input = input.cpu().numpy()
        print(input.shape)
        
        for i in range(input.shape[0]):
            print(input[i,:,:].squeeze(1).shape)
            imfs = emd.emd(input[i,:,:].squeeze(1),max_imf=10)
            print("shape: ",imfs.shape)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        B, L, E = x_enc.shape
        x_enc = x_enc[:,:,-1:]
        x_dec = x_dec[:,:,-1:]

        self.emd_decompose(x_enc)

        outputs = torch.zeros(B, self.pred_len, self.dec_out).to(x_enc.device)
        _, hidden = self.encoder(x_enc)

        #print("x_enc: ",x_enc.shape)
        x = x_enc[:, -1, -self.dec_out:]
        # x = x_enc[:, -1, -1:]
        #print("x: ",x.shape)
        if self.totrain: # teacher force training
            for t in range(self.pred_len):
                x = x.unsqueeze(1)
                #print("x1: ",x.shape)
                output, hidden = self.decoder(x, hidden)
                output = self.outputlinear(output.squeeze(1))
                outputs[:,t] = output
                x = x_dec[:,t,-self.dec_out:]
        else: # dynamic decoding
            for t in range(self.pred_len):
                x = x.unsqueeze(1)
                output, hidden = self.decoder(x, hidden)
                output = self.outputlinear(output.squeeze(1))
                outputs[:,t] = output
                x = output
        return outputs
