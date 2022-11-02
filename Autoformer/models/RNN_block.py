import torch
import torch.nn as nn
import torch.nn.functional as F


class Seq2SeqRNN(nn.Module):
    def __init__(self, configs):
        super(Seq2SeqRNN, self).__init__()
        self.train = configs.Seq2SeqtrainState
        self.encoder = nn.GRU(configs.enc_in, configs.d_model, configs.num_layer, batch_first=True, dropout=configs.dropout)
        self.decoder = nn.GRU(configs.enc_out, configs.d_model, configs.num_layer, batch_first=True, dropout=configs.dropout)
        self.outputlinear = nn.Linear(configs.d_model, configs.enc_out)
        self.target_idx_loc = configs.target_idx_loc

    def forward(self, source, target):
        B, S, E = target.shape
        outputs = torch.zeros(B, S, E).to(source.device)
        _, hidden = self.encoder(source)
        
        x = source[:, -1, self.target_idx_loc]
        if self.train: # teacher force training
            for t in range(S):
                x = x.unsqueeze(1)
                output, hidden = self.decoder(x, hidden)
                output = self.outputlinear(output.squeeze(1))
                outputs[t] = output
                x = target[t]
        else: # dynamic decoding
            for t in range(S):
                x = x.unsqueeze(1)
                output, hidden = self.decoder(x, hidden)
                output = self.outputlinear(output.squeeze(1))
                outputs[t] = output
                x = output
        return outputs