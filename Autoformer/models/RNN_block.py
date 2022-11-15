import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.totrain = configs.Seq2SeqtrainState
        self.pred_len = configs.pred_len
        self.dec_out = configs.c_out
        self.encoder = nn.GRU(configs.enc_in, configs.d_model, configs.rnn_layers, batch_first=True, dropout=configs.dropout)
        self.decoder = nn.GRU(configs.c_out, configs.d_model, configs.rnn_layers, batch_first=True, dropout=configs.dropout)
        self.outputlinear = nn.Linear(configs.d_model, configs.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        B, L, E = x_enc.shape
        outputs = torch.zeros(B, self.pred_len, self.dec_out).to(x_enc.device)
        _, hidden = self.encoder(x_enc)

        
        x = x_enc[:, -1, -self.dec_out:]
        if self.totrain: # teacher force training
            for t in range(self.pred_len):
                x = x.unsqueeze(1)
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