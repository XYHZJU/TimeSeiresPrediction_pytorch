import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp


class DLinear(nn.Module):
    def __init__(self, configs):
        super(DLinear, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_activation = configs.use_activationc

        # Decompsition Kernel Size
        self.decompsition = series_decomp(configs.kernel_size)
        # self.individual = configs.individual
        # self.channels = configs.enc_in

        # if self.individual:
        #    self.Linear_Seasonal = nn.ModuleList()
        #    self.Linear_Trend = nn.ModuleList()

            # for i in range(self.channels):
                # self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                # self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        # else:
        self.temporallinear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
        self.outlinear_Seasonal = nn.Linear(configs.enc_in, configs.enc_out)
        self.temporallinear_Trend = nn.Linear(self.seq_len,self.pred_len)
        self.outlinear_Trend = nn.Linear(configs.enc_in, configs.enc_out)
            
            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        if self.use_activation and configs.activation == "relu":
            self.activation = F.relu
        elif self.use_activation and configs.activation == "sigmoid":
            self.activation = torch.sigmoid
        elif self.use_activation:
            self.activation = F.gelu

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        # if self.individual:
            # seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            # trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            # for i in range(self.channels):
                # seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                # trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        # else:
        seasonal_output = self.temporallinear_Seasonal(seasonal_init)
        if self.use_activation:
            seasonal_output = self.activation(seasonal_output)
        seasonal_output = self.outlinear_Seasonal(seasonal_output)
        trend_output = self.temporallinear_Trend(trend_init)
        if self.use_activation:
            trend_output = self.activation(trend_output)
        trend_output = self.outlinear_Trend(trend_output)

        x = seasonal_output + trend_output
        return x.permute(0,2,1) # to [Batch, Output length, Channel]