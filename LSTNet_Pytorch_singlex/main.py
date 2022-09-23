import argparse
import math
import time

from models import LSTNet
import torch.nn as nn

from Optim import Optim
from utils import *
from datetime import date
import time

import matplotlib.pyplot as plt

# plt.cla()
fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size,flag=False):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None
    predict_ = None
    test_ = None

    for X, Y in data.get_batches(X, Y, batch_size, False):
        output = model(X)
        if predict is None:
            scale = data.scale.expand(output.size(0), data.m)
            predict = output
            test = Y
            predict_ = output*scale
            test_ = Y*scale
        else:
            scale = data.scale.expand(output.size(0), data.m)
            predict_ = torch.cat((predict_, output*scale))
            test_ = torch.cat((test_, Y*scale))
            predict = torch.cat((predict,output))
            test = torch.cat((test, Y))

        scale = data.scale.expand(output.size(0), data.m)
        # print("size:",(output*scale).shape)
        total_loss += evaluateL2((output * scale)[:,-3:], (Y * scale)[:,-3:]).item()
        total_loss_l1 += evaluateL1(output * scale, Y * scale).item()
        n_samples += (output.size(0) * data.m)
    rse = math.sqrt(total_loss / n_samples)
    rae = (total_loss_l1 / n_samples) / data.rae

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    predict_ = predict_.data.cpu().numpy()
    Ytest_ = test_.data.cpu().numpy()
    sigma_p = predict.std(axis=0)
    sigma_g = Ytest.std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()

    if(flag):
        # scale = data.scale.expand(output.size(0), data.m)
        # predict_ = predict*scale
        # Ytest_ = Ytest*scale
        predict1_ = predict[:,-1]
        Ytest1_ = Ytest[:,-1]

        predict2_ = predict[:,-2]
        Ytest2_ = Ytest[:,-2]

        predict1_local = predict1_[-5:]
        Ytest1_local = Ytest1_[-5:]
        print("predict_!!!!!:",predict1_.shape)
        # print("predict_:",predict_.shape,predict_[:,-1].shape)
        # plt.clf()
        ax1.cla()
        ax2.cla()
        # plt.subplot(2,2,1)
        ax1.plot(range(len(predict1_)),predict1_,label='predict1')
        ax1.plot(range(len(Ytest1_)),Ytest1_,label = 'true1')
        ax1.legend()
        # plt.subplot(2,2,2)
        ax2.plot(range(len(predict1_local)),predict1_local,label='predict1_local')
        ax2.plot(range(len(Ytest1_local)),Ytest1_local,label = 'true1_local')
        ax2.legend()
        today = date.today()
        timenow = time.strftime("%Y-%m-%d-%H", time.localtime())
        plt.savefig('save/fig/test'+timenow+'.jpg')
        plt.pause(0.1)
        # plt.draw()

    return rse, rae, correlation


def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()
        output = model(X)
        scale = data.scale.expand(output.size(0), data.m)
        loss = criterion((output * scale)[:,-3:], (Y * scale)[:,-3:])
        loss.backward()
        optim.step()
        total_loss += loss.item()
        n_samples += (output.size(0) * data.m)
    return total_loss / n_samples


parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, default='data/exchange_rate.txt',
                    help='location of the data file')  # required=True,
parser.add_argument('--model', type=str, default='LSTNet',
                    help='')
parser.add_argument('--hidCNN', type=int, default=50,
                    help='number of CNN hidden units')
parser.add_argument('--hidRNN', type=int, default=50,
                    help='number of RNN hidden units')
parser.add_argument('--window', type=int, default=36,
                    help='window size')
parser.add_argument('--CNN_kernel', type=int, default=6,
                    help='the kernel size of the CNN layers')
parser.add_argument('--highway_window', type=int, default=36,
                    help='The window size of the highway component')
parser.add_argument('--clip', type=float, default=10.,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=54321,
                    help='random seed')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='models/exchange_rate.pt',
                    help='path to save the final model')
parser.add_argument('--cuda', type=str, default=False)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.0009)
parser.add_argument('--horizon', type=int, default=5)
parser.add_argument('--skip', type=float, default=1)
parser.add_argument('--hidSkip', type=int, default=5)
parser.add_argument('--L1Loss', type=bool, default=False)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--output_fun', type=str, default='None')
args = parser.parse_args()

args.cuda = args.gpu is not None
if args.cuda:
    torch.cuda.set_device(args.gpu)

""" Set the random seed manually for reproducibility. """
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

Data = Data_Utility(args.data, 0.6, 0.2, args.cuda, args.horizon, args.window, args.normalize)
print(Data.rse)

model = eval(args.model).Model(args, Data)

if args.cuda:
    model.cuda()

nParams = sum([p.nelement() for p in model.parameters()])
print('* number of parameters: %d' % nParams)

if args.L1Loss:
    criterion = nn.L1Loss(reduction='sum')
else:
    criterion = nn.MSELoss(reduction='sum')
evaluateL2 = nn.MSELoss(reduction='sum')
evaluateL1 = nn.L1Loss(reduction='sum')
if args.cuda:
    criterion = criterion.cuda()
    evaluateL1 = evaluateL1.cuda()
    evaluateL2 = evaluateL2.cuda()

best_val = 10000000
optim = Optim(model.parameters(), args.optim, args.lr, args.clip, )

# At any point you can hit Ctrl + C to break out of training early.
try:
    print('Start training....')
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)
        val_loss, val_rae, val_corr = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                                               args.batch_size)
        print(
            '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.format(
                epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr))
        # Save the model if the validation loss is the best we've seen so far.

        if val_loss < best_val:
            timenow = time.strftime("%Y-%m-%d-%H", time.localtime())
            with open(args.save+timenow+'.pt', 'wb') as f:
                torch.save(model, f)
            best_val = val_loss
        if epoch % 5 == 0:
            test_acc, test_rae, test_corr = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
                                                     args.batch_size,True)
            print("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save+timenow+'.pt', 'rb') as f:
    model = torch.load(f)
test_acc, test_rae, test_corr = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
                                         args.batch_size,True)
print("test_loss {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))