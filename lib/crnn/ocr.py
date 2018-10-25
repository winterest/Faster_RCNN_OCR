import torch.nn as nn
from . import config

import torch
from torch.autograd import Variable

minW = config.minW 
minH = config.minH
scale = config.scale

nRoIFeature = sum(scale) 

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class ocrModel(nn.Module):

    def __init__(self, nc=3, nclass=37, nh=64,  nRoIFeature=nRoIFeature, n_rnn=2,nRin=256, leakyRelu=False):# 1, 37
        super(ocrModel, self).__init__()
        '''
        Revised version of original CNN-RNN model: CNN-RoI-FC-RNN
        
        CNN: b*1*H*W -- b*c*h*w -- (RoI) -- b*c*nFeature*n -- (FC) -- b*c*1*n -- (RNN) -- n*b*class 
        
        '''
        #ind  0, 1, 2, 3, 4, 5, 6

        ks = [3, 3, 3, 2, 2, 3, 4, 2]
        ss = [1, 1, 1, 1, 1, 1, 1, 1]
        ps = [1, 1, 1, 1, 1, 1, 2, 0]
        nm = [64, 128, 256, 256, 256, 256, 256]

        cnn0 = nn.Sequential()
        cnn1 = nn.Sequential()
        cnn2 = nn.Sequential()
        cnn3 = nn.Sequential()        

        def convRelu(net, i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            net.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                net.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                net.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                net.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(cnn0,0) # ch: 1-64
        cnn0.add_module('pooling{0}'.format(0), nn.MaxPool2d((2,2), (1,2), 0))  # 64,h,w/2
        convRelu(cnn0,1) # ch: 64-128
        cnn0.add_module('pooling{0}'.format(1), nn.MaxPool2d((2,2), (1,2), 0))  # 128,h,w/4
        convRelu(cnn0,2, True) # ch: 128-256     256,h,w followed by roi0
        
        convRelu(cnn1,3) # 
        cnn1.add_module('pooling{0}'.format(2), nn.MaxPool2d((2, 2), (2, 1), 0))  # 256,h/2,w/4 followed by roi1
        
        convRelu(cnn2,4) # 
        convRelu(cnn2,5, True)  
        cnn2.add_module('pooling{0}'.format(2), nn.MaxPool2d((2, 2), (2, 1), 0))  # 256,h/4,w/4 followed by roi2

        convRelu(cnn3,6)
        cnn3.add_module('pooling{0}'.format(3), nn.MaxPool2d((2, 2), (2, 1), 0))  # 256,h/8,w/4 followed by roi3
        #convRelu(6, True)  # 512x1x16
        
        self.cnn0 = cnn0
        self.cnn1 = cnn1
        self.cnn2 = cnn2
        self.cnn3 = cnn3
        
        self.fc = nn.Sequential(
            nn.Linear(nRoIFeature, 1),
            )
            
        self.rnn = nn.Sequential(
            BidirectionalLSTM(nRin, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):

        roi = []
        conv = self.cnn0(input)
        b, c, h, w = conv.size()
        h0 = int(max(minH, ((h+scale[0]-1)//scale[0])*scale[0]))
        conv0 = Variable(torch.zeros(b,c,h0,w)).cuda()
        conv0[:,:,0:h,:] = conv
        roi0 = nn.MaxPool2d((h0//scale[0],1))(conv0)        
        
        conv = self.cnn1(conv)
        b, c, h, w = conv.size()
        h0 = int(max(minH, ((h+scale[1]-1)//scale[1])*scale[1]))
        conv0 = Variable(torch.zeros(b,c,h0,w)).cuda()
        conv0[:,:,0:h,:] = conv
        roi1 = nn.MaxPool2d((h0//scale[1],1))(conv0)
        
        conv = self.cnn2(conv)
        b, c, h, w = conv.size()
        h0 = int(max(minH, ((h+scale[2]-1)//scale[2])*scale[2]))
        conv0 = Variable(torch.zeros(b,c,h0,w)).cuda()
        conv0[:,:,0:h,:] = conv
        roi2 = nn.MaxPool2d((h0//scale[2],1))(conv0)
        
        conv = self.cnn3(conv)
        b, c, h, w = conv.size()
        h0 = int(max(minH, ((h+scale[3]-1)//scale[3])*scale[3]))
        conv0 = Variable(torch.zeros(b,c,h0,w)).cuda()
        conv0[:,:,0:h,:] = conv
        roi3 = nn.MaxPool2d((h0//scale[3],1))(conv0)

        roi = torch.cat((roi0,roi1,roi2,roi3),2)    

        #print(roi.size())
        roi = roi.permute(0,1,3,2) #(b,c,n,nRoIFeature = 30)
        conv = self.fc(roi)    #(b,c,n,1)
        conv = conv.squeeze(3)
        conv = conv.permute(2, 0, 1)  # [n, b, c]     

        # rnn features
        conv = self.rnn(conv)
        
        return conv 
