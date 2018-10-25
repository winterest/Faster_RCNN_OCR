import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

class angleModel(nn.Module):
    def __init__(self, nc=3, leakyRelu=False):
        super(angleModel, self).__init__()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(nc, 16, kernel_size=7, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            #nn.MaxPool2d(1, stride=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            #nn.MaxPool2d(1, stride=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            #nn.MaxPool2d(1, stride=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            #nn.MaxPool2d(1, stride=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            #nn.MaxPool2d(1, stride=1),
            nn.ReLU(True)
        )
        
        # Regressor for the 3 * 2 affine matrix !! just one angle
        self.fc_loc = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(256 * 6 * 6, 256),
            nn.ReLU(True),            
            nn.Dropout(p=0.25),
            nn.Linear(256, 1)
        )

    # Spatial transformer network forward function
    def stn(self, x):
        conv = self.localization(x)
        #print(conv.size())
        #b, c, h, w = conv.size()  # b, 16, 2, 10
        conv_loc = conv.view(-1,256*6 *6)
        ang = self.fc_loc(conv_loc)
        return ang 
    def forward(self, input):
        conv = self.stn(input)
        return conv
