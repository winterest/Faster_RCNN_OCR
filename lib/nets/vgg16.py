# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nets.network import Network
from model.config import cfg

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models

class vgg16(Network):
  def __init__(self):
    Network.__init__(self)
    self._feat_stride = [16, ]
    self._feat_compress = [1. / float(self._feat_stride[0]), ]
    self._net_conv_channels = 514
    #self._net_conv_channels = 512
    self._fc7_channels = 4096

  def _init_head_tail(self):
    self.vgg = models.vgg16()
    # Remove the last fc layer of VGG ( of classifier part)
    self.vgg.classifier = nn.Sequential(nn.Linear(25186,4096),
                    *list(self.vgg.classifier._modules.values())[1:-1])
    #self.vgg.classifier[0].in_features = 25186
    # Fix the layers before conv3:
    for layer in range(10):
      for p in self.vgg.features[layer].parameters(): p.requires_grad = False

    # not using the last maxpool layer
    self._layers['head'] = nn.Sequential(*list(self.vgg.features._modules.values())[:-1])

  def _image_to_head(self):
    net_conv = self._layers['head'](self._image)
    '''
    '''
    W,H = net_conv.size(2), net_conv.size(3)
    coord_tensor = torch.FloatTensor(1,2,W,H)
    coord_tensor = coord_tensor.cuda()
    coord_tensor = Variable(coord_tensor)
    for w in range(W):
      for h in range(H):
        coord_tensor[:,0,w,h] = w/W
        coord_tensor[:,1,w,h] = h/H
    net_conv = torch.cat((net_conv,coord_tensor),1)
    '''
    '''
    self._act_summaries['conv'] = net_conv
    
    return net_conv

  def _head_to_tail(self, pool5):
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.vgg.classifier(pool5_flat)

    return fc7

  def load_pretrained_cnn(self, state_dict):
    state_dict = {k: v for k,v in state_dict.items() if "features" in k}
    self.vgg.load_state_dict({k:v for k,v in state_dict.items() if k in self.vgg.state_dict()}, strict=False)
