import torch.nn as nn
import torch.nn.functional as F
import torch
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


from torch.autograd import Variable
from torchvision import models

#import models.stn_alex as stn_model
from .easyAngle import angleModel
from .ocr import ocrModel

class CRNN(nn.Module):
    def __init__(self, nc, nclass, nh, imgH=32, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        
        self.ocr = ocrModel(nclass=nclass)
        self.angle = angleModel()
        model_path = '/izola/xyl/pytorch-faster-rcnn/pretrainedModel.pth'
        self.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))        
        for param in self.angle.parameters():
            param.requires_grad = False
        '''
        for param in self.angle.parameters():
            param.requires_grad = False
        '''
        for param in self.ocr.parameters():
            param.requires_grad = True

    # Spatial transformer network forward function
    def stn(self, x):
        _,_,h,w = x.size()
        conv = self.angle(x)
        b,c = conv.size()
        ang = conv.view(1,-1)/57.3
        #print(ang*57.3)
        #print(torch.cos(ang))
        theta = Variable(torch.zeros(b,6).cuda())
        theta[:,0] = torch.cos(ang)
        theta[:,1] = torch.sin(ang) #* 0.3
        theta[:,3] = -torch.sin(ang)
        theta[:,4] = torch.cos(ang) #* 0.3
        theta = theta.view(-1,2,3)
        theta = theta*1.4

        grid = F.affine_grid(theta, torch.Size((b,c,30,100)))
        x = F.grid_sample(x, grid)
        return x, ang
    def forward(self, input):
        #print(input)
        conv, ang = self.stn(input)
        #print(ang)
        output = self.ocr(conv)
        return output#, ang
    
    def load_angle(self, model_path):
        angle_pre = angleModel()
        angle_pre.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))        
        self.angle.load_state_dict(angle_pre.state_dict())
        for param in self.angle.parameters():
            param.requires_grad = False
        del angle_pre
    def load_ocr(self, model_path):
        ocr_pre = ocrModel()
        ocr_pre.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))        
        self.ocr.load_state_dict(ocr_pre.state_dict())
        del ocr_pre
    def trainAngleOnly(self):
        #self.ocr.eval()
        #self.angle.train()
        print('train angle only')
        for param in self.angle.parameters():
            param.requires_grad = True
        for param in self.ocr.parameters():
            param.requires_grad = False
    def trainOcrOnly(self):
        #self.ocr.train()
        #self.angle.eval()
        print('train ocr only')
        for param in self.angle.parameters():
            param.requires_grad = False
        for param in self.ocr.parameters():
            param.requires_grad = True
