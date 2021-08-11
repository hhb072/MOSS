import numbers
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from  torch.nn.parallel import data_parallel
from functools import wraps
import random

class _Residual_Block(nn.Module):
    def __init__(self, in_planes, out_planes, groups=1, wide_width=True, downsample=False, upsample=False):
        super().__init__()
        
        self.downsample = downsample
        self.upsample = upsample
        
        middle_planes = (in_planes if in_planes > out_planes else out_planes) if wide_width else out_planes
        
        self.conv1 = nn.Conv2d(in_planes, middle_planes, 3, 1, 1, bias=False, groups=groups)
        self.relu1 = nn.LeakyReLU(0.02, inplace=True)        
        self.conv2 = nn.Conv2d(middle_planes, out_planes, 3, 1, 1, bias=False, groups=groups)
                
        if in_planes != out_planes:            
            self.translation = nn.Conv2d(in_planes, out_planes, 1, 1, 0, bias=False, groups=groups)
        else:
            self.translation = None
    
    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear')
    
        identity = x

        out = self.conv1(x)        
        out = self.relu1(out)

        out = self.conv2(out)
        
        if self.translation is not None:
            identity = self.translation(identity)
        
        out += identity
        
        if self.downsample:
            out = F.avg_pool2d(out, 2)
        
        return out
        
def make_layer(block, blocks, in_planes, out_planes, groups=1, norm_layer=nn.BatchNorm2d, downsample=False, upsample=False):
    assert blocks >= 1
    layers = []
    layers.append(block(in_planes, out_planes, groups, norm_layer, downsample=downsample, upsample=upsample))
    for i in range(1, blocks):
        layers.append(block(out_planes, out_planes, groups, norm_layer))
        
    return nn.Sequential(*layers)        

class _Memory_Block(nn.Module):        
    def __init__(self, hdim, kdim, moving_average_rate=0.999):
        super().__init__()
        
        self.c = hdim
        self.k = kdim
        
        self.moving_average_rate = moving_average_rate
        
        self.units = nn.Embedding(kdim, hdim)
                
    def update(self, x, score, m=None):
        '''
            x: (n, c)
            e: (k, c)
            score: (n, k)
        '''
        if m is None:
            m = self.units.weight.data
        x = x.detach()
        embed_ind = torch.max(score, dim=1)[1] # (n, )
        embed_onehot = F.one_hot(embed_ind, self.k).type(x.dtype) # (n, k)        
        embed_onehot_sum = embed_onehot.sum(0)
        embed_sum = x.transpose(0, 1) @ embed_onehot # (c, k)
        embed_mean = embed_sum / (embed_onehot_sum + 1e-6)
        new_data = m * self.moving_average_rate + embed_mean.t() * (1 - self.moving_average_rate)
        if self.training:
            self.units.weight.data = new_data
        return new_data
                
    def forward(self, x, update_flag=True):
        '''
          x: (b, c, h, w)
          embed: (k, c)
        '''
        
        b, c, h, w = x.size()        
        assert c == self.c        
        k, c = self.k, self.c
        
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, c) # (n, c)
        
        m = self.units.weight.data # (k, c)
                
        xn = F.normalize(x, dim=1) # (n, c)
        mn = F.normalize(m, dim=1) # (k, c)
        score = torch.matmul(xn, mn.t()) # (n, k)
        
        if update_flag:
            m = self.update(x, score, m)
            mn = F.normalize(m, dim=1) # (k, c)
            score = torch.matmul(xn, mn.t()) # (n, k)
        
        soft_label = F.softmax(score, dim=1)
        out = torch.matmul(soft_label, m) # (n, c)
        out = out.view(b, h, w, c).permute(0, 3, 1, 2)
                                
        return out, score

class _Fuse_Block(nn.Module):
    def __init__(self, x_dim, res_dim):
        super().__init__()
        
        self.norm = nn.BatchNorm2d(res_dim, affine=False)
        
        self.alpha = nn.Conv2d(res_dim, 1, 1, 1, 0, bias=False)
        
        self.gamma = nn.Conv2d(x_dim, res_dim, 1, 1, 0, bias=False)
        self.beta = nn.Conv2d(x_dim, res_dim, 1, 1, 0, bias=False)
        
        self.affine_gamma = nn.Parameter(torch.zeros(1, res_dim, 1, 1))
        self.affine_beta = nn.Parameter(torch.zeros(1, res_dim, 1, 1))
                
        self.gamma.weight.data.zero_()
        self.beta.weight.data.zero_()
        
    def forward(self, x, res):
        res = self.norm(res)
        gamma = self.gamma(x) + 1
        beta = self.beta(x)
        out0 = res * gamma + beta
        out1 = res * (self.affine_gamma + 1) + self.affine_beta
        alpha = torch.sigmoid(self.alpha(res))
        if self.training:
            drop = torch.rand((res.shape[0], 1, 1, 1), device=res.device).gt(0.2).float()
            alpha = alpha * drop
        out = (1 - alpha) * out0 + alpha * out1
        return out

class NetG(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.num_blocks = config.num_blocks
        
        self.head = nn.Conv2d(3, config.ngf, 3, 1, 1, bias=False)
        cc = config.ngf 
        self.enc = nn.ModuleDict()
        for i in range(config.num_blocks):
            self.enc['enc{}'.format(i)] = make_layer(_Residual_Block, config.num_layers, cc, cc+config.delta, downsample=i<config.num_scales)
            cc += config.delta
        
        self.memory = _Memory_Block(cc, config.kdim, config.moving_average_rate)
        
        self.dec = nn.ModuleDict()
        self.fuse = nn.ModuleDict()
        for i in range(config.num_blocks):            
            self.dec['dec{}'.format(i)] = make_layer(_Residual_Block, config.num_layers, cc, cc-config.delta, upsample=i>config.num_blocks-config.num_scales-1)
            cc -= config.delta             
            if i < config.num_blocks-1:
                self.fuse['fuse{}'.format(i)] = _Fuse_Block(cc, cc)
               
        self.tail = nn.Conv2d(config.ngf, 3, 3, 1, 1)   
    
            
    def forward(self, x):
        x = self.head(x)
        xi = x
        
        res = []
        for i in range(self.num_blocks):
            x = self.enc['enc{}'.format(i)](x)
            res.append(x)
        
        x, _ = self.memory(x)
        
        res = res[::-1]
        x = self.dec['dec0'](x)
        for i in range(self.num_blocks-1):
            x = self.fuse['fuse{}'.format(i)](x, res[i+1])
            x = self.dec['dec{}'.format(i+1)](x)
        
        x = xi - x
        x = self.tail(x)
        x = torch.tanh(x)
        return x
                 
         


