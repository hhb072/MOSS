import os
import argparse
import numpy as np
import random 
import time
from os.path import join
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from dataset import *

try:
    from ruamel import yaml
except:
    import yaml
from easydict import EasyDict as edict

from PIL import Image, ImageOps
import torchvision.transforms.functional as TF

import torchvision.utils as vutils

import skimage
from skimage import io,transform
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from networks import NetG

str_to_list = lambda x: [int(xi) for xi in x.split(',')]

def enable_path(path):
    try:
        os.makedirs(path)
    except OSError:
        pass

def get_config(parser):

    args = parser.parse_args()
    args = edict(vars(args))

    cfg_file_path = args.config_file

    with open(cfg_file_path, 'r') as stream:
        config = edict(yaml.load(stream))

    config.update(args)
    return config
  
def compute_metrics(img, gt, multichannel=True):
    img = img.numpy().transpose((0, 2, 3, 1))
    gt = gt.numpy().transpose((0, 2, 3, 1)) 
    img = img[0,:,:,:] * 255.
    gt = gt[0,:,:,:] * 255.
    img = np.array(img, dtype = 'uint8')
    gt = np.array(gt, dtype = 'uint8')
    if not multichannel:
        gt = skimage.color.rgb2ycbcr(gt)[:,:,0]
        img = skimage.color.rgb2ycbcr(img)[:,:,0] 
    cur_psnr = compare_psnr(img, gt, data_range=255)
    cur_ssim = compare_ssim(img, gt, data_range=255, multichannel=multichannel)
    return cur_psnr, cur_ssim       

def main(config):

    model = NetG(config).cuda()
    if config.pretrained:
        state = torch.load(config.pretrained)
        model.load_state_dict(state)       
        
    test_list = os.listdir(config.testroot)    
            
    def test():
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        test_set = DoubleImageDataset(test_list, config.testroot, crop_height=None, output_height=None, is_random_crop=False, is_mirror=False, normalize=normalize)
        test_data_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    
        psnrs, ssims = [], [] 
        enable_path(config.save_image_path)
        
        model.eval()
        with torch.no_grad():
            for iteration, batch in enumerate(test_data_loader, 0):                
                data, label = batch
                if len(data.size()) == 3:
                    data, label = data.unsqueeze(0), label.unsqueeze(0)                
                data = Variable(data).cuda()
                label = Variable(label).cuda()
                
                fake = model(data)
                                
                data, label, fake = [x*0.5+0.5 for x in [data, label, fake]]
                
                fake, label = fake.data.cpu(), label.data.cpu()
                for i in range(fake.shape[0]):
                    psnr, ssim = compute_metrics(fake[i:i+1], label[i:i+1])
                    psnrs.append(psnr)
                    ssims.append(ssim)
                                
                vutils.save_image(fake, '{}/{}'.format(config.save_image_path, test_list[iteration]))
              
        print('Dense:\tPSNR: {:.2f}, SSIM: {:.4f}'.format(np.mean(psnrs[:10]), np.mean(ssims[:10])))
        print('Sparse:\tPSNR: {:.2f}, SSIM: {:.4f}'.format(np.mean(psnrs[10:]), np.mean(ssims[10:])))
        print('Average:\tPSNR: {:.2f}, SSIM: {:.4f}'.format(np.mean(psnrs), np.mean(ssims)))
        
        return np.mean(psnrs), np.mean(ssims)
        
    def test_real():
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        real_set = SingleImageDataset(test_list, config.testroot, crop_height=None, output_height=None, is_random_crop=False, is_mirror=False, normalize=normalize)
        real_dataloader = DataLoader(real_set, batch_size=1, shuffle=False)
    
        psnrs, ssims = [], []
        
        enable_path(config.save_image_path)
    
        model.eval()
      
        with torch.no_grad():
            for iteration, batch in enumerate(real_dataloader, 0):                
                data = batch
                if len(data.size()) == 3:
                    data = data.unsqueeze(0)               
                data = Variable(data).cuda()
                _, c, h, w = data.size()
                h1 = math.ceil(h / 8.) * 8
                w1 = math.ceil(w / 8.) * 8
                if h1 != h or w1 != w:
                    data = F.interpolate(data, (h1, w1), mode='bicubic')
                                
                fake = model(data)
                if h1 != h or w1 != w:
                    fake = F.interpolate(fake, (h, w), mode='bicubic')
                               
                vutils.save_image(fake*0.5+0.5, '{}/{}'.format(config.save_image_path, test_list[iteration]))
    
    if config.test_real:
        test_real()
    else:
        test()
            
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default="test.yaml", type=str, help='the path of config file')         

    main(get_config(parser))

