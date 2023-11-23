import torch
import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
      

    
def load_double_image(file_path, input_height=None, input_width=None, output_height=None, output_width=None,
              crop_height=None, crop_width=None, is_random_crop=True, is_mirror=True, is_gray=False, random_scale=None):
    
    if input_width is None:
      input_width = input_height
    if output_width is None:
      output_width = output_height
    if crop_width is None:
      crop_width = crop_height
    
    img = Image.open(file_path)
    if is_gray is False and img.mode != 'RGB':
      img = img.convert('RGB')
    if is_gray and img.mode != 'L':
      img = img.convert('L')
    
    if random_scale is not None:
        if random.random() < 0.2:
            [w, h] = img.size
            ww = random.randint(int(w*random_scale), w)
            hh = random.randint(int(h*random_scale), h)
            img = img.resize((ww, hh), Image.BICUBIC)
        
    [w, h] = img.size
    # print(img.size, flush=True)
    imgR = ImageOps.crop(img, (0, 0, w//2, 0))
    imgL = ImageOps.crop(img, (w//2, 0, 0, 0))
    # print(imgR.size, flush=True)
    # print(imgL.size, flush=True)
    # assert imgR.size[0] == imgL.size[0]
    # assert imgR.size[1] == h and  imgL.size[1] == h
    
    if is_mirror and random.randint(0,1) == 0:
      imgR = ImageOps.mirror(imgR)
      imgL = ImageOps.mirror(imgL)  

    # if is_mirror and random.randint(0, 1) == 0:
      # imgR = ImageOps.flip(imgR)
      # imgL = ImageOps.flip(imgL)  
    # if is_mirror and random.randint(0,2) < 1:
        # angle = random.randint(-15,15)
        # imgR = TF.rotate(imgR, angle)
        # imgL = TF.rotate(imgL, angle)
      
    if input_height is not None:
      imgR = imgR.resize((input_width, input_height),Image.BICUBIC)
      imgL = imgL.resize((input_width, input_height),Image.BICUBIC)
    
    [w, h] = imgR.size     
    if crop_height is not None:         
      if is_random_crop:
        #print([w,cropSize])        
        cx1 = random.randint(0, w-crop_width) if crop_width < w else 0
        cx2 = w - crop_width - cx1
        cy1 = random.randint(0, h-crop_height) if crop_height < h else 0
        cy2 = h - crop_height - cy1        
      else:
        cx2 = cx1 = int(round((w-crop_width)/2.))
        cy2 = cy1 = int(round((h-crop_height)/2.))
      imgR = ImageOps.crop(imgR, (cx1, cy1, cx2, cy2))
      imgL = ImageOps.crop(imgL, (cx1, cy1, cx2, cy2))          
    if output_height is not None:
      imgR = imgR.resize((output_width, output_height),Image.BICUBIC)
      imgL = imgL.resize((output_width, output_height),Image.BICUBIC)
    return imgR, imgL
     
class DoubleImageDataset(data.Dataset):
    def __init__(self, image_list, root_path, 
                input_height=None, input_width=None, output_height=None, output_width=None,
                crop_height=None, crop_width=None, is_random_crop=False, 
                is_mirror=True, is_gray=False, normalize=None, augment=1, random_scale=None):
        super().__init__()
                
        self.image_filenames = image_list 
        self.is_random_crop = is_random_crop
        self.is_mirror = is_mirror
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.root_path = root_path
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.is_gray = is_gray      
        self.random_scale = random_scale
        
        if normalize is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(), normalize])
            
        self.length = len(self.image_filenames)
        self.augment = augment

    def __getitem__(self, index):          
        idx = index %  self.length
        imgR, imgL = load_double_image(join(self.root_path, self.image_filenames[idx]), 
                                  self.input_height, self.input_width, self.output_height, self.output_width,
                                  self.crop_height, self.crop_width, self.is_random_crop, self.is_mirror, self.is_gray, random_scale=self.random_scale)
        
        imgR = self.transform(imgR)   
        imgL = self.transform(imgL)        
        
        return imgR, imgL

    def __len__(self):
        return len(self.image_filenames) * self.augment
      
def load_single_image(file_path, input_height=128, input_width=None, output_height=128, output_width=None,
              crop_height=None, crop_width=None, is_random_crop=True, is_mirror=True, is_gray=False, random_scale=None):
    
    if input_width is None:
      input_width = input_height
    if output_width is None:
      output_width = output_height
    if crop_width is None:
      crop_width = crop_height
    
    img = Image.open(file_path)
    if is_gray is False and img.mode is not 'RGB':
      img = img.convert('RGB')
    if is_gray and img.mode is not 'L':
      img = img.convert('L')
      
    if random_scale is not None:
        if random.random() < 0.5:
            [w, h] = img.size
            ww = random.randint(int(w*random_scale), w)
            hh = random.randint(int(h*random_scale), h)
            img = img.resize((ww, hh), Image.BICUBIC)
      
    if is_mirror and random.randint(0,1) is 0:
      img = ImageOps.mirror(img)    
      
    if input_height is not None:
      img = img.resize((input_width, input_height),Image.BICUBIC)
      
    [w, h] = img.size  
    if crop_height is not None:      
      if is_random_crop:
        #print([w,cropSize])
        if crop_width < w:
            cx1 = random.randint(0, w-crop_width)
            cx2 = w - crop_width - cx1
        else:
            cx1, cx2 = 0, 0
        if crop_height < h:
            cy1 = random.randint(0, h-crop_height) 
            cy2 = h - crop_height - cy1
        else:
            cy1, cy2 = 0, 0
      else:
        if crop_width < w:
            cx2 = cx1 = int(round((w-crop_width)/2.))
        else:
            cx2 = cx1 = 0
        if crop_height < h:
            cy2 = cy1 = int(round((h-crop_height)/2.))
        else:
            cy2 = cy1 = 0
      img = ImageOps.crop(img, (cx1, cy1, cx2, cy2))      
    if output_height is not None:
        img = img.resize((output_width, output_height),Image.BICUBIC)
    return img
      
class SingleImageDataset(data.Dataset):
    def __init__(self, image_list, root_path, 
                input_height=None, input_width=None, output_height=None, output_width=None,
                crop_height=None, crop_width=None, is_random_crop=False, 
                is_mirror=True, is_gray=False, normalize=None, augment=1, random_scale=None):
        super().__init__()
                
        self.image_filenames = image_list 
        self.is_random_crop = is_random_crop
        self.is_mirror = is_mirror
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.root_path = root_path
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.is_gray = is_gray      
        self.random_scale = random_scale
                       
        if normalize is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(), normalize])
                               
        self.length = len(self.image_filenames)
        self.augment = augment

    def __getitem__(self, index):  
        
        idx = index %  self.length  
        img = load_single_image(join(self.root_path, self.image_filenames[idx]), 
                                  self.input_height, self.input_width, self.output_height, self.output_width,
                                  self.crop_height, self.crop_width, self.is_random_crop, self.is_mirror, self.is_gray, random_scale=self.random_scale)
        
        img = self.transform(img)             
        
        return img

    def __len__(self):
        return len(self.image_filenames) * self.augment