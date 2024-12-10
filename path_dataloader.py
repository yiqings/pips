'''
Specialized dataset for pathological image dataset

It can load the following datasets:
    1. CRC-100K-Nonorm/~/7K
    2. lc25000
    
Other datasets can be added as long as they are each-class-one-folder.
    
'''
import torch
from torch.utils.data import Dataset,DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import argparse
from PIL import Image
from PIL import ImageFile
from skimage import io,color
def get_dataset_info(data_name, data_path):
    if data_name=="crc-100k-nonorm":
        traindir=os.path.join(data_path,'NCT-CRC-HE-100K-NONORM') 
        class_to_idx={'ADI': 0, 'BACK': 1, 'DEB': 2, 'LYM': 3, 'MUC': 4, 'MUS': 5, 'NORM': 6, 'STR': 7, 'TUM': 8}
     
    elif data_name=="crc-100k-norm":
        traindir=os.path.join(data_path,'NCT-CRC-HE-100K') 
        class_to_idx={'ADI': 0, 'BACK': 1, 'DEB': 2, 'LYM': 3, 'MUC': 4, 'MUS': 5, 'NORM': 6, 'STR': 7, 'TUM': 8}
    
    elif data_name=="lc25000":
        traindir=data_path
        #traindir = os.path.join(data_path, 'lung_colon_image_set/lung_image_sets')
        class_to_idx={'lung_aca': 0, 'lung_n': 1, 'lung_scc': 2,'colon_aca': 3, 'colon_n': 4}

    elif data_name=="lc25000-colon":
        traindir = os.path.join(data_path, 'lung_colon_image_set/colon_image_sets')
        class_to_idx={'colon_aca': 0, 'colon_n': 1}
        
    return traindir,class_to_idx




class WSIData(Dataset):
    def __init__(self,file_path,data_name,transform=None,test_flag=0,img_color_space="lab"):
        super(WSIData, self).__init__()
        self.file_path=file_path
        self.transform=transform
        self.data_path, self.class_to_idx=get_dataset_info(data_name,file_path)
        self.num_class=len(self.class_to_idx)
        self.cnt=np.zeros(self.num_class)
        self.img_color_space=img_color_space
        self.image_names=[]
        for i in os.listdir(self.file_path):
            tem=os.path.join(file_path,i)
            print(i,tem,len(os.listdir(tem)))
            
            for j in os.listdir(tem):
                self.image_names.append(os.path.join(tem,j))
                self.cnt[self.class_to_idx[i]]+=1
            
    def __getitem__(self,index):
        y=self.class_to_idx[self.image_names[index].split('/')[-2]] # .../class_name/xxxx.jpg
        rgb=Image.open(self.image_names[index])
        rgb=self.transform(rgb).permute(1,2,0)
        return rgb,y
        '''
        if self.img_type=="rgb":
            return rgb.permute(1,2,0),y
        elif self.img_type=='lab':
            lab=color.rgb2lab(rgb).transpose(2,0,1).astype(np.float32)
            return lab,y
        '''
        print("ERROR: Color space only support rgb, lab")
    
    '''get_class_count: output the actual number of each classes'''
    def get_class_count(self):
        now_path=self.data_path
        cnt=np.zeros(self.num_class)
        for i in os.listdir(now_path):
            x=os.path.join(now_path,i)
            cnt[self.class_to_idx[i]]=len(os.listdir(x))
            #for j in os.listdir(x):
                #self.image_names.append(os.pth.join(x,j))
        
        for i in self.class_to_idx.keys():
            print(i,self.class_to_idx[i],cnt[self.class_to_idx[i]])   
            
    def get_detailed_info(self):
        print("Not yet")        
    def __len__(self):
        return len(self.image_names)      


    
     
        
        
    
    