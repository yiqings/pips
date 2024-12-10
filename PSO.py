import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import argparse
from PIL import Image
from PIL import ImageFile
import random

class Particle():
    def __init__(self,num_color_space,minn,maxn,maxv,c1=2,c2=2):
        self.num_color_space=num_color_space 
        
        # 7: weight + [,,] mean + [,,] std
        self.template=[[random.randint(minn,maxn)for i in range(7)]for j in range(num_color_space)]
        self.template=np.array(self.template,dtype=np.float32)
        
        self.c1,self.c2=c1,c2
        self.v=[[random.randint(0,maxv) for i in range(7)]for j in range(num_color_space)]
        self.v=np.array(self.v)
        self.reward=-100
        self.histo=[]
        self.pbest=self.template
        self.pbest_reward=self.reward
        self.maxv=maxv
        self.dec=0.99
        
        self.w_init=0.9
        self.w_end=0.4
        
        self.G_k=20
        self.now_g=1
        
    def update(self,gbest):
        '''
            gbest: global best template
            now_g: now generation
        '''
        # update self past best template
        if self.pbest_reward<self.reward:
            self.pbest_reward=self.reward
            self.pbest=self.template
        tem_w=(self.w_init-self.w_end)*(self.G_k-self.now_g)/self.G_k+self.w_end
        tem_v=self.v*tem_w
        new_v_part1=self.c1*random.random()*(self.pbest-self.template)
        new_v_part2=self.c1*random.random()*(gbest-self.template)
        
        self.v=tem_v+new_v_part1+new_v_part2
        self.v=self.v/np.mean(self.v)*self.maxv
        self.template=self.template+self.v
        
        # justify the template (std/mean)
        for k in range(len(self.template)):
            for i in range(1,7):
                if int(self.template[k][i])>=255:
                    self.template[k][i]=255
                elif self.template[k][i]<=0:
                    self.template[k][i]=1
            for i in range(1,4):
                j=i+3
                if self.template[k][i]+self.template[k][j]>255:
                    self.template[k][j]=255-self.template[k][i]
                elif self.template[k][i]-self.template[k][j]<0:
                    self.template[k][j]=self.template[k][i]
            
        self.c1*=self.dec
        self.c2*=self.dec
        self.now_g+=1
        
    def info_print(self):
        print("--------Particle info-----------")
        print("now",self.reward,self.template)
        print("pbest:",self.pbest_reward,self.pbest)
        
        
# PSO class
class PSO():
    def __init__(self, part_num=10, num_color_space=3,dim=6,minn=1,maxn=255,maxv=10,c1=2,c2=2):
        
        self.group=[Particle(num_color_space=num_color_space,minn=minn,maxn=maxn,maxv=maxv,c1=c1,c2=c2)for i in range(part_num)]
        self.gbest=self.group[0].template
        self.gbest_reward=-100
        self.part_num=part_num
    def update(self):
        for i in range(self.part_num): # find the global best in the group
            if self.group[i].reward>self.gbest_reward:
                self.gbest_reward=self.group[i].reward
                self.gbest=self.group[i].template
        for i in range(self.part_num):
            self.group[i].update(self.gbest)
            
    def info_print(self):
        print("Global best",self.gbest_reward, self.gbest)
        for i in self.group:
            i.info_print()