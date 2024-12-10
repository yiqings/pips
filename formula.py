#from wsi_task import Classifier
import torch
import random
import tqdm
import numpy as np
from utils import progress_bar
import torch
import torch.nn as nn
import time
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset,DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision import models

# from torchsummary import summary

import os
import argparse
from skimage import io,color
from progressbar import *
import cv2
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def apply_color_style_lab_tensor(img,new_color,color_space_type,example_flg=0):
    '''
     color_space_type:
      0 - rgb
      1 - lab
      2 - hsv
      3 - hed
    '''
    if example_flg==0 or example_flg==-1:
        #print(img.shape)
        #img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
        if color_space_type==1: # lab
            img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        elif color_space_type==2: # hsv
            img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV_FULL)
            img[:,:,0]=img[:,:,0]/360*255
        elif color_space_type==3: #hed
            img=cv2.cvtColor(img, cv2.COLOR_BGR2HED)
            #img=color.rgb2hed(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  
        elif color_space_type==0: # rgb
            img=img
            
        '''
        if color_space_type==2:
            print(img_avg,img_std)
            print(new_color)
        '''
        img_avg,img_std=getavgstd(img)
        img_std=np.clip(img_std,0.0001,255)
        img=(img-img_avg)*(new_color[3:]/img_std)+new_color[0:3]
        img=np.clip(img,0.0001, 255)
        if color_space_type==1 or color_space_type==2 or color_space_type==0: # lab hsv
            if example_flg==-1:
                if color_space_type==1: # lab
                    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_LAB2BGR) 
                    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                    img=(img*255).astype(np.uint8)
                    return img
                elif color_space_type==2: # hav
                    
                    img[:,:,0]=img[:,:,1]/255.0*360.0
                    img=cv2.cvtColor((img).astype(np.uint8),cv2.COLOR_HSV2BGR_FULL)
                    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                    img=(img*255).astype(np.uint8)
                    return img
                elif color_space_type==0:
                    img=img.astype(np.uint8)#cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_RGB2BGR)
            img=np.clip(img,0.001,255).astype(np.uint8)
        return img
    else:
        return apply_color_style_lab_tensor(img, new_color,example_flg-1,-1)

def virtualstain(image, state,example_flg=0):
    #print(image.shape)
    vs_epoch=time.time()
    image=np.array(image)
    tem=[]
    for i in image:
        ttem=None
        if example_flg!=0:
            ttem=apply_color_style_lab_tensor(i,state[example_flg-1][1:],0,example_flg)
            return ttem
        for j in range(len(state)):
            tem_state=state[:,0]
            tem_state=state[:,0]/np.mean(tem_state)/len(state)
            if ttem is None:
                ttem=tem_state[j]*apply_color_style_lab_tensor(i,state[j][1:],j,example_flg)/255.0
            else:
                ttem+=tem_state[j]*apply_color_style_lab_tensor(i,state[j][1:],j,example_flg)/255.0
        tem.append(torch.tensor(np.array(ttem), dtype=torch.float32))
    return torch.tensor(np.array([i.detach().numpy() for i in tem])).permute(0,3,2,1)
    #return torch.tensor(np.array(tem).astype(np.float32), dtype=torch.float32)

def train_one_epoch(epoch, state,cls_model,cls_optimizer, cls_criterion, trainloader, device=torch.device("cuda")):
    cls_model.train()
    train_loss,total,correct=0,0,0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs=virtualstain(inputs,state)
        inputs, targets=inputs.to(device), targets.to(device)
        cls_optimizer.zero_grad()
        
        outputs=cls_model(inputs)
        loss=cls_criterion(outputs, targets)
        loss.backward()
        cls_optimizer.step()
        
        train_loss+=loss.item()
        _, predicted=outputs.max(1)
        total+=targets.size(0)
        correct+=predicted.eq(targets).sum().item()
        
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return 100. * correct / total

def test_one_time(epoch, state,cls_model, cls_criterion, testloader,device=torch.device("cuda")):
    cls_model.eval()
    test_loss,total,correct=0,0,0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs=virtualstain(inputs,state)
            inputs, targets=inputs.to(device), targets.to(device)
            #inputs, targets=inputs.to(device), targets.to(device)
            outputs=cls_model(inputs)
            loss=cls_criterion(outputs, targets)
            test_loss+=loss.item()
            _,predicted=outputs.max(1)
            total+=targets.size(0)
            correct+=predicted.eq(targets).sum().item()
        '''
        if cm_flag:
            if batch_idx==0:
                true_y,pred_y=targets,predicted
            else:
                true_y=torch.cat((true_y,targets),0)
                pred_y=torch.cat((pred_y,predicted),0)
        
            progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        '''
    acc = 100. * correct / total
    loss=test_loss / (batch_idx + 1) 
    return acc, loss
            


def individual_train(epoch, individual, trainloader, testloader, start_training, device=torch.device("cuda")):
    
    record=[]
    reward=-100
    if start_training==1:
        cls_model=models.resnet18(pretrained=True)
        cls_model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        cls_model.fc = nn.Linear(cls_model.fc.in_features, 5) # classes number
        cls_criterion=nn.CrossEntropyLoss()
        cls_optimizer=optim.SGD(cls_model.parameters(), lr=0.001,momentum=0.9, weight_decay=5e-4)
        cls_model.to(device)
    else:
        individual.cls_model=individual.cls_model.to(device)
    for i in range(epoch):
        if start_training==1:
            now_acc=train_one_epoch(i,individual.template,cls_model,cls_optimizer,cls_criterion, trainloader)
            test_acc,_=test_one_time(i,individual.template,cls_model,cls_criterion,testloader)
        else:
            now_acc=train_one_epoch(i,individual.template,individual.cls_model, individual.cls_optimizer, individual.cls_criterion, trainloader)
            test_acc,_=test_one_time(i,individual.template, individual.cls_model, individual.cls_criterion, testloader)
        reward=max(reward, test_acc)
        record.append((now_acc,test_acc))
        print("Epoch", i, "Training/test",now_acc,"/",test_acc)
        
    if start_training==0:
        individual.cls_model=individual.cls_model.to('cpu')
    else:
        cls_model.to('cpu')
    individual.histo.append(record)
    for i in record:
        print(i)
    individual.reward=reward
    return reward
        

def PSO_train(pso,trainloader,testloader,start_training, epoch=3,device=torch.device("cuda")):
    '''
     Train the entire Particle Sward and return (the global best template in this generation ,its reward)
        start_training: test the fitness from the scratch(a brand-new model)
    '''
    # init gbest & its reward
    gbest=pso.group[0].template
    gbest_reward=-100
    cnt=0
    for i in pso.group:
        print("############ Particle",cnt,"##############")
        now_reward=individual_train(epoch,i,trainloader,testloader,start_training,device)
        if gbest_reward<now_reward:
            gbest=i.template
            gbest_reward=now_reward
        cnt+=1
        
    if gbest_reward>pso.gbest_reward: # if better than past record -> update
        pso.gbest=gbest 
        pso.gbest_reward=gbest_reward
    return gbest,gbest_reward

def getavgstd(image):
    # image: np.array
    avg = []
    std = []
    image_avg_l = np.mean(image[:, :, 0])
    image_std_l = np.std(image[:, :, 0])
    image_avg_a = np.mean(image[:, :, 1])
    image_std_a = np.std(image[:, :, 1])
    image_avg_b = np.mean(image[:, :, 2])
    image_std_b = np.std(image[:, :, 2])
    avg=[image_avg_l,image_avg_a,image_avg_b]
    std=[image_std_l,image_std_a,image_std_b]
    return (avg, std)
