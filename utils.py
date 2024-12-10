'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init

import torch
import random
import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset,DataLoader
from skimage import io,color
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np





def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

term_width = 80
#_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

import torch

def rgb2xyz(img):
    """
    RGB from 0 to 255
    :param img:
    :return:
    """
    r, g, b = torch.split(img, 1, dim=1)

    r = torch.where(r > 0.04045, torch.pow((r+0.055) / 1.055, 2.4), r / 12.92)
    g = torch.where(g > 0.04045, torch.pow((g+0.055) / 1.055, 2.4), g / 12.92)
    b = torch.where(b > 0.04045, torch.pow((b+0.055) / 1.055, 2.4), b / 12.92)

    r = r * 100
    g = g * 100
    b = b * 100

    x = r * 0.412453 + g * 0.357580 + b * 0.180423
    y = r * 0.212671 + g * 0.715160 + b * 0.072169
    z = r * 0.019334 + g * 0.119193 + b * 0.950227
    return torch.cat([x,y,z], dim=1)


def xyz2lab(xyz):
    x, y, z = torch.split(xyz, 1, dim=1)
    ref_x, ref_y, ref_z = 95.047, 100.000, 108.883
    # ref_x, ref_y, ref_z = 0.95047, 1., 1.08883
    x = x / ref_x
    y = y / ref_y
    z = z / ref_z

    x = torch.where(x > 0.008856, torch.pow( x , 1/3 ), (7.787 * x) + (16 / 116.))
    y = torch.where(y > 0.008856, torch.pow( y , 1/3 ), (7.787 * y) + (16 / 116.))
    z = torch.where(z > 0.008856, torch.pow( z , 1/3 ), (7.787 * z) + (16 / 116.))

    l = (116. * y) - 16.
    a = 500. * (x - y)
    b = 200. * (y - z)
    return torch.cat([l,a,b], dim=1)


def lab2xyz(lab):
    ref_x, ref_y, ref_z = 95.047, 100.000, 108.883
    l, a, b = torch.split(lab, 1, dim=1)
    y = (l + 16) / 116.
    x = a / 500. + y
    z = y - b / 200.

    y = torch.where(torch.pow( y , 3 ) > 0.008856, torch.pow( y , 3 ), ( y - 16 / 116. ) / 7.787)
    x = torch.where(torch.pow( x , 3 ) > 0.008856, torch.pow( x , 3 ), ( x - 16 / 116. ) / 7.787)
    z = torch.where(torch.pow( z , 3 ) > 0.008856, torch.pow( z , 3 ), ( z - 16 / 116. ) / 7.787)

    x = ref_x * x
    y = ref_y * y
    z = ref_z * z
    return torch.cat([x,y,z],dim=1)


def xyz2rgb(xyz):
    x, y, z = torch.split(xyz, 1, dim=1)

    x = x / 100.
    y = y / 100.
    z = z / 100.

    r = x * 3.2406 + y * -1.5372 + z * -0.4986
    g = x * -0.9689 + y * 1.8758 + z * 0.0415
    b = x * 0.0557 + y * -0.2040 + z * 1.0570

    r = torch.where(r > 0.0031308, 1.055 * torch.pow( r , ( 1 / 2.4 ) ) - 0.055,  12.92 * r)
    g = torch.where(g > 0.0031308, 1.055 * torch.pow( g , ( 1 / 2.4 ) ) - 0.055,  12.92 * g)
    b = torch.where(b > 0.0031308, 1.055 * torch.pow( b , ( 1 / 2.4 ) ) - 0.055,  12.92 * b)

    r = torch.round(r * 255.)
    g = torch.round(g * 255.)
    b = torch.round(b * 255.)

    return torch.cat([r,g,b], dim=1)

def lab2rgb(lab):
    return xyz2rgb(lab2xyz(lab))

class WSIData(Dataset):
    def __init__(self, file_path, transform = None,test_flag=0):
        super(WSIData, self).__init__()
        self.file_path = file_path
        self.transform = transform  
        self.image_names = []
        self.cnt=[0 for i in range(9)]
        self.class_to_idx={'ADI': 0, 'BACK': 1, 'DEB': 2, 'LYM': 3, 'MUC': 4, 'MUS': 5, 'NORM': 6, 'STR': 7, 'TUM': 8}
        
        for i in os.listdir(self.file_path):
            tem=os.path.join(file_path,i)
            print(tem,len(os.listdir(tem)))
            for j in os.listdir(tem):
                self.image_names.append(os.path.join(tem,j))
                self.cnt[self.class_to_idx[i]]+=1
        print(self.cnt, sum(self.cnt))
        self.flag=test_flag
        #print(self.image_names)

    def __getitem__(self,index):
        y=self.class_to_idx[self.image_names[index].split('/')[-2]]
        rgb=io.imread(self.image_names[index])
        '''
        if y==1:
            if self.flag:
                io.imsave('./1.jpg', rgb)
                rgb=cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                io.imsave('./2.jpg', rgb)
            else:
                io.imsave('./3.jpg', rgb)
        '''
        lab=color.rgb2lab(rgb).transpose(2,0,1).astype(np.float32)
        #x=self.transform(lab)
        return lab,y
    def __len__(self):
        return len(self.image_names)
    
    
def plot_confusion_matrix(cm, savepath,classes,title='Confusion Matrix'):

    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%.3f" % (c,), color='black', fontsize=15, va='center', ha='center')
    
    plt.imshow(cm, interpolation='nearest', cmap='YlOrBr')
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')
    
    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    
    # show confusion matrix
    plt.savefig(savepath, format='png')
    plt.show()