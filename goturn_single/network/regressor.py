# Date: Friday 02 June 2017 05:04:00 PM IST 
# Email: nrupatunga@whodat.com
# Name: Nrupatunga
# Description: Basic regressor function implemented

from __future__ import print_function
import os
import glob
import numpy as np
import sys
import cv2
from ..helper import config
import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11,stride=4)
        self.conv2 = nn.Conv2d(96,256,kernel_size=5,groups=2,padding=2)
        self.conv3 = nn.Conv2d(256,384,kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(384,384,kernel_size=3,padding=1,groups=2)
        self.conv5 = nn.Conv2d(384,256,kernel_size=3,padding=1,groups=2)
        self.fc6_new = nn.Linear(in_features=6*6*256*2, out_features=4096)
        self.fc7_new = nn.Linear(in_features=4096, out_features=4096)
        self.fc7_newb = nn.Linear(in_features=4096, out_features=4096)
        self.fc8_shapes = nn.Linear(in_features=4096, out_features=4)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.lrn = nn.LocalResponseNorm(5)

        self.channels = 3
        self.height = 227
        self.width = 227
        self.mean = [104,117,123]


    def preprocess(self, image):
        num_channels = self.channels
        if num_channels == 1 and image.shape[2] == 3:
            image_out = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif num_channels == 1 and image.shape[2] == 4:
            image_out = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif num_channels == 3 and image.shape[2] == 4:
            image_out = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif num_channels == 3 and image.shape[2] == 1:
            image_out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_out = image

        if image_out.shape != (self.height, self.width, self.channels):
            image_out = cv2.resize(image_out, (self.width, self.height), interpolation=cv2.INTER_CUBIC)

        image_out = np.float32(image_out)
        image_out -= np.array(self.mean)
        image_out = np.transpose(image_out, [2, 0, 1])
        image_out = image_out.reshape(1, 3, 227,227)
        return image_out

    def conv(self,x):
        x = self.lrn(self.pool(F.relu(self.conv1(x))))
        x = self.lrn(self.pool(F.relu(self.conv2(x))))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(F.relu(self.conv5(x)))
        x = x.view(1,-1,x.shape[2],x.shape[3])
        return x

    def forward(self, s, t):
        s_p = self.preprocess(s)
        t_p = self.preprocess(t)
        x = np.concatenate((s_p,t_p),0)
        x = torch.from_numpy(x)
        x = self.conv(x)
        x = x.view(-1, 6*6*2*256)
        x = F.relu(self.fc6_new(x))
        x = F.relu(self.fc7_new(x))
        x = F.relu(self.fc7_newb(x))
        x = self.fc8_shapes(x)
        x = x.numpy()
        #print(x)
        #print(x.shape)
        return x
