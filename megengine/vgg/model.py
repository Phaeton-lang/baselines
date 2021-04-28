# Copyright (c) 2021 JSON LEE, lijiansong@ict.ac.cn.
# All rights reserved.

import megengine.module as M
import megengine.functional as F

# VGG with cifar-10 dataset, model architecture refers to:
# http://torch.ch/blog/2015/07/30/cifar.html
class VGG(M.Module):
    def __init__(self):
        super().__init__()
        # conv1_1: ConvBnRelu
        self.conv1_1 = M.Sequential(
                M.Conv2d(3, 64, 3, stride=1, padding=1),
                M.BatchNorm2d(64),
                M.ReLU(),)
        self.drop1 = M.Dropout(0.3)
        # conv1_2: ConvBnRelu
        self.conv1_2 = M.Sequential(
                M.Conv2d(64, 64, 3, stride=1, padding=1),
                M.BatchNorm2d(64),
                M.ReLU(),)
        self.pool1 = M.MaxPool2d(2, 2)
        # conv2_1: ConvBnRelu
        self.conv2_1 = M.Sequential(
                M.Conv2d(64, 128, 3, stride=1, padding=1),
                M.BatchNorm2d(128),
                M.ReLU(),)
        self.drop2_1 = M.Dropout(0.4)
        # conv2_2: ConvBnRelu
        self.conv2_2 = M.Sequential(
                M.Conv2d(128, 128, 3, stride=1, padding=1),
                M.BatchNorm2d(128),
                M.ReLU(),)
        self.pool2 = M.MaxPool2d(2, 2)
        # conv3_1: ConvBnRelu
        self.conv3_1 = M.Sequential(
                M.Conv2d(128, 256, 3, stride=1, padding=1),
                M.BatchNorm2d(256),
                M.ReLU(),)
        self.drop3_1 = M.Dropout(0.4)
        # conv3_2: ConvBnRelu
        self.conv3_2 = M.Sequential(
                M.Conv2d(256, 256, 3, stride=1, padding=1),
                M.BatchNorm2d(256),
                M.ReLU(),)
        self.drop3_2 = M.Dropout(0.4)
        # conv3_3: ConvBnRelu
        self.conv3_3 = M.Sequential(
                M.Conv2d(256, 256, 3, stride=1, padding=1),
                M.BatchNorm2d(256),
                M.ReLU(),)
        self.pool3 = M.MaxPool2d(2, 2)
        # conv4_1: ConvBnRelu
        self.conv4_1 = M.Sequential(
                M.Conv2d(256, 512, 3, stride=1, padding=1),
                M.BatchNorm2d(512),
                M.ReLU(),)
        self.drop4_1 = M.Dropout(0.4)
        # conv4_2: ConvBnRelu
        self.conv4_2 = M.Sequential(
                M.Conv2d(512, 512, 3, stride=1, padding=1),
                M.BatchNorm2d(512),
                M.ReLU(),)
        self.drop4_2 = M.Dropout(0.4)
        # conv4_3: ConvBnRelu
        self.conv4_3 = M.Sequential(
                M.Conv2d(512, 512, 3, stride=1, padding=1),
                M.BatchNorm2d(512),
                M.ReLU(),)
        self.pool4 = M.MaxPool2d(2, 2)
        # conv5_1: ConvBnRelu
        self.conv5_1 = M.Sequential(
                M.Conv2d(512, 512, 3, stride=1, padding=1),
                M.BatchNorm2d(512),
                M.ReLU(),)
        self.drop5_1 = M.Dropout(0.4)
        # conv5_2: ConvBnRelu
        self.conv5_2 = M.Sequential(
                M.Conv2d(512, 512, 3, stride=1, padding=1),
                M.BatchNorm2d(512),
                M.ReLU(),)
        self.drop5_2 = M.Dropout(0.4)
        # conv5_3: ConvBnRelu
        self.conv5_3 = M.Sequential(
                M.Conv2d(512, 512, 3, stride=1, padding=1),
                M.BatchNorm2d(512),
                M.ReLU(),)
        self.pool5 = M.MaxPool2d(2, 2)
        self.drop_flat = M.Dropout(0.5)
        self.fc1 = M.Linear(512, 512)
        self.batch_fc1 = M.BatchNorm1d(512)
        self.relu_fc1 = M.ReLU()
        self.drop_fc2 = M.Dropout(0.5)
        self.classifier = M.Linear(512, 10)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.drop1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)
        x = self.conv2_1(x)
        x = self.drop2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)
        x = self.conv3_1(x)
        x = self.drop3_1(x)
        x = self.conv3_2(x)
        x = self.drop3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)
        x = self.conv4_1(x)
        x = self.drop4_1(x)
        x = self.conv4_2(x)
        x = self.drop4_2(x)
        x = self.conv4_3(x)
        x = self.pool4(x)
        x = self.conv5_1(x)
        x = self.drop5_1(x)
        x = self.conv5_2(x)
        x = self.drop5_2(x)
        x = self.conv5_3(x)
        x = self.pool5(x)
        x = F.flatten(x, 1)
        x = self.drop_flat(x)
        x = self.fc1(x)
        x = self.batch_fc1(x)
        x = self.relu_fc1(x)
        x = self.drop_fc2(x)
        x = self.classifier(x)
        return x
