import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary as summary_

import numpy as np

class MyModel_1(nn.Module):
    def __init__(self):
        super(MyModel_1, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        self.bn = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 3)

        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.AdaptiveAvgPool2d(1)
        self.act_fn = nn.ReLU()

        self.drop = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.conv1(x)  # (batch, 3, 320, 320) -> (batch, 16, 320, 320)
        x = self.act_fn(x)
        x = self.pool(x)  # (batch, 16, 320, 320) -> (batch, 16, 160, 160)

        x = self.conv2(x)  # (batch, 16, 160, 160) -> (batch, 32, 160 ,160)
        x = self.bn(x)
        x = self.act_fn(x)
        x = self.pool(x)  # (batch, 32, 160, 160) -> (batch, 32, 80, 80)

        x = self.conv3(x)  # (batch, 32, 80, 80) -> (batch, 64, 80, 80)
        x = self.act_fn(x)
        x = self.pool(x)  # (batch, 64, 80, 80) -> (batch, 64, 40, 40)

        x = self.conv4(x)  # (batch, 64, 40, 40) -> (batch, 128, 40, 40)
        x = self.act_fn(x)
        x = self.pool(x)  # (batch, 128, 40, 40) -> (batch, 128, 20, 20)

        x = self.flatten(x)  # (batch, 128, 20, 20) -> (batch, 128, 1, 1)
        x = x.view(-1, 128 * 1 * 1)  # (batch, 128, 1, 1) -> (batch, 128)

        x = self.fc1(x)  # (batch, 128) -> (batch, 32)
        x = self.act_fn(x)
        x = self.drop(x)

        x = self.fc2(x)  # (batch, 32) -> (batch, 3)
        return x

class MyModel_2(nn.Module):
    def __init__(self):
        super(MyModel_2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        self.att1 = nn.Conv2d(16, 32, 3, 1, 1)
        self.att2 = nn.Conv2d(64, 128, 3, 1, 1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 3)

        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.AdaptiveAvgPool2d(1)
        self.act_fn = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.drop = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.conv1(x)  # (batch, 3, 320, 320) -> (batch, 16, 320, 320)
        x = self.act_fn(x)
        x = self.pool(x)   # (batch, 16, 320, 320) -> (batch, 16, 160, 160)

        att = self.att1(x) # (batch, 16, 160, 160) -> (batch, 32, 160, 160)
        x = self.conv2(x)  # (batch, 16, 160, 160) -> (batch, 32, 160 ,160)
        att = self.sigmoid(x)
        x = x*att
        x = self.bn1(x)
        x = self.act_fn(x)
        x = self.pool(x)   # (batch, 32, 160, 160) -> (batch, 32, 80, 80)

        x = self.conv3(x)  # (batch, 32, 80, 80) -> (batch, 64, 80, 80)
        x = self.act_fn(x)
        x = self.pool(x)   # (batch, 64, 80, 80) -> (batch, 64, 40, 40)

        att = self.att2(x)  # (batch, 64, 40, 40) -> (batch, 128, 40, 40)
        x = self.conv4(x)   # (batch, 64, 40, 40) -> (batch, 128, 40, 40)
        att = self.sigmoid(x)
        x = x*att
        x = self.bn2(x)
        x = self.act_fn(x)
        x = self.pool(x)    # (batch, 128, 40, 40) -> (batch, 128, 20, 20)

        x = self.flatten(x)  # (batch, 128, 20, 20) -> (batch, 128, 1, 1)
        x = x.view(-1, 128 * 1 * 1)  # (batch, 128, 1, 1) -> (batch, 128)

        x = self.fc1(x)     # (batch, 128) -> (batch, 32)
        x = self.act_fn(x)
        x = self.drop(x)

        x = self.fc2(x)     # (batch, 32) -> (batch, 3)
        return x

class MyModel_3(nn.Module):
    def __init__(self):
        super(MyModel_3, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        self.att1 = nn.Conv2d(16, 32, 3, 1, 1)
        self.att2 = nn.Conv2d(64, 128, 3, 1, 1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(128)

        self.Q = nn.Linear(4, 16)
        self.K = nn.Linear(4, 16)
        self.V = nn.Linear(4, 16)

        self.fc1 = nn.Linear(512, 32)
        self.fc2 = nn.Linear(32, 3)

        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.AdaptiveAvgPool2d(1)
        self.act_fn = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.drop = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.conv1(x)  # (batch, 3, 320, 320) -> (batch, 16, 320, 320)
        x = self.act_fn(x)
        x = self.pool(x)   # (batch, 16, 320, 320) -> (batch, 16, 160, 160)

        att = self.att1(x) # (batch, 16, 160, 160) -> (batch, 32, 160, 160)
        x = self.conv2(x)  # (batch, 16, 160, 160) -> (batch, 32, 160 ,160)
        att = self.sigmoid(x)
        x = x*att
        x = self.bn1(x)
        x = self.act_fn(x)
        x = self.pool(x)   # (batch, 32, 160, 160) -> (batch, 32, 80, 80)

        x = self.conv3(x)  # (batch, 32, 80, 80) -> (batch, 64, 80, 80)
        x = self.act_fn(x)
        x = self.pool(x)   # (batch, 64, 80, 80) -> (batch, 64, 40, 40)

        att = self.att2(x)  # (batch, 64, 40, 40) -> (batch, 128, 40, 40)
        x = self.conv4(x)   # (batch, 64, 40, 40) -> (batch, 128, 40, 40)
        att = self.sigmoid(x)
        x = x*att
        x = self.bn2(x)
        x = self.act_fn(x)
        x = self.pool(x)    # (batch, 128, 40, 40) -> (batch, 128, 20, 20)

        x = self.flatten(x)  # (batch, 128, 20, 20) -> (batch, 128, 1, 1)
        x = x.view(-1, 128 * 1 * 1)  # (batch, 128, 1, 1) -> (batch, 128)

        x = x.view(-1, 32, 4)       # (batch, 128) -> (batch, 32, 4)

        q = self.Q(x)               # (batch, 32, 4) -> (batch, 32, 16)
        k = self.V(x)               # (batch, 32, 4) -> (batch, 32, 16)
        v = self.K(x)               # (batch, 32, 4) -> (batch, 32, 16)

        # (batch, 32, 16) x (batch, 16, 32) = (batch, 32, 32)
        score = torch.matmul(q, torch.transpose(k, 1, 2)) / np.sqrt(16)
        score = F.softmax(score, dim=-1)

        x = torch.matmul(score, v)  # (batch, 32, 32) x (batch, 32, 16) = (batch, 32, 16)
        x = x.view(-1, 32*16)       # (batch, 32, 16) -> (batch, 512)

        x = self.fc1(x)             # (batch, 512) -> (batch, 32)
        x = self.act_fn(x)
        x = self.drop(x)

        x = self.fc2(x)             # (batch, 32) -> (batch, 3)
        return x

class MyModel_4(nn.Module):
    def __init__(self):
        super(MyModel_4, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        self.conv6_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)

        self.conv7_1 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.conv7_2 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)

        # self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        # self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(512)
        # self.bn7 = nn.BatchNorm2d(1024)

        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 3)

        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.AdaptiveAvgPool2d(1)
        self.act_fn = nn.ReLU()

        self.drop = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.conv1_1(x)         # (batch, 3, 320, 320) -> (batch, 16, 320, 320)
        # x = self.bn1(x)
        x = self.act_fn(x)
        x = self.conv1_2(x)         # (batch, 16, 320, 320) -> (batch, 16, 320, 320)
        x = self.act_fn(x)
        x = self.pool(x)            # (batch, 16, 320, 320) -> (batch, 16, 160, 160)

        x = self.conv2_1(x)         # (batch, 16, 160, 160) -> (batch, 32, 160, 160)
        x = self.bn2(x)
        x = self.act_fn(x)
        x = self.pool(x)            # (batch, 32, 160, 160) -> (batch, 32, 80, 80)

        x = self.conv3_1(x)         # (batch, 32, 80, 80) -> (batch, 64, 80, 80)
        x = self.bn3(x)
        x = self.act_fn(x)
        x = self.pool(x)            # (batch, 64, 80, 80) -> (batch, 64, 40, 40)

        x = self.conv4_1(x)         # (batch, 64, 40, 40) -> (batch, 128, 40, 40)
        # x = self.bn4(x)
        x = self.act_fn(x)
        x = self.conv4_2(x)         # (batch, 128, 40, 40) -> (batch, 128, 40, 40)
        x = self.act_fn(x)
        x = self.pool(x)            # (batch, 128, 40, 40) -> (batch, 128, 20, 20)

        x = self.conv5_1(x)         # (batch, 128, 20, 20) -> (batch, 256, 20, 20)
        x = self.bn5(x)
        x = self.act_fn(x)
        x = self.pool(x)            # (batch, 256, 20, 20) -> (batch, 256, 10, 10)

        x = self.conv6_1(x)         # (batch, 256, 10, 10) -> (batch, 512, 10, 10)
        x = self.bn6(x)
        x = self.act_fn(x)
        x = self.pool(x)            # (batch, 512, 10, 10) -> (batch, 512, 5, 5)

        x = self.conv7_1(x)         # (batch, 512, 5, 5) -> (batch, 1024, 5, 5)
        # x = self.bn7(x)
        x = self.act_fn(x)
        x = self.conv7_2(x)         # (batch, 1024, 5, 5) -> (batch, 1024, 5, 5)
        x = self.act_fn(x)
        x = self.pool(x)            # (batch, 1024, 5, 5) -> (batch, 1024, 2, 2)

        x = self.flatten(x)         # (batch, 1024, 2, 2) -> (batch, 1024, 1, 1)
        x = x.view(-1, 1024*1*1)    # (batch, 1024, 1, 1) -> (batch, 1024)

        x = self.fc1(x)  # (batch, 1024) -> (batch, 128)
        x = self.act_fn(x)
        x = self.drop(x)

        x = self.fc2(x)  # (batch, 128) -> (batch, 3)
        return x

class MyModel_5(nn.Module):
    def __init__(self):
        super(MyModel_5, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        self.conv6_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)

        self.conv7_1 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.conv7_2 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)

        # self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        # self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(512)
        # self.bn7 = nn.BatchNorm2d(1024)

        self.att1 = nn.Conv2d(16, 32, 3, 1, 1)
        self.att2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.att3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.att4 = nn.Conv2d(256, 512, 3, 1, 1)

        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 3)

        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.AdaptiveAvgPool2d(1)
        self.act_fn = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.drop = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.conv1_1(x)         # (batch, 3, 320, 320) -> (batch, 16, 320, 320)
        # x = self.bn1(x)
        x = self.act_fn(x)
        x = self.conv1_2(x)         # (batch, 16, 320, 320) -> (batch, 16, 320, 320)
        x = self.act_fn(x)
        x = self.pool(x)            # (batch, 16, 320, 320) -> (batch, 16, 160, 160)

        att = self.att1(x)          # (batch, 16, 160, 160) -> (batch, 32, 160, 160)
        att = self.sigmoid(att)
        x = self.conv2_1(x)         # (batch, 16, 160, 160) -> (batch, 32, 160, 160)
        x = x * att
        x = self.bn2(x)
        x = self.act_fn(x)
        x = self.pool(x)            # (batch, 32, 160, 160) -> (batch, 32, 80, 80)

        att = self.att2(x)          # (batch, 32, 80, 80) -> (batch, 64, 80, 80)
        att = self.sigmoid(att)
        x = self.conv3_1(x)         # (batch, 32, 80, 80) -> (batch, 64, 80, 80)
        x = x * att
        x = self.bn3(x)
        x = self.act_fn(x)
        x = self.pool(x)            # (batch, 64, 80, 80) -> (batch, 64, 40, 40)

        x = self.conv4_1(x)         # (batch, 64, 40, 40) -> (batch, 128, 40, 40)
        # x = self.bn4(x)
        x = self.act_fn(x)
        x = self.conv4_2(x)         # (batch, 128, 40, 40) -> (batch, 128, 40, 40)
        x = self.act_fn(x)
        x = self.pool(x)            # (batch, 128, 40, 40) -> (batch, 128, 20, 20)

        att = self.att3(x)          # (batch, 128, 20, 20) -> (batch, 256, 20, 20)
        att = self.sigmoid(att)
        x = self.conv5_1(x)         # (batch, 128, 20, 20) -> (batch, 256, 20, 20)
        x = x * att
        x = self.bn5(x)
        x = self.act_fn(x)
        x = self.pool(x)            # (batch, 256, 20, 20) -> (batch, 256, 10, 10)

        att = self.att4(x)          # (batch, 256, 10, 10) -> (batch, 512, 10, 10)
        att = self.sigmoid(att)
        x = self.conv6_1(x)         # (batch, 256, 10, 10) -> (batch, 512, 10, 10)
        x = x* att
        x = self.bn6(x)
        x = self.act_fn(x)
        x = self.pool(x)            # (batch, 512, 10, 10) -> (batch, 512, 5, 5)

        x = self.conv7_1(x)         # (batch, 512, 5, 5) -> (batch, 1024, 5, 5)
        # x = self.bn7(x)
        x = self.act_fn(x)
        x = self.conv7_2(x)         # (batch, 1024, 5, 5) -> (batch, 1024, 5, 5)
        x = self.act_fn(x)
        x = self.pool(x)            # (batch, 1024, 5, 5) -> (batch, 1024, 2, 2)

        x = self.flatten(x)         # (batch, 1024, 2, 2) -> (batch, 1024, 1, 1)
        x = x.view(-1, 1024*1*1)    # (batch, 1024, 1, 1) -> (batch, 1024)

        x = self.fc1(x)  # (batch, 1024) -> (batch, 128)
        x = self.act_fn(x)
        x = self.drop(x)

        x = self.fc2(x)  # (batch, 128) -> (batch, 3)
        return x

class MyModel_6(nn.Module):
    def __init__(self):
        super(MyModel_6, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        self.conv6_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)

        self.conv7_1 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.conv7_2 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)

        # self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        # self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(512)
        # self.bn7 = nn.BatchNorm2d(1024)

        self.att1 = nn.Conv2d(16, 32, 3, 1, 1)
        self.att2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.att3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.att4 = nn.Conv2d(256, 512, 3, 1, 1)

        self.Q = nn.Linear(32, 16)
        self.V = nn.Linear(32, 16)
        self.K = nn.Linear(32, 16)

        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 3)

        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.AdaptiveAvgPool2d(1)
        self.act_fn = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.drop = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.conv1_1(x)         # (batch, 3, 320, 320) -> (batch, 16, 320, 320)
        # x = self.bn1(x)
        x = self.act_fn(x)
        x = self.conv1_2(x)         # (batch, 16, 320, 320) -> (batch, 16, 320, 320)
        x = self.act_fn(x)
        x = self.pool(x)            # (batch, 16, 320, 320) -> (batch, 16, 160, 160)

        att = self.att1(x)          # (batch, 16, 160, 160) -> (batch, 32, 160, 160)
        att = self.sigmoid(att)
        x = self.conv2_1(x)         # (batch, 16, 160, 160) -> (batch, 32, 160, 160)
        x = x * att
        x = self.bn2(x)
        x = self.act_fn(x)
        x = self.pool(x)            # (batch, 32, 160, 160) -> (batch, 32, 80, 80)

        att = self.att2(x)          # (batch, 32, 80, 80) -> (batch, 64, 80, 80)
        att = self.sigmoid(att)
        x = self.conv3_1(x)         # (batch, 32, 80, 80) -> (batch, 64, 80, 80)
        x = x * att
        x = self.bn3(x)
        x = self.act_fn(x)
        x = self.pool(x)            # (batch, 64, 80, 80) -> (batch, 64, 40, 40)

        x = self.conv4_1(x)         # (batch, 64, 40, 40) -> (batch, 128, 40, 40)
        # x = self.bn4(x)
        x = self.act_fn(x)
        x = self.conv4_2(x)         # (batch, 128, 40, 40) -> (batch, 128, 40, 40)
        x = self.act_fn(x)
        x = self.pool(x)            # (batch, 128, 40, 40) -> (batch, 128, 20, 20)

        att = self.att3(x)          # (batch, 128, 20, 20) -> (batch, 256, 20, 20)
        att = self.sigmoid(att)
        x = self.conv5_1(x)         # (batch, 128, 20, 20) -> (batch, 256, 20, 20)
        x = x * att
        x = self.bn5(x)
        x = self.act_fn(x)
        x = self.pool(x)            # (batch, 256, 20, 20) -> (batch, 256, 10, 10)

        att = self.att4(x)          # (batch, 256, 10, 10) -> (batch, 512, 10, 10)
        att = self.sigmoid(att)
        x = self.conv6_1(x)         # (batch, 256, 10, 10) -> (batch, 512, 10, 10)
        x = x* att
        x = self.bn6(x)
        x = self.act_fn(x)
        x = self.pool(x)            # (batch, 512, 10, 10) -> (batch, 512, 5, 5)

        x = self.conv7_1(x)         # (batch, 512, 5, 5) -> (batch, 1024, 5, 5)
        # x = self.bn7(x)
        x = self.act_fn(x)
        x = self.conv7_2(x)         # (batch, 1024, 5, 5) -> (batch, 1024, 5, 5)
        x = self.act_fn(x)
        x = self.pool(x)            # (batch, 1024, 5, 5) -> (batch, 1024, 2, 2)

        x = self.flatten(x)         # (batch, 1024, 2, 2) -> (batch, 1024, 1, 1)
        x = x.view(-1, 1024*1*1)    # (batch, 1024, 1, 1) -> (batch, 1024)

        x = x.view(-1, 32, 32)       # (batch, 1024) -> (batch, 32, 32)

        q = self.Q(x)               # (batch, 32, 32) -> (batch, 32, 16)
        k = self.V(x)               # (batch, 32, 32) -> (batch, 32, 16)
        v = self.K(x)               # (batch, 32, 32) -> (batch, 32, 16)

        # (batch, 32, 16) x (batch, 16, 32) = (batch, 32, 32)
        score = torch.matmul(q, torch.transpose(k, 1, 2)) / np.sqrt(16)
        score = F.softmax(score, dim=-1)

        x = torch.matmul(score, v)  # (batch, 32, 32) x (batch, 32, 16) = (batch, 32, 16)
        x = x.view(-1, 32*16)       # (batch, 32, 16) -> (batch, 512)

        x = self.fc1(x)  # (batch, 512) -> (batch, 64)
        x = self.act_fn(x)
        x = self.drop(x)

        x = self.fc2(x)  # (batch, 128) -> (batch, 3)
        return x

class MyModel_7(nn.Module):
    def __init__(self):
        super(MyModel_7, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        self.conv6_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)

        self.conv7_1 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.conv7_2 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(512)
        self.bn7 = nn.BatchNorm2d(1024)

        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 3)

        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.AdaptiveAvgPool2d(1)
        self.act_fn = nn.ReLU()

        self.drop = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.conv1_1(x)         # (batch, 3, 320, 320) -> (batch, 16, 320, 320)
        out = self.bn1(x)
        out = self.act_fn(out)
        out = self.conv1_2(out)       # (batch, 16, 320, 320) -> (batch, 16, 320, 320)
        out += x
        out = self.act_fn(out)
        x = self.pool(out)            # (batch, 16, 320, 320) -> (batch, 16, 160, 160)

        x = self.conv2_1(x)         # (batch, 16, 160, 160) -> (batch, 32, 160, 160)
        x = self.bn2(x)
        x = self.act_fn(x)
        x = self.pool(x)            # (batch, 32, 160, 160) -> (batch, 32, 80, 80)

        x = self.conv3_1(x)         # (batch, 32, 80, 80) -> (batch, 64, 80, 80)
        x = self.bn3(x)
        x = self.act_fn(x)
        x = self.pool(x)            # (batch, 64, 80, 80) -> (batch, 64, 40, 40)

        x = self.conv4_1(x)         # (batch, 64, 40, 40) -> (batch, 128, 40, 40)
        out = self.bn4(x)
        out = self.act_fn(out)
        out = self.conv4_2(out)     # (batch, 128, 40, 40) -> (batch, 128, 40, 40)
        out += x
        out = self.act_fn(out)
        x = self.pool(out)          # (batch, 128, 40, 40) -> (batch, 128, 20, 20)

        x = self.conv5_1(x)         # (batch, 128, 20, 20) -> (batch, 256, 20, 20)
        x = self.bn5(x)
        x = self.act_fn(x)
        x = self.pool(x)            # (batch, 256, 20, 20) -> (batch, 256, 10, 10)

        x = self.conv6_1(x)         # (batch, 256, 10, 10) -> (batch, 512, 10, 10)
        x = self.bn6(x)
        x = self.act_fn(x)
        x = self.pool(x)            # (batch, 512, 10, 10) -> (batch, 512, 5, 5)

        x = self.conv7_1(x)         # (batch, 512, 5, 5) -> (batch, 1024, 5, 5)
        out = self.bn7(x)
        out = self.act_fn(out)
        out = self.conv7_2(out)     # (batch, 1024, 5, 5) -> (batch, 1024, 5, 5)
        out += x
        out = self.act_fn(out)
        x = self.pool(out)          # (batch, 1024, 5, 5) -> (batch, 1024, 2, 2)

        x = self.flatten(x)         # (batch, 1024, 2, 2) -> (batch, 1024, 1, 1)
        x = x.view(-1, 1024*1*1)    # (batch, 1024, 1, 1) -> (batch, 1024)

        x = self.fc1(x)  # (batch, 1024) -> (batch, 128)
        x = self.act_fn(x)
        x = self.drop(x)

        x = self.fc2(x)  # (batch, 128) -> (batch, 3)
        return x

class MyModel_8(nn.Module):
    def __init__(self):
        super(MyModel_8, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        self.conv6_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)

        self.conv7_1 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.conv7_2 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(512)
        self.bn7 = nn.BatchNorm2d(1024)

        self.att1 = nn.Conv2d(16, 32, 3, 1, 1)
        self.att2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.att3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.att4 = nn.Conv2d(256, 512, 3, 1, 1)

        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 3)

        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.AdaptiveAvgPool2d(1)
        self.act_fn = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.drop = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.conv1_1(x)         # (batch, 3, 320, 320) -> (batch, 16, 320, 320)
        out = self.bn1(x)
        out = self.act_fn(out)
        out = self.conv1_2(out)       # (batch, 16, 320, 320) -> (batch, 16, 320, 320)
        out += x
        out = self.act_fn(out)
        x = self.pool(out)            # (batch, 16, 320, 320) -> (batch, 16, 160, 160)

        att = self.att1(x)          # (batch, 16, 160, 160) -> (batch, 32, 160, 160)
        att = self.sigmoid(att)
        x = self.conv2_1(x)         # (batch, 16, 160, 160) -> (batch, 32, 160, 160)
        x = x * att
        x = self.bn2(x)
        x = self.act_fn(x)
        x = self.pool(x)            # (batch, 32, 160, 160) -> (batch, 32, 80, 80)

        att = self.att2(x)          # (batch, 32, 80, 80) -> (batch, 64, 80, 80)
        att = self.sigmoid(att)
        x = self.conv3_1(x)         # (batch, 32, 80, 80) -> (batch, 64, 80, 80)
        x = x*att
        x = self.bn3(x)
        x = self.act_fn(x)
        x = self.pool(x)            # (batch, 64, 80, 80) -> (batch, 64, 40, 40)

        x = self.conv4_1(x)         # (batch, 64, 40, 40) -> (batch, 128, 40, 40)
        out = self.bn4(x)
        out = self.act_fn(out)
        out = self.conv4_2(out)     # (batch, 128, 40, 40) -> (batch, 128, 40, 40)
        out += x
        out = self.act_fn(out)
        x = self.pool(out)          # (batch, 128, 40, 40) -> (batch, 128, 20, 20)

        att = self.att3(x)          # (batch, 128, 20, 20) -> (batch, 256, 20, 20)
        att = self.sigmoid(att)
        x = self.conv5_1(x)         # (batch, 128, 20, 20) -> (batch, 256, 20, 20)
        x = x * att
        x = self.bn5(x)
        x = self.act_fn(x)
        x = self.pool(x)            # (batch, 256, 20, 20) -> (batch, 256, 10, 10)

        att = self.att4(x)          # (batch, 256, 10, 10) -> (batch, 512, 10, 10)
        att = self.sigmoid(att)
        x = self.conv6_1(x)         # (batch, 256, 10, 10) -> (batch, 512, 10, 10)
        x = x * att
        x = self.bn6(x)
        x = self.act_fn(x)
        x = self.pool(x)            # (batch, 512, 10, 10) -> (batch, 512, 5, 5)

        x = self.conv7_1(x)         # (batch, 512, 5, 5) -> (batch, 1024, 5, 5)
        out = self.bn7(x)
        out = self.act_fn(out)
        out = self.conv7_2(out)     # (batch, 1024, 5, 5) -> (batch, 1024, 5, 5)
        out += x
        out = self.act_fn(out)
        x = self.pool(out)          # (batch, 1024, 5, 5) -> (batch, 1024, 2, 2)

        x = self.flatten(x)         # (batch, 1024, 2, 2) -> (batch, 1024, 1, 1)
        x = x.view(-1, 1024*1*1)    # (batch, 1024, 1, 1) -> (batch, 1024)

        x = self.fc1(x)  # (batch, 1024) -> (batch, 128)
        x = self.act_fn(x)
        x = self.drop(x)

        x = self.fc2(x)  # (batch, 128) -> (batch, 3)
        return x

class MyModel_9(nn.Module):
    def __init__(self):
        super(MyModel_9, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        self.conv6_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)

        self.conv7_1 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.conv7_2 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(512)
        self.bn7 = nn.BatchNorm2d(1024)

        self.att1 = nn.Conv2d(16, 32, 3, 1, 1)
        self.att2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.att3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.att4 = nn.Conv2d(256, 512, 3, 1, 1)

        self.Q = nn.Linear(32, 16)
        self.V = nn.Linear(32, 16)
        self.K = nn.Linear(32, 16)

        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 3)

        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.AdaptiveAvgPool2d(1)
        self.act_fn = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.drop = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.conv1_1(x)         # (batch, 3, 320, 320) -> (batch, 16, 320, 320)
        out = self.bn1(x)
        out = self.act_fn(out)
        out = self.conv1_2(out)       # (batch, 16, 320, 320) -> (batch, 16, 320, 320)
        out += x
        out = self.act_fn(out)
        x = self.pool(out)            # (batch, 16, 320, 320) -> (batch, 16, 160, 160)

        att = self.att1(x)          # (batch, 16, 160, 160) -> (batch, 32, 160, 160)
        att = self.sigmoid(att)
        x = self.conv2_1(x)         # (batch, 16, 160, 160) -> (batch, 32, 160, 160)
        x = x * att
        x = self.bn2(x)
        x = self.act_fn(x)
        x = self.pool(x)            # (batch, 32, 160, 160) -> (batch, 32, 80, 80)

        att = self.att2(x)          # (batch, 32, 80, 80) -> (batch, 64, 80, 80)
        att = self.sigmoid(att)
        x = self.conv3_1(x)         # (batch, 32, 80, 80) -> (batch, 64, 80, 80)
        x = x*att
        x = self.bn3(x)
        x = self.act_fn(x)
        x = self.pool(x)            # (batch, 64, 80, 80) -> (batch, 64, 40, 40)

        x = self.conv4_1(x)         # (batch, 64, 40, 40) -> (batch, 128, 40, 40)
        out = self.bn4(x)
        out = self.act_fn(out)
        out = self.conv4_2(out)     # (batch, 128, 40, 40) -> (batch, 128, 40, 40)
        out += x
        out = self.act_fn(out)
        x = self.pool(out)          # (batch, 128, 40, 40) -> (batch, 128, 20, 20)

        att = self.att3(x)          # (batch, 128, 20, 20) -> (batch, 256, 20, 20)
        att = self.sigmoid(att)
        x = self.conv5_1(x)         # (batch, 128, 20, 20) -> (batch, 256, 20, 20)
        x = x * att
        x = self.bn5(x)
        x = self.act_fn(x)
        x = self.pool(x)            # (batch, 256, 20, 20) -> (batch, 256, 10, 10)

        att = self.att4(x)          # (batch, 256, 10, 10) -> (batch, 512, 10, 10)
        att = self.sigmoid(att)
        x = self.conv6_1(x)         # (batch, 256, 10, 10) -> (batch, 512, 10, 10)
        x = x * att
        x = self.bn6(x)
        x = self.act_fn(x)
        x = self.pool(x)            # (batch, 512, 10, 10) -> (batch, 512, 5, 5)

        x = self.conv7_1(x)         # (batch, 512, 5, 5) -> (batch, 1024, 5, 5)
        out = self.bn7(x)
        out = self.act_fn(out)
        out = self.conv7_2(out)     # (batch, 1024, 5, 5) -> (batch, 1024, 5, 5)
        out += x
        out = self.act_fn(out)
        x = self.pool(out)          # (batch, 1024, 5, 5) -> (batch, 1024, 2, 2)

        x = self.flatten(x)         # (batch, 1024, 2, 2) -> (batch, 1024, 1, 1)
        x = x.view(-1, 1024*1*1)    # (batch, 1024, 1, 1) -> (batch, 1024)

        x = x.view(-1, 32, 32)       # (batch, 1024) -> (batch, 32, 32)

        q = self.Q(x)               # (batch, 32, 32) -> (batch, 32, 16)
        k = self.V(x)               # (batch, 32, 32) -> (batch, 32, 16)
        v = self.K(x)               # (batch, 32, 32) -> (batch, 32, 16)

        # (batch, 32, 16) x (batch, 16, 32) = (batch, 32, 32)
        score = torch.matmul(q, torch.transpose(k, 1, 2)) / np.sqrt(16)
        score = F.softmax(score, dim=-1)

        x = torch.matmul(score, v)  # (batch, 32, 32) x (batch, 32, 16) = (batch, 32, 16)
        x = x.view(-1, 32*16)       # (batch, 32, 16) -> (batch, 512)

        x = self.fc1(x)  # (batch, 512) -> (batch, 64)
        x = self.act_fn(x)
        x = self.drop(x)

        x = self.fc2(x)  # (batch, 64) -> (batch, 3)
        return x

def get_Model(class_name):
    try:
        Myclass = eval(class_name)()
        return Myclass
    except NameError as e:
        print("Class [{}] is not defined".format(class_name))

def main():
    model = get_Model("MyModel_9").cuda()

    summary_(model, input_size = (3, 320, 320), device = "cuda")

if __name__ == "__main__":
    main()