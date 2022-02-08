import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary as summary_

"""class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

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
        return x"""


class MyModel(nn.Module):
    """
    VGG 컨셉 : https://bskyvision.com/504
    """
    def __init__(self):
        super(MyModel, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(1024)

        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 3)

        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.AdaptiveAvgPool2d(1)
        self.act_fn = nn.ReLU()

        self.drop = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.conv1_1(x)         # (batch, 3, 320, 320) -> (batch, 64, 320, 320)
        x = self.bn1(x)
        x = self.act_fn(x)
        x = self.conv1_2(x)         # (batch, 64, 320, 320) -> (batch, 64, 320, 320)
        x = self.act_fn(x)
        x = self.pool(x)            # (batch, 64, 320, 320) -> (batch, 64, 160, 160)

        x = self.conv2_1(x)         # (batch, 64, 160, 160) -> (batch, 128, 160, 160)
        x = self.bn2(x)
        x = self.act_fn(x)
        x = self.conv2_2(x)         # (batch, 128, 160, 160) -> (batch, 128, 160, 160)
        x = self.act_fn(x)
        x = self.pool(x)            # (batch, 128, 160, 160) -> (batch, 128, 80, 80)

        x = self.conv3_1(x)         # (batch, 128, 80, 80) -> (batch, 256, 80, 80)
        x = self.bn3(x)
        x = self.act_fn(x)
        x = self.conv3_2(x)         # (batch, 128, 80, 80) -> (batch, 256, 80, 80)
        x = self.act_fn(x)
        x = self.conv3_3(x)         # (batch, 256, 80, 80) -> (batch, 256, 80, 80)
        x = self.act_fn(x)
        x = self.pool(x)            # (batch, 256, 80, 80) -> (batch, 256, 40, 40)

        x = self.conv4_1(x)         # (batch, 256, 40, 40) -> (batch, 512, 40, 40)
        x = self.bn4(x)
        x = self.act_fn(x)
        x = self.conv4_2(x)         # (batch, 512, 40, 40) -> (batch, 512, 40, 40)
        x = self.act_fn(x)
        x = self.conv4_3(x)         # (batch, 512, 40, 40) -> (batch, 512, 40, 40)
        x = self.act_fn(x)
        x = self.pool(x)            # (batch, 512, 40, 40) -> (batch, 512, 20, 20)

        x = self.conv5_1(x)         # (batch, 512, 20, 20) -> (batch, 1024, 20, 20)
        x = self.bn5(x)
        x = self.act_fn(x)
        x = self.conv5_2(x)         # (batch, 1024, 40, 40) -> (batch, 1024, 20, 20)
        x = self.act_fn(x)
        x = self.conv5_3(x)         # (batch, 1024, 40, 40) -> (batch, 1024, 20, 20)
        x = self.act_fn(x)
        x = self.pool(x)            # (batch, 1024, 40, 40) -> (batch, 1024, 10, 10)

        x = self.flatten(x)         # (batch, 1024, 10, 10) -> (batch, 1024, 1, 1)
        x = x.view(-1, 1024*1*1)    # (batch, 1024, 1, 1) -> (batch, 1024)

        x = self.fc1(x)  # (batch, 1024) -> (batch, 128)
        x = self.act_fn(x)
        x = self.drop(x)

        x = self.fc2(x)  # (batch, 128) -> (batch, 3)
        return x

def main():
    model = MyModel().cuda()

    summary_(model, (3,320,320), batch_size=16)

if __name__ == "__main__":
    main()