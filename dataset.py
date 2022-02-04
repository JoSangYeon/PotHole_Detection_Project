import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import os

class MyDataset(Dataset):
    def __init__(self, data, num_classes=3):
        super(MyDataset, self).__init__()

        self.data = data
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data.iloc[idx][0]

        feature = np.array(PIL.Image.open(file_path)) / 255.
        label = torch.tensor(self.data.iloc[idx][1])

        feature = np.transpose(torch.tensor(feature).float(), (2,0,1))
        label = F.one_hot(label, num_classes=self.num_classes).float()

        return feature, label

    def show_item(self, idx=0):
        feature, label = self.__getitem__(idx)

        print("Feature's Shape : {}".format(feature.shape))
        print("Label's Shape : {}".format(label.shape))

        return feature, label


def main():
    train_data = pd.read_csv("Train_datasets_labels.csv")
    valid_data = pd.read_csv("Validation_datasets_labels.csv")
    label_tags = ["포트홀 없음", "포트홀", "보수 완료된 포트홀"]

    train = MyDataset(train_data)
    train_loader = DataLoader(train, batch_size=32)

    valid = MyDataset(valid_data)
    valid_loader = DataLoader(valid, batch_size=32)

    print(train.show_item(0))

    img, label = valid.show_item(147)
    print("\nLabel :", label_tags[label.max(0)[1]])
    plt.imshow(np.transpose(img, (1,2,0)))
    plt.show()

if __name__ == "__main__":
    main()