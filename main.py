# https://github.com/HideOnHouse/TorchBase

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from dataset import MyDataset
from model import MyModel
from learning import train, evaluate, calc_acc
from inference import inference


def main():
    train_path = "Train_datasets_labels.csv"
    test_path = "Validation_datasets_labels.csv"

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # your Data Pre-Processing
    train_x, train_y = train_data.iloc[:, :1], train_data.iloc[:, 1:]
    test_x, test_y = test_data.iloc[:, :1], test_data.iloc[:, 1:]

    # data split
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, stratify=train_y, random_state=17, test_size=0.1)

    train_data = pd.concat([train_x, train_y], axis=1)
    valid_data = pd.concat([valid_x, valid_y], axis=1)

    # Check Train, Valid, Test Data's Shape
    print("The Shape of Train Data: ", train_data.shape)
    print("The Shape of Valid Data: ", valid_data.shape)
    print("The Shape of Test Data: ", test_data.shape, end="\n\n")

    # Create Dataset and DataLoader
    train_dataset = MyDataset(train_data)
    valid_dataset = MyDataset(valid_data)
    test_dataset = MyDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # label_tags
    label_tags = ["포트홀 없음", "포트홀", "보수 완료된 포트홀"]

    model = MyModel()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optimizer = torch.optim.AdamW(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    # train
    print("============================== Train ==============================")
    history = train(model, device, optimizer, criterion, 20, train_loader, valid_loader)

    # Test
    print("============================== Test ==============================")
    test_loss, test_acc = evaluate(model, device, criterion, test_loader)

    torch.save(model, "models/model1.pt")


if __name__ == '__main__':
    main()