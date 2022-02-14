import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from learning import evaluate
from dataset import MyDataset
from model import get_Model

import pandas as pd


def inference(device, criterion, inference_loader):
    file_list = os.listdir(f"models/")

    for file in file_list:
        if file[-2:] != "pt":
            continue

        print("Inference {}".format(file))
        model = torch.load(f"models/"+file)
        model.to(device); model.eval()

        loss, acc = evaluate(model, device, criterion, inference_loader)
        print("\tloss : {:.6f}".format(loss))
        print("\tacc : {:.3f}".format(acc))
        print("\n")


def main():
    train_path = "datasets_labels.csv"
    train_data = pd.read_csv(train_path)
    train_x, train_y = train_data.iloc[:, :1], train_data.iloc[:, 1:]

    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, stratify=train_y, random_state=17, test_size=0.05)
    valid_x, test_x, valid_y, test_y = train_test_split(valid_x, valid_y, stratify=valid_y, random_state=7, test_size=0.1)

    train_data = pd.concat([train_x, train_y], axis=1)
    valid_data = pd.concat([valid_x, valid_y], axis=1)
    test_data = pd.concat([test_x, test_y], axis=1)

    train_dataset = MyDataset(train_data)
    valid_dataset = MyDataset(valid_data)
    test_dataset = MyDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = torch.nn.CrossEntropyLoss()

    inference(device, criterion, test_loader)


if __name__ == '__main__':
    main()