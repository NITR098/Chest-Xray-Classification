import argparse
import logging
import os
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
import pandas as pd

import medmnist
from medmnist import INFO
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torchvision import transforms as T
from torchvision.utils import make_grid

import matplotlib as mpl
mpl.rcParams["text.usetex"] = True
import matplotlib.pyplot as plt

from dataloader import ChestXray14, JSRT
from run_classification import run_experiments
from models import build_model
from utils import seed_it_all, train_one_epoch, validation, save_checkpoint, my_transform
from utils import plot_performance, test_model, test_classification, metric_AUROC

from torchinfo import summary
from sklearn.metrics import accuracy_score



parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="ChestMNIST",
                    help="ChestXray14|JSRT|ChestMNIST")
parser.add_argument("--model_name", type = str, default="resnet18", 
                    help="swin_base|swin_tiny|resnet18|resnet50")
parser.add_argument("--isinit", type=str, default="ImageNet",
                    help="False for Random| True for ImageNet")
parser.add_argument("--normalization", type=str, default="imagenet", 
                    help="how to normalize data (imagenet|chestx-ray)")
parser.add_argument('--num_classes', type=int,
                    default=14, help='number of labels')
parser.add_argument('--output_dir', type=str,
                    help='output dir')
parser.add_argument('--max_epochs', type=int, default=2,
                    help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='classification network learning rate')
parser.add_argument('--img_size', type=int, default=224,
                    help='input patch size of network input')
parser.add_argument('--seed', type=int, default=1234,
                    help='random seed')
parser.add_argument("--exp_name", type=str, default="test",
                    help="experiment name")
parser.add_argument("--num_trial", type=int, default=2,
                    help="number of trials")
parser.add_argument("--device", type=str, default="cuda",
                    help="cpu|cuda")
parser.add_argument("--train_list", type=str, default=None,
                    help="file for training list")
parser.add_argument("--val_list", type=str, default=None,
                    help="file for validation list")
parser.add_argument("--test_list", type=str, default=None,
                    help="file for test list")
parser.add_argument("--in_chans", type=int, default=1, 
                    help="input data channel numbers")
parser.add_argument("--dataset_path", type=str, default="./images",
                    help="dataset path")

args = parser.parse_args()




if __name__ == "__main__":
    
    args.init = "ImageNet" if args.isinit=="ImageNet" else "Random"
    args.exp_name = args.model_name + "_" + args.init + "_" + args.exp_name
    model_path = Path("./Models").joinpath(args.dataset_name, args.exp_name)
    output_path = Path("./Outputs").joinpath(args.dataset_name, args.exp_name)
    model_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)
    
    seed_it_all(args.seed)
    
    if args.device == "cuda":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    data_flag = args.dataset_name.lower()
    info = INFO[data_flag]
    task = info["task"]
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    samples = info["n_samples"]
    DataClass = getattr(medmnist, info["python_class"])
    print(DataClass)

    size = 224
    data_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[.5], std=[.5]),
        T.Resize((size, size))
    ])
    train_set = DataClass(split="train", transform=data_transform ,download=True)
    val_set = DataClass(split="val", transform=data_transform, download=True)
    test_set = DataClass(split="test", transform=data_transform, download=True)
    
    _, mini_train_set = random_split(val_set, (0.8, 0.2))
    
    train_loader = DataLoader(dataset=mini_train_set, batch_size=16, shuffle=True)
    
    run_experiments(args, train_loader, train_loader, train_loader, model_path, output_path)