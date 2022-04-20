import torch
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from torchsummary import summary

from model_ResNet import MTANDeepLabv3, MTLDeepLabv3
from model_SegNet import SegNetMTAN, SegNetSplit
from model_EdgeSegNet import EdgeSegNet
from create_dataset import *
from utils import *


""" Script for testing various networks at inference time. All timings are given in milliseconds """

parser = argparse.ArgumentParser(description='Multi-task/Auxiliary Learning: Dense Prediction Tasks')
parser.add_argument('--network', default='SegNet_split', type=str, help='SegNet_split, SegNet_mtan, ResNet_split, Resnet_mtan')
parser.add_argument('--dataset', default='nyuv2', type=str, help='nyuv2, cityscapes')
opt = parser.parse_args()


if __name__ == "__main__":
    model_name = opt.network
    data_set = opt.dataset

    batch_size = 1

    # Define tasks based on dataset
    train_tasks = create_task_flags('all', opt.dataset, with_noise=False)

    # Load CPU
    device = torch.device("cpu")
    print("Device: ", device)

    # Load model
    if opt.network == 'ResNet_split':
        model = MTLDeepLabv3(train_tasks).to(device)
    elif opt.network == 'ResNet_mtan':
        model = MTANDeepLabv3(train_tasks).to(device)
    elif opt.network == "SegNet_split":
        model = SegNetSplit(train_tasks).to(device)
    elif opt.network == "SegNet_mtan":
        model = SegNetMTAN(train_tasks).to(device)
    elif opt.network == "EdgeSegNet":
        model = EdgeSegNet(train_tasks).to(device)   

    model.load_state_dict(torch.load(f"models/model_{model_name}_{data_set}.pth", map_location=device))

    summary(model)
