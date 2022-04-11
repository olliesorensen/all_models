import argparse

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

parser = argparse.ArgumentParser(description='Multi-task/Auxiliary Learning: Dense Prediction Tasks')
parser.add_argument('--network', default='SegNet_split', type=str, help='SegNet_split, SegNet_mtan, ResNet_split, Resnet_mtan')
parser.add_argument('--dataset', default='nyuv2', type=str, help='nyuv2, cityscapes')
opt = parser.parse_args()

model_name = opt.network
data_set = opt.dataset

# Read txt file with results 
#file_name = "/Users/olemartinsorensen/Desktop/Thesis/Results/model_{model_name}_{dataset_name}.pth"
file_name = "/Users/olemartinsorensen/Desktop/Thesis/Results/WRONG_ONE.pth"

with open(file_name, "r") as a_file:
    for line in a_file:
        print(line)
