import argparse

import pandas as pd 
import numpy as np

from utils import create_task_flags
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Multi-task/Auxiliary Learning: Dense Prediction Tasks')
parser.add_argument('--network', default='SegNet_split', type=str, help='SegNet_split, SegNet_mtan, ResNet_split, Resnet_mtan')
parser.add_argument('--dataset', default='nyuv2', type=str, help='nyuv2, cityscapes')
opt = parser.parse_args()

model_name = opt.network
data_set = opt.dataset

tasks = create_task_flags("all", opt.dataset)
tasks = list(tasks.keys())
print(tasks)

# Read txt file with results 
file_name = "/Users/olemartinsorensen/Desktop/Thesis/Results/model_{model_name}_{dataset_name}.rtf"

train_task1 = []
train_task2 = []
train_task3 = []
test_task1 = []
test_task2 = []
test_task3 = []

# Iterate through text file
with open(file_name, "r") as a_file:
    for line in a_file:
        if "Epoch" in line:
            words = line.split(" ")
            train_task1.append(float(words[7]))
            train_task2.append(float(words[10]))
            train_task3.append(float(words[13]))
            test_task1.append(float(words[18]))
            test_task2.append(float(words[21]))
            test_task3.append(float(words[24]))


# Create subplots
fig, axarr = plt.subplots(1, 3)
axarr[0].plot(train_task1)
axarr[0].plot(test_task1)
axarr[0].set_title(tasks[0])
axarr[1].plot(train_task2)
axarr[1].plot(test_task2)
axarr[1].set_title(tasks[1])
axarr[2].plot(train_task3)
axarr[2].plot(test_task3)
axarr[2].set_title(tasks[2])

plt.show()




