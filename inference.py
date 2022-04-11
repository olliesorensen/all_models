from cgi import test
from configparser import Interpolation

import torch
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from model_ResNet import MTANDeepLabv3, MTLDeepLabv3
from model_SegNet import SegNetMTAN, SegNetSplit
from create_dataset import *
from utils import *

parser = argparse.ArgumentParser(description='Multi-task/Auxiliary Learning: Dense Prediction Tasks')
parser.add_argument('--network', default='SegNet_split', type=str, help='SegNet_split, SegNet_mtan, ResNet_split, Resnet_mtan')
parser.add_argument('--dataset', default='nyuv2', type=str, help='nyuv2, cityscapes')
opt = parser.parse_args()

model_name = opt.network
data_set = opt.dataset

batch_size = 1

# Define tasks based on dataset
train_tasks = create_task_flags('all', opt.dataset, with_noise=False)

# Define dataset
if opt.dataset == 'nyuv2':
    dataset_path = 'dataset/nyuv2'
    test_set = NYUv2(root=dataset_path, train=False)
    batch_size = 1

elif opt.dataset == 'cityscapes':
    dataset_path = 'dataset/cityscapes'
    test_set = CityScapes(root=dataset_path, train=False)
    batch_size = 1

test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    shuffle=False
)

# Load CUDA
device = torch.device("cuda")

# Load model
if opt.network == 'ResNet_split':
        model = MTLDeepLabv3(train_tasks).to(device)
    elif opt.network == 'ResNet_mtan':
        model = MTANDeepLabv3(train_tasks).to(device)
    elif opt.network == "SegNet_split":
        model = SegNetSplit(train_tasks).to(device)
    elif opt.network == "SegNet_mtan":
        model = SegNetMTAN(train_tasks).to(device)   

model.load_state_dict(torch.load(f"model_{model_name}_{data_set}.pth", map_location=device))
model.eval()

# Create iteratable object
test_dataset = iter(test_loader)

# Define start event, end event and number of repetitions
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 300
timings = np.zeros((repetitions, 1))

# GPU warm-up
for _ in range(10):
    test_data, test_target = test_dataset.next()
    test_data = test_data.to(device)
    test_target = {task_id: test_target[task_id].to(device) for task_id in train_tasks.keys()}
    _ = model(test_data)

# Measure performance 
with torch.no_grad():
    for rep in range(repetitions):
        test_data, test_target = test_dataset.next()
        test_data = test_data.to(device)
        test_target = {task_id: test_target[task_id].to(device) for task_id in train_tasks.keys()}
        starter.record()
        _ = model(test_data)
        ender.record()

        # Wait for GPU to sync
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time

mean_time = np.sum(timings) / repetitions
std_time = np.std(timings)

print("All times given in milliseconds")
print(f"Mean time: {mean_time}")
print(f"std: {std_time}")

print("All timings:")
print(timings)