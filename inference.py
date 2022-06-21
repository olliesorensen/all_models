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
from model_EdgeSegNet import EdgeSegNet
from model_GuideDepth import GuideDepth
from model_DDRNet import DualResNet, BasicBlock
from create_dataset import *
from utils import *


""" Script for testing various networks at inference time. All timings are given in milliseconds """


def inference_test(model_name, data_set):

    batch_size = 1

    # Define tasks based on dataset
    train_tasks = create_task_flags('all', data_set)

    # Define dataset
    if data_set == 'nyuv2':
        dataset_path = 'dataset/nyuv2'
        test_set = NYUv2(root=dataset_path, train=False)
        batch_size = 1

    elif data_set == 'cityscapes':
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
    if model_name == 'ResNet_split':
        model = MTLDeepLabv3(train_tasks).to(device)
    elif model_name == 'ResNet_mtan':
        model = MTANDeepLabv3(train_tasks).to(device)
    elif model_name == "SegNet_split":
        model = SegNetSplit(train_tasks).to(device)
    elif model_name == "SegNet_mtan":
        model = SegNetMTAN(train_tasks).to(device)
    elif model_name == "EdgeSegNet":
        model = EdgeSegNet(train_tasks).to(device)
    elif model_name == "GuidedDepth":
        model = GuideDepth(train_tasks).to(device) 
    elif model_name == "DDRNet":
        model = DualResNet(BasicBlock, [2, 2, 2, 2], train_tasks, planes=32, spp_planes=128, head_planes=64).to(device)   

    # model.load_state_dict(torch.load(f"models/model_{model_name}_{data_set}.pth", map_location=device))
    model.eval()

    # Create iteratable object
    test_dataset = iter(test_loader)

    # Define start event, end event and number of repetitions
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings = np.zeros((repetitions, 1))

    # GPU warm-up
    for _ in range(20):
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

    # Calculate mean and std
    mean_time = np.sum(timings) / repetitions
    std_time = np.std(timings)

    return mean_time, std_time

    
if __name__ == "__main__":

    models = ["ResNet_split", "ResNet_MTAN", "SegNet_split", "SegNet_mtan", "EdgeSegNet", "GuidedDepth", "DDRNet"]

    nyuv2_timings = {}
    cityscapes_timings = {}
    
    # Create timings for NYUV2 dataset
    for model in models:
        model_mean, model_std = inference_test(model, "nyuv2")
        nyuv2_timings[model] = [model_mean, model_std]

    # Create timings for CityScapes dataset
    for model in models:
        model_mean, model_std = inference_test(model, "cityscapes")
        cityscapes_timings[model] = [model_mean, model_std]

    print("Timings for all models on NYUV2 given in milliseconds")
    print(nyuv2_timings)

    print("Timings for all models on CityScapes given in milliseconds")
    print(cityscapes_timings)

