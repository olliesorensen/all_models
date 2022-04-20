import argparse

import pandas as pd 
import numpy as np

from utils import create_task_flags
from matplotlib import pyplot as plt


""" Script for visualizing training and validation error for various models. """


parser = argparse.ArgumentParser(description='Multi-task/Auxiliary Learning: Dense Prediction Tasks')
parser.add_argument('--network', default='SegNet_split', type=str, help='SegNet_split, SegNet_mtan, ResNet_split, Resnet_mtan')
parser.add_argument('--dataset', default='nyuv2', type=str, help='nyuv2, cityscapes')
opt = parser.parse_args()


if __name__ == "__main__":
    model_name = opt.network
    data_set = opt.dataset

    tasks = create_task_flags("all", opt.dataset)
    tasks = list(tasks.keys())

    # Read txt file with results 
    file_name = f"/Users/olemartinsorensen/Desktop/Thesis/Results/Equal/Train_and_Eval/Equal_{model_name}_{data_set}_Results.rtf"

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
    fig.suptitle(f"Training and Validation Error\n {model_name}, {data_set}")
    axarr[0].plot(train_task1, label="Train")
    axarr[0].plot(test_task1, label="Val")
    axarr[0].set_title("Semantic Segmentation")
    axarr[1].plot(train_task2, label="Train")
    axarr[1].plot(test_task2, label="Val")
    axarr[1].set_title("Depth" if opt.dataset == "nyuv2" else "Part Segmentation")
    axarr[2].plot(train_task3, label="Train")
    axarr[2].plot(test_task3, label="Val")
    axarr[2].set_title("Surface Normals" if opt.dataset == "nyuv2" else "Disparity")
    axarr[0].legend()
    axarr[1].legend()
    axarr[2].legend()
    axarr[0].set_xlabel("Epoch")
    axarr[0].set_ylabel("mIoU")
    axarr[1].set_xlabel("Epoch")
    axarr[1].set_ylabel("aErr." if opt.dataset == "nyuv2" else "mIoU")
    axarr[2].set_xlabel("Epoch")
    axarr[2].set_ylabel("mDist" if opt.dataset == "nyuv2" else "aErr.")
    plt.show()




