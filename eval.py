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
from create_dataset import *
from utils import *


parser = argparse.ArgumentParser(description='Multi-task/Auxiliary Learning: Dense Prediction Tasks')
parser.add_argument('--network', default='SegNet_split', type=str, help='SegNet_split, SegNet_mtan, ResNet_split, Resnet_mtan')
parser.add_argument('--dataset', default='nyuv2', type=str, help='nyuv2, cityscapes')
opt = parser.parse_args()

def main():

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
    device = torch.device("cpu")

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
    model.eval()

    # Create iteratable object
    test_dataset = iter(test_loader)

    # Make prediction
    with torch.no_grad():
        test_data, test_target = test_dataset.next()
        test_data = test_data.to(device)
        test_target = {task_id: test_target[task_id].to(device) for task_id in train_tasks.keys()}
        test_pred = model(test_data)

    test_target = list(test_target.values())
    
    print(test_data.size())
    print(test_target[0].size())
    print(test_target[1].size())
    print(test_target[2].size())

    # Reshape input data
    test_data = torch.squeeze(test_data, 0)
    #test_target[0] = torch.squeeze(test_target[0], 0)
    test_target[1] = torch.squeeze(test_target[1], 0)
    test_target[2] = torch.squeeze(test_target[2], 0)
    test_data = test_data.permute(1, 2, 0)
    test_target[0] = test_target[0].permute(1, 2, 0)
    test_target[1] = test_target[1].permute(1, 2, 0)
    test_target[2] = test_target[2].permute(1, 2, 0)   

    # Reshape output data 
    test_pred[0] = torch.squeeze(test_pred[0], 0)
    test_pred[1] = torch.squeeze(test_pred[1], 0)
    test_pred[2] = torch.squeeze(test_pred[2], 0)
    test_pred[0] = test_pred[0].permute(1, 2, 0)
    test_pred[1] = test_pred[1].permute(1, 2, 0)
    test_pred[2] = test_pred[2].permute(1, 2, 0)   

    print(test_data.size())
    print(test_target[0].size())
    print(test_target[1].size())
    print(test_target[2].size())


    colours = iter([
        [0, 255, 0],
        [0, 0, 255],
        [255, 0, 0],
        [0, 255, 255],
        [255, 255, 0],
        [255, 0, 255],
        [80, 70, 180],
        [250, 80, 190],
        [245, 145, 50],
        [70, 150, 250],
        [50, 190, 190],
        [190, 50, 80],
        [170, 100, 40]
    ])

    # Initialize segmented image with original
    fused_img = test_data
    plt.imshow(fused_img)
    plt.show()

    # Run through the 13 categories and add masks to original 
    for img in test_pred[0]:
        color = next(colours)
        seg_img = get_colored_segmentation_image(img.astype(np.uint8), 13, color)
        fused_img = cv2.addWeighted(fused_img.astype(np.uint8), 1, seg_img.astype(np.uint8), 0.5, 0)


    # Create subplots
    fig, axarr = plt.subplots(2, 4)
    axarr[0, 0].imshow(test_data) # Original
    axarr[0, 1].imshow(test_target[0]) # Segmentation truth
    axarr[0, 2].imshow(test_target[1], cmap="gray") # Depth truth
    axarr[0, 3].imshow(test_target[2]) # Normals truth
    axarr[1, 0].imshow(test_data) # Original
    axarr[1, 1].imshow(fused_img) # Segmentation prediction
    axarr[1, 2].imshow(test_pred[1], cmap="gray") # Depth prediction
    axarr[1, 3].imshow(test_pred[2]) # Normals prediction
    
    plt.show()


def get_colored_segmentation_image(seg_arr, n_classes, color):

    output_height = seg_arr.shape[0]
    output_width = seg_arr.shape[1]

    seg_img = np.zeros((output_height, output_width, 3))

    for c in range(n_classes):
        seg_arr_c = seg_arr[:, :] == c
        seg_img[:, :, 0] += ((seg_arr_c)*(color[0])).astype('uint8')
        seg_img[:, :, 1] += ((seg_arr_c)*(color[1])).astype('uint8')
        seg_img[:, :, 2] += ((seg_arr_c)*(color[2])).astype('uint8')

    return seg_img


def overlay_seg_image(inp_img, seg_img):
    orininal_h = inp_img.shape[0]
    orininal_w = inp_img.shape[1]
    seg_img = cv2.resize(seg_img, (orininal_w, orininal_h), interpolation=cv2.INTER_NEAREST)

    fused_img = (inp_img/2 + seg_img/2).astype('uint8')
    return fused_img


if __name__ == "__main__":

    main()