from cgi import test
from configparser import Interpolation

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from model_segnet_mtan import *
from utils import *


def eval():

    batch_size = 2

    # Load data set
    dataset_path = opt.dataroot
    nyuv2_test_set = NYUv2(root=dataset_path, train=False)

    nyuv2_test_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_test_set,
        batch_size=batch_size,
        shuffle=False)

    # Load model (if model is saved on GPU but loaded on CPU, use map_location=torch.device())
    model = SegNet()
    model.load_state_dict(torch.load("model_mtan_nuyv2.pth", map_location=torch.device('cpu')))
    model.eval()

    # Create iteratable object
    test_dataset = iter(nyuv2_test_loader)

    # Define device 
    device = torch.device("cpu")

    for i in range(51):

        if i == 50:

            test_data, test_label, test_depth, test_normal = test_dataset.next()
            test_data, test_label = test_data.to(device), test_label.long().to(device)
            test_depth, test_normal = test_depth.to(device), test_normal.to(device)
        else:
            test_dataset.next()

    # Make prediction
    test_pred, _ = model(test_data)

    # Reshape test data from [2, 3, 288, 384] to [288, 384, 3]
    test_data = test_data.permute(0, 2, 3, 1)
    test_data = test_data.cpu().detach().numpy()
    test_data = test_data[0, :, :, :]

    # Reshape pred[0] data 
    test_pred[0] = test_pred[0].cpu().detach().numpy()
    test_pred[0] = test_pred[0][0, :, :, :]

    # Reshape pred[1] data [2, 3, 288, 384] to [288, 384, 1]
    test_pred[1] = test_pred[1].permute(0, 2, 3, 1)
    test_pred[1] = test_pred[1].cpu().detach().numpy()
    test_pred[1] = test_pred[1][0, :, :, :]

    # Reshape pred[2] data [2, 3, 288, 384] to [288, 384, 3]
    test_pred[2] = test_pred[2].permute(0, 2, 3, 1)
    test_pred[2] = test_pred[2].cpu().detach().numpy()
    test_pred[2] = test_pred[2][0, :, :, :]

    # Reshape test_label data [2, 288, 384] to [288, 384]
    test_label = test_label.cpu().detach().numpy()
    test_label = test_label[0, :, :]

    # Reshape test_depth data [2, 1, 288, 384] to [288, 384, 1]
    test_depth = test_depth.permute(0, 2, 3, 1)
    test_depth = test_depth.cpu().detach().numpy()
    test_depth = test_depth[0, :, :, :]

    # Reshape test_normal data [2, 3, 288, 384] to [288, 384, 3]
    test_normal = test_normal.permute(0, 2, 3, 1)
    test_normal = test_normal.cpu().detach().numpy()
    test_normal = test_normal[0, :, :, :]


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
        #plt.imshow(fused_img)
        #plt.show()


    # Create subplots
    fig, axarr = plt.subplots(2, 4)
    axarr[0, 0].imshow(test_data) # Original
    axarr[0, 1].imshow(test_label) # Segmentation truth
    axarr[0, 2].imshow(test_depth, cmap="gray") # Depth truth
    axarr[0, 3].imshow(test_normal) # Normals truth
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

    eval()