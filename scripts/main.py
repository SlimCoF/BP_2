from __future__ import division

import os
from azureml.core import Dataset, Run
import argparse
import numpy as np
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from auxiliary_functions import *
from dataset import CustomDataset
from model import U_net
from train import *


# (HEIGHT, WIDTH)
TILE_SIZE = (256, 256)
OFFSET = (128, 128)

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 4
NUM_WORKERS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = ''
TRAIN_MAK_DIR = ''

if __name__ == '__main__':

    print(f"Device: {DEVICE}")

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='Path to dataset')
    parser.add_argument('--wandb', help='Path to dataset')
    args = parser.parse_args()
    data_path = args.data_path

    data_path = args.data_path
    
    TRAIN_IMG_DIR = data_path + '/train/images'
    TRAIN_MAK_DIR = data_path + '/train/masks'

    # If output path for images does not exist create it
    if not os.path.isdir('outputs/images'):
        os.mkdir('outputs/images')
    if not os.path.isdir('outputs/images/normalized'):
        os.mkdir('outputs/images/normalized')
    if not os.path.isdir('outputs/images/tiles'):
        os.mkdir('outputs/images/tiles')
    if not os.path.isdir('outputs/images/prediction_images'):
        os.mkdir('outputs/images/prediction_images')
    if not os.path.isdir('outputs/model_checkpoint'):
        os.mkdir('outputs/model_checkpoint')
        f = open('outputs/model_checkpoint/checkpoint.pth.tar', 'w')
    # Load raw images
    images_train, masks_train = load_dataset(data_path)
    
    # Edit mask to binary values 0 - outside roi, 1 - tumor
        # TODO: edit masks for predominant classes
    for index in range(len(masks_train)):
        masks_train[index][masks_train[index] != 1 ] = 0
    
    # Normalize images
    images_train = normalize_images(images_train[0], images_train)

    # Output some normalized images
    for index in range(len(images_train)):
        if (index >= 10 and index%10 == 0):
            cv2.imwrite(f'outputs/images/normalized/normalized_image{index}.png', images_train[index])

    # Create tiles from images
    images_train, masks_train = tiles_from_images(images_train[0:3], masks_train[0:3], TILE_SIZE, OFFSET)
    
    # Output some tiles
    for index in range(len(images_train)):
        if (index >= 10000 and index%10000 == 0):
            cv2.imwrite(f'outputs/images/tiles/tile{index}.png', images_train[index])

    print(f"Full length: {len(images_train)}")
    # Create validation dataset
    images_validation = []
    masks_validation = []
    images_train_length = len(images_train)
    images_validation_length = images_train_length * 0.2
    for number in range(0, int(images_validation_length)):
        rand_index = random.randint(0, images_train_length - number - 1)
        images_validation.append(images_train[rand_index])
        masks_validation.append(masks_train[rand_index])

        images_train.pop(rand_index)
        masks_train.pop(rand_index)
    
    print(f"Images train length: {len(images_train)}")
    print(f"Masks train length: {len(masks_train)}")
    print(f"Images validation length: {len(images_validation)}")
    print(f"Masks validation length: {len(masks_validation)}")
    
    # Create custom image dataset from images_train, masks_train, image_validation and masks_validation
    train_dataset = CustomDataset(images_train, masks_train, train=True)
    train_loader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            shuffle=True
        )

    validation_dataset = CustomDataset(images_validation, masks_validation, train=True)
    validation_loader = DataLoader(
            	validation_dataset,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                shuffle=False
            )

    print(train_loader)
    print(validation_loader)

    # Train the model
    model = U_net(input_channels=3, output_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    if LOAD_MODEL:
        load_checkpoint(torch.load('outputs/model_checkpoint/checkpoint.pth.tar'), model)
    
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train(train_loader, validation_loader, model, optimizer, loss_fn, scaler, DEVICE)


        perform_validation(model, optimizer, validation_loader, DEVICE)

        # # save model
        # checkpoint = {
        #     "state_dict": model.state_dict(),
        #     "optimizer": optimizer.state_dict()
        # }
        # save_checkpoint(checkpoint)

        # # check accuracy
        # check_accuracy(validation_loader, model, device=DEVICE)

        # # print some examples to a folder
        # save_predictions_as_img(validation_loader, model, folder='outputs/images/prediction_images', device=DEVICE)


