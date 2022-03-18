from os import access
import cv2
from glob import glob
import numpy as np

import torch
import torchvision
import Stain_Normalization.stain_utils as utils
import Stain_Normalization.stainNorm_Reinhard as reinhard 

def load_dataset(data_path):
    print(f"datest dir.: {data_path}")

    images = []
    masks = []

    imgs_train_path = glob(data_path + '/train/images/*.png')
    masks_train_path = glob(data_path + '/train/masks/*.png')
        
    # TODO: devide values by 255 to get values between 0 - 1
    for img_path in imgs_train_path:
        images.append(np.array(cv2.imread(img_path, cv2.IMREAD_COLOR)))
    for mask_path in masks_train_path:
        masks.append(np.array(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)))
#     print(imgs_train_path)

    return images, masks


def normalize_images(target_image, train_batch):
    
    normalized_images = []
    
    n = reinhard.Normalizer()
    n.fit(target_image)
    for index in range(len(train_batch)):
        normalized_images.append(n.transform(train_batch[index]))
        
    return normalized_images

def tiles_from_images(images_train, masks_train, tile_size, offset):

    cropped_images = []
    cropped_masks = []

    for index in range(len(images_train)):
        # Get image/mask parameters
        image_height, image_width, image_channels = images_train[index].shape
        mask_height, mask_width = masks_train[index].shape
            
        row_index = 0
        column_index = 0
            
        # Cycle by offset and cut images untill it is possible
        while row_index + tile_size[0] <= image_height:
            column_index = 0 
            while column_index + tile_size[1] <= image_width:
                cropped_images.append(images_train[index][row_index : row_index + tile_size[0], column_index : column_index + tile_size[1]])
                cropped_masks.append(masks_train[index][row_index : row_index + tile_size[0], column_index : column_index + tile_size[1]])
                    
                column_index += offset[1]   
            row_index += offset[0]  
            
        # Add tiles from last column (image width - tile size)
        row_index_tmp = 0 
        last_column = image_width - tile_size[1]
        while row_index_tmp + tile_size[0] <= image_height:
            cropped_images.append(images_train[index][row_index_tmp : row_index_tmp + tile_size[0], last_column : last_column + tile_size[1]])
            cropped_masks.append(masks_train[index][row_index_tmp : row_index_tmp + tile_size[0], last_column : last_column + tile_size[1]])
                
            row_index_tmp += offset[0]
            
        # Adding tiles from last row (image height - tile size )
        column_index_tmp = 0
        last_row = image_height - tile_size[0]
        while column_index_tmp + tile_size[1] <= image_width:
            cropped_images.append(images_train[index][last_row : last_row + tile_size[0], column_index_tmp : column_index_tmp + tile_size[1]])
            cropped_masks.append(masks_train[index][last_row : last_row + tile_size[0], column_index_tmp : column_index_tmp + tile_size[1]])
                
            column_index_tmp += offset[1]
                
        # Adding one last tile from last column and row
        cropped_images.append(images_train[index][last_row : last_row + tile_size[0], last_column : last_column + tile_size[1]])
        cropped_masks.append(masks_train[index][last_row : last_row + tile_size[0], last_column : last_column + tile_size[1]])

    return cropped_images, cropped_masks

def save_checkpoint(state, filename="outputs/model_checkpoint/checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def check_accuracy(loader, model, device="cuda"):
    correct = 0
    pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            correct += (preds == y).sum()
            pixels += torch.numel(preds)
            dice_score += (2*(preds * y).sum() / (preds + y).sum() + 1e-8)

    acc = correct/pixels*100
    dice = dice_score/len(loader) * 100 
    # print(f"Got {correct}/{pixels} with acc {acc:.2f}")
    # print(f"Dice score: {dice}")
    model.train()
    return acc, dice

def save_predictions_as_img(loader, model, folder, device="cuda"):
    model.eval()
    
    for index, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}/prediction_{index}.png")
        torchvision.utils.save_image(y.float().unsqueeze(1), f"{folder}/correct_{index}.png")

        if index > 10:
            break

    model.train()

def perform_validation(model, optimizer, validation_loader, DEVICE):
        # save model
        # checkpoint = {
            # "state_dict": model.state_dict(),
            # "optimizer": optimizer.state_dict()
        # }
        # save_checkpoint(checkpoint)

        # check accuracy
        acc, dice = check_accuracy(validation_loader, model, device=DEVICE)
        
        # print some examples to a folder
        # save_predictions_as_img(validation_loader, model, folder='outputs/images/prediction_images', device=DEVICE)
        return acc, dice