import cv2
from glob import glob
import numpy as np
import os

import torch
import torchvision
import Stain_Normalization.stain_utils as utils
import Stain_Normalization.stainNorm_Reinhard as reinhard 
import torch.nn as nn

def load_dataset(data_path):
    print(f"datest dir.: {data_path}")

    images = []
    masks = []

    imgs_train_path = glob(data_path + '/train/images/*.png')
    masks_train_path = glob(data_path + '/train/masks/*.png')
    
    for index in range(0, len(imgs_train_path[0:3])):
        img_path = imgs_train_path[index]
        mask_path = masks_train_path[index]
        
        if os.path.basename(img_path) == os.path.basename(mask_path):
            mask = np.array(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE))
            image = np.array(cv2.imread(img_path, cv2.IMREAD_COLOR))
            masks.append(mask)
            images.append(image)
        else:
            print("Image and Mask are not corresponding")

    return images, masks

def normalize_images(target_image, train_batch):
    
    normalized_images = []
    
    n = reinhard.Normalizer()
    n.fit(target_image)

    for image in train_batch:
        normalized_images.append(n.transform(image))
        
    return normalized_images

def tiles_from_images(images_train, masks_train, tile_size, offset):

    cropped_images = []
    cropped_masks = []

    outside_roi_threshold = (tile_size[0] * tile_size[1]) * 0.9
    number_of_deleted = 0

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
                cropped_image = images_train[index][row_index : row_index + tile_size[0], column_index : column_index + tile_size[1]]
                cropped_mask = masks_train[index][row_index : row_index + tile_size[0], column_index : column_index + tile_size[1]]
                
                # Check if outside roi <= 90% of tile
                if np.sum(cropped_mask == 0) <= outside_roi_threshold:
                    cropped_images.append(cropped_image)
                    cropped_masks.append(cropped_mask)
                else:
                    number_of_deleted += 1
                column_index += offset[1]   
            row_index += offset[0]  
            
        # Add tiles from last column (image width - tile size)
        row_index_tmp = 0 
        last_column = image_width - tile_size[1]
        while row_index_tmp + tile_size[0] <= image_height:
            cropped_image = images_train[index][row_index_tmp : row_index_tmp + tile_size[0], last_column : last_column + tile_size[1]]
            cropped_mask = masks_train[index][row_index_tmp : row_index_tmp + tile_size[0], last_column : last_column + tile_size[1]]

            # Check if outside roi <= 90% of tile
            if np.sum(cropped_mask == 0) <= outside_roi_threshold:
                cropped_images.append(cropped_image)
                cropped_masks.append(cropped_mask)
            else:
                number_of_deleted += 1
            row_index_tmp += offset[0]
            
        # Adding tiles from last row (image height - tile size )
        column_index_tmp = 0
        last_row = image_height - tile_size[0]
        while column_index_tmp + tile_size[1] <= image_width:
            cropped_image = images_train[index][last_row : last_row + tile_size[0], column_index_tmp : column_index_tmp + tile_size[1]]
            cropped_mask = masks_train[index][last_row : last_row + tile_size[0], column_index_tmp : column_index_tmp + tile_size[1]]

            # Check if outside roi <= 90% of tile
            if np.sum(cropped_mask == 0) <= outside_roi_threshold:
                cropped_images.append(cropped_image)
                cropped_masks.append(cropped_mask)
            else:
                number_of_deleted += 1
            column_index_tmp += offset[1]
                
        # Adding one last tile from last column and row
        cropped_image = images_train[index][last_row : last_row + tile_size[0], last_column : last_column + tile_size[1]]
        cropped_mask = masks_train[index][last_row : last_row + tile_size[0], last_column : last_column + tile_size[1]]

        # Check if outside roi <= 90% of tile
        if np.sum(cropped_mask == 0) <= outside_roi_threshold:
            cropped_images.append(cropped_image)
            cropped_masks.append(cropped_mask)
        else:
            number_of_deleted += 1
    
    print(f"NUMBER OF DELETED TILES IS: {number_of_deleted}")
    return cropped_images, cropped_masks

def save_checkpoint(state, filename="outputs/model_checkpoint/checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

# TODO ACC/DICE description
def calculate_metrics(loader, model, device="cuda"):
    correct = 0
    pixels = 0
    dice_score = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            preds = model(x).to(device=device)
            sm = nn.Softmax(dim=1)
            preds = sm(preds)

            # DICE
            number_of_classes = preds.size(1)
            dice_class = 0
            for index in range(number_of_classes):
                preds_class = preds[:, index, :, :] == torch.max(preds, dim=1)[0]
                preds_class = preds_class.unsqueeze(1).to(device=device)

                targets_class = torch.eq(y[:, 0, :, :], index)
                targets_class = targets_class.unsqueeze(1).to(device=device)

                intersection = (preds_class * targets_class).sum()
                smooth = 0.0001
                dice_class += (2. * intersection + smooth) / ((preds_class + targets_class).sum() + smooth)
            
            dice_score += dice_class/number_of_classes

            # ACC
            preds = torch.argmax(preds, dim=1).unsqueeze(1)
            correct += (preds == y).sum()
            pixels += torch.numel(preds)

    dice_score = dice_score/len(loader) * 100 
    acc = correct/pixels*100
    print(f"\nGot {correct}/{pixels} with acc {acc:.2f}")
    print(f"Multi-class dice score: {dice_score}")
    model.train()
    return acc, dice_score

# Save predictions and targets as images 
# Description:
#   To visualize the prediction, RGB values ​​are first mapped to the individual classes in the mask.
#   From the prediction, the strongest value is selected with the limit of all five and than the color is applyed
#       GREEN - outside roi (+ others)
#       RED - tumor
#       ORANGE - stroma
#       BLUE - lymphocitic_infiltrate
#       YELLOS - necrosis
#   The same type of mapping is performed for the target masks.
#   Images are than saved under names: prediction_{index} for prediction and correct_{index} for corresponding target mask.
def save_predictions_as_img(loader, model, folder, device="cuda"):
    model.eval()
    class_to_color = [
        torch.tensor([0.0, 1.0, 0.0]).to(device=device), # GREEN - outside roi (+ others)
        torch.tensor([1.0, 0.0, 0.0]).to(device=device), # RED - tumor
        torch.tensor([1.0, 0.5, 0.0]).to(device=device), # ORANGE - stroma
        torch.tensor([0.0, 0.0, 1.0]).to(device=device), # BLUE - lymphocitic_infiltrate
        torch.tensor([1.0, 0.9, 0.0]).to(device=device)  # YELLOW - necrosis
    ]
    for index, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = model(x).to(device=device)
            sm = nn.Softmax(dim=1)
            preds = sm(preds)

            preds_output = torch.zeros(preds.shape[0], 3, preds.size(-2), preds.size(-1), dtype=torch.float).to(device=device)
            target_output = torch.zeros(y.shape[0], 3, y.size(-2), y.size(-1), dtype=torch.float).to(device=device)

            for class_idx, color in enumerate(class_to_color):
                curr_color = color.reshape(1, 3, 1, 1).to(device=device)

                preds_mask = preds[:,class_idx,:,:] == torch.max(preds, dim=1)[0]
                preds_mask = preds_mask.unsqueeze(1).to(device=device)
                preds_segment = preds_mask*curr_color
                preds_output += preds_segment

                target_mask = torch.eq(y[:, 0, :, :], class_idx)
                target_mask = target_mask.unsqueeze(1).to(device=device)
                target_segment = target_mask*curr_color
                target_output += target_segment
            
            print(x.shape)
            prediction_dir = folder["prediction_masks"]
            target_dir = folder["target_masks"]
            target_img_dir = folder["target_images"]
            torchvision.utils.save_image(preds_output, f"{prediction_dir}/prediction_{index}.png")
            torchvision.utils.save_image(target_output, f"{target_dir}/target_{index}.png")
            torchvision.utils.save_image(x, f"{target_img_dir}/image_{index}.png")
        if index > 10:
            break

    model.train()

def edit_masks(masks):
    
    mapping = {
            # 1: 1,
            # 2: 2,
            # 3: 3,
            # 4: 4,
            
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            
            9: 1,

            10: 3,
            11: 3,
            
            12: 0,
            13: 0,
            14: 0,
            15: 0,
            16: 0,
            17: 0,
            18: 0,
            19: 0,
           
            20: 1,

            21: 0
        }
    for index in range(len(masks)):
        for k in mapping:
            masks[index][masks[index] == k] = mapping[k]
    return masks