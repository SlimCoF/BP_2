import torch
from torch.utils.data import Dataset
from torch.nn.functional import normalize
import torchvision.transforms as transforms
import torchvision
import numpy as np
import cv2

class CustomDataset(Dataset):
    def __init__(self, images, masks, train=True):

        self.images = images
        self.masks = masks
        self.transforms = transforms.ToTensor()
    #     self.mapping = {
    #         # 1: 1,
    #         # 2: 2,
    #         # 3: 3,
    #         # 4: 4,
            
    #         5: 0,
    #         6: 0,
    #         7: 0,
    #         8: 0,
            
    #         9: 1,

    #         10: 3,
    #         11: 3,
            
    #         12: 0,
    #         13: 0,
    #         14: 0,
    #         15: 0,
    #         16: 0,
    #         17: 0,
    #         18: 0,
    #         19: 0,
           
    #         20: 1,

    #         21: 0
    #     }
    # def mask_to_class(self, mask):
    #     for k in self.mapping:
    #         mask[mask == k] = self.mapping[k]
    #     return mask

    def __getitem__(self, index):
        image = self.images[index]
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = image.astype('uint8')
        image = self.transforms(image)
        # image = torch.from_numpy(image)
        # image = image.float()/255

        mask = self.masks[index]
        # mask = self.mask_to_class(mask)
        mask = torch.from_numpy(mask)
        mask = mask.long().unsqueeze(0)

        # mask = mask.type(torch.LongTensor)
        return image, mask

    def __len__(self):
        return len(self.images)