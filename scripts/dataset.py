import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision

class CustomDataset(Dataset):
    def __init__(self, images, masks, train=True):

        self.images = images
        self.masks = masks
        self.transforms = transforms.ToTensor()

    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]
        t_image = self.transforms(image)
        return t_image, mask

    def __len__(self):
        return len(self.images)