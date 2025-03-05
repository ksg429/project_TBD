import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from skimage import io


class labeled_dataset_from_path(Dataset):
    def __init__(self, data_frame, data_indexes, transforms = None):
        self.img_path = data_frame['image_path']
        self.labels = data_frame['labels']
        self.data_indexes = data_indexes
        self.transforms = transforms

    def __getitem__(self, index):
        path = self.img_path[self.data_indexes[index]]
        img = Image.fromarray(io.imread(path)).convert("RGB")
        
        if self.transforms:
            img = self.transforms(img)
        

        label = self.labels[self.data_indexes[index]]
        label = torch.FloatTensor(label)

        return  img, label, self.data_indexes[index]

    def __len__(self):
        return len(self.data_indexes)
    
class unlabeled_dataset_from_path(Dataset):
    def __init__(self, data_frame, data_indexes, transforms = None):
        self.img_path = data_frame['image_path']
        self.data_indexes = data_indexes
        self.transforms = transforms


    def __getitem__(self, index):
        path = self.img_path[self.data_indexes[index]]
        img = Image.fromarray(io.imread(path)).convert("RGB")
        if self.transforms:
            img = self.transforms(img)

        return  img, self.data_indexes[index]

    def __len__(self):
        return len(self.data_indexes)
    