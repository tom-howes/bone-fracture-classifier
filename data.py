import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class FractureDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = os.listdir(img_dir)
        self.num_classes = 7

        self.class_names = ['elbow positive', 'fingers positive', 'forearm fracture', 'humerus fracture', 'humerus', 'shoulder fracture', 'wrist positive']
    
    def __len__(self):
        return(len(self.images))
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")

        # Multi-hot encoded label [0, 0, 0...]
        label = torch.zeros(self.num_classes + 1) # 8th class is no fracture

        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_name)

        has_label = False
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        label[class_id] = 1
                        has_label = True
        
        if not has_label:
            label[7] = 1 # "no fracture"

        if self.transform:
            img = self.transform(img)

        return img, label     



