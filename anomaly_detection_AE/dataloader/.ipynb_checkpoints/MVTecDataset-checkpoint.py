import os

import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from utils.tools import get_config, default_loader, is_image_file, normalize

class MVTecDataset(Dataset):
    def __init__(self, TYPE='bottle', isTrain='train'):
        self.gt_path = '/root/AFS/Corn/AEGAN/MVTec/'+TYPE+'/ground_truth_resize/all'
        self.train_path = '/root/AFS/Corn/AEGAN/MVTec/'+TYPE+'/train_resize/train'
        self.val_path = '/root/AFS/Corn/AEGAN/MVTec/'+TYPE+'/train_resize/validation'
        self.test_path = '/root/AFS/Corn/AEGAN/MVTec/'+TYPE+'/test_resize/all'
        
        self.data_path = self.train_path if isTrain=='train' else self.val_path if isTrain=='val' else self.test_path
        self.samples = [x for x in os.listdir(self.data_path) if is_image_file(x)]
        self.isTrain = isTrain
    
    def __getitem__(self, index):
        path = os.path.join(self.data_path, self.samples[index])
        
        img = default_loader(path)
        img = transforms.ToTensor()(img)  # turn the image to a tensor
        
        if self.isTrain == 'test':
            gt_path = os.path.join(self.gt_path, self.samples[index])
        
            mask = default_loader(gt_path)
            mask = transforms.ToTensor()(mask)  # turn the image to a tensor
            return img, mask
        return img
    
    
    def __len__(self):
        return len(self.samples)