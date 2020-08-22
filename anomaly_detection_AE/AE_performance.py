# import for training
import sys
sys.path.append('/workspace')

import os
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from utils.tools import default_loader, is_image_file
from data.dataset import Dataset
from tensorboardX import SummaryWriter 
import matplotlib.pyplot as plt
import torchvision.utils as vutils

import sys
sys.path.append('../PerceptualSimilarity')

import models as PerceptualSimilarity
from sklearn.metrics import roc_auc_score, accuracy_score

from networks import autoencoder, simulator, discriminator

class MVTecDataset(Dataset):
    def __init__(self, TYPE='bottle', isTrain=True):
        self.gt_path = '../MVTec/'+TYPE+'/ground_truth_resize/all'
        self.train_path = '../MVTec/'+TYPE+'/train_resize/'
        self.test_path = '../MVTec/'+TYPE+'/test_resize/all'
        
        self.data_path = self.train_path if isTrain else self.test_path
        self.samples = [x for x in os.listdir(self.data_path) if is_image_file(x)]
        self.isTrain = isTrain
    
    def __getitem__(self, index):
        path = os.path.join(self.data_path, self.samples[index])
        
        img = default_loader(path)
        img = transforms.ToTensor()(img)  # turn the image to a tensor
        
        if not self.isTrain:
            gt_path = os.path.join(self.gt_path, self.samples[index])
        
            mask = default_loader(gt_path)
            mask = transforms.ToTensor()(mask)  # turn the image to a tensor
            return img, mask

        return img
    
    
    def __len__(self):
        return len(self.samples)

def difNormalize(input_matrix, threshold=None):
    _min = input_matrix.min()
    _max = input_matrix.max()
    
    input_matrix = (input_matrix - _min) / (_max - _min)
    
    if threshold != None:
        input_matrix[input_matrix < threshold] = 0
        input_matrix[input_matrix >= threshold] = 1
        
    return input_matrix

def denormalize(input_matrix):
    return np.around(input_matrix * 127.5 + 127.5).astype(int)

ALL_TYPES = [
    "bottle", "cable" ,
    # "capsule", 
    "carpet",
    "grid" , "hazelnut" ,"leather", "metal_nut",
    "pill" ,"screw" ,"tile", "toothbrush", 
    "transistor" ,"wood", "zipper"
]

for index, TYPE in enumerate(ALL_TYPES):
    # HYPYER PARAMETERS
    num_epochs = 150
    batch_size = 8
    lr = 1e-3
    weight_decay = 1e-5
    writer = SummaryWriter('checkpoint_performance/ALLexp-'+str(index)+'')

    # Data Loader
    trainDatset = MVTecDataset(TYPE=TYPE, isTrain=True)
    testDatset = MVTecDataset(TYPE=TYPE, isTrain=False)
    train_loader = DataLoader(
        dataset=trainDatset,
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4
    )
    test_loader = DataLoader(
        dataset=testDatset,
        batch_size=1, 
        shuffle=False,
        num_workers=4
    )

    model = autoencoder.Autoencoder().cuda()
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=lr,
        weight_decay=weight_decay
    )

    # try:
    print('start training')
    for epoch in range(num_epochs):
        for img in train_loader:
            img = Variable(img).cuda()
            
            # ===================forward=====================
            output = model(img)
            loss = criterion(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        writer.add_scalar('loss', loss, epoch)
        
        if epoch % 5 == 0:
            writer.add_images('reconstruct', output, epoch)

    perceptual_loss = PerceptualSimilarity.PerceptualLoss(model='net-lin', net='alex', use_gpu=True, gpu_ids=[0])
    L2_loss = nn.MSELoss(reduction='none')

    total_auc = 0
    total_acc = 0
    total_image = 0
    for index, (img, mask) in enumerate(test_loader):
        img = Variable(img).cuda()
        mask = Variable(mask).cuda()

        # 取得 model 輸出
        output = model(img)

        # 計算 dif (相似度以及 L2)
        dif = perceptual_loss.forward(output, img)
        l1Dif = L2_loss(output, img)
        l1Dif = torch.mean(l1Dif, 1, True)
        
        maskEdge = difNormalize(dif[0].cpu().detach().numpy() * l1Dif[0].cpu().detach().numpy())
        maskEdge2 = difNormalize(dif[0].cpu().detach().numpy() * l1Dif[0].cpu().detach().numpy(), threshold=0.5)
        
        mask = torch.mean(mask, 1, True)
        true_mask = mask[0].cpu().detach().numpy().flatten()

        AUC = roc_auc_score(true_mask, maskEdge.flatten())
        ACC = accuracy_score(true_mask, maskEdge2.flatten())
        
        total_auc += AUC
        total_acc += ACC
        total_image += 1
        
        writer.add_images('Test reconstruct', output, epoch)
        writer.add_images('Test origin', img, epoch)


    print("===={}====".format(TYPE))
    print("Acerage AUC = {}".format(total_auc / total_image))
    print("Acerage ACC = {}".format(total_acc / total_image))
    # except Exception as e:
    #     print("{} get some error".format(TYPE))
    #     print(e)