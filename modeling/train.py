import os
import numpy as np
import torch
torch.cuda.empty_cache()
from torch.autograd import Variable
from torchvision.models.vgg import model_urls
import torchvision.models as models
from attention_model import Attn_Net
from dataloader import MyJP2Dataset, NFDataset, FLDataset, Balancer
from evaluation import sklearn_Compatible_preds_and_targets, accuracy_score

# All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn as nn 
import torch.nn.functional as F

# For all Optimization algorithms, SGD, Adam, etc.
import torch.optim as optim

# Loading and Performing transformations on dataset
import torchvision
import torchvision.transforms as transforms 
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.sampler import Sampler, WeightedRandomSampler

#Labels in CSV and Inputs in Fits in a folder
import pandas as pd
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt

#For Confusion Matrix
from sklearn.metrics import confusion_matrix

#Warnings
import warnings
warnings.simplefilter("ignore", Warning)

#Time Computation
import timeit
import argparse

parser = argparse.ArgumentParser(description="fullDiskModeltrainer")
parser.add_argument("--fold", type=int, default=1, help="Fold Selection")
parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--im_size", type=int, default=256, help="image_size")
parser.add_argument("--attention", type=int, default=1, help="Turn on or off attention")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--weight_decay", type=float, default=0.5, help="regularization parameter")
opt = parser.parse_args()


# Random Seeds for Reproducibility
def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(4)


# CUDA for PyTorch -- GPU SETUP
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
# device= torch.device('cpu')
torch.backends.cudnn.benchmark = True
# print(device)

def dataloading():
    #Define Transformations
    #General transformation which resizes the input to specified size. Default=256
    #This transformation is for majority class NF and also for entire validation set.
    transformations = transforms.Compose([
        transforms.Resize(opt.im_size),
        transforms.ToTensor()
    ])

    #We augment only the flaring instances
    #Augmentation1: Rotating full-disk magnetogram images within +/- 5degrees.
    rotation = transforms.Compose([
        transforms.Resize(opt.im_size),
        transforms.RandomRotation(degrees=(-5,5)),
        transforms.ToTensor()
    ])

    #Augmentation2: Flipping full-disk magnetogram images horizontally.
    hr_flip = transforms.Compose([
        transforms.Resize(opt.im_size),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor()
    ])

    #Augmentation3: Flipping full-disk magnetogram images vertically.
    vr_flip = transforms.Compose([
        transforms.Resize(opt.im_size),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor()
    ])

    #Specifying image and labels location
    image_dir = '/data/hmi_jpgs_512/'
    if opt.fold==1:
        csv_file_train = '../data_labeling/data_labels/simplified_data_labels/Fold1_train.csv'
        csv_file_val = '../data_labeling/data_labels/simplified_data_labels/Fold1_val.csv'

    elif opt.fold==2:
        csv_file_train = '../data_labeling/data_labels/simplified_data_labels/Fold2_train.csv'
        csv_file_val = '../data_labeling/data_labels/simplified_data_labels/Fold2_val.csv'

    elif opt.fold==3:
        csv_file_train = '../data_labeling/data_labels/simplified_data_labels/Fold3_train.csv'
        csv_file_val = '../data_labeling/data_labels/simplified_data_labels/Fold3_val.csv'

    else:
        csv_file_train = '../data_labeling/data_labels/simplified_data_labels/Fold4_train.csv'
        csv_file_val = '../data_labeling/data_labels/simplified_data_labels/Fold4_val.csv'

    #Loading Dataset -- Trimonthly partitioned with defined augmentation
    ori_nf = NFDataset(csv_file = csv_file_train, 
                                root_dir = image_dir,
                                transform = transformations)

    ori_fl = FLDataset(csv_file = csv_file_train, 
                                root_dir = image_dir,
                                transform = transformations)
    hr_flip_fl = FLDataset(csv_file = csv_file_train, 
                                root_dir = image_dir,
                                transform = hr_flip)
    vr_flip_fl = FLDataset(csv_file = csv_file_train, 
                                root_dir = image_dir,
                                transform = vr_flip)
    rotation_fl = FLDataset(csv_file = csv_file_train, 
                                root_dir = image_dir,
                                transform = rotation)

    #Balancer is to load the remaining instances after augmentation. Suppose there are 100 NF instances, and 21 FL instances.
    #Then 4*21=84 instances comes from original and 3 augmentations. remaining 16 required to create a balance dataset will be loaded from balancer without any
    # augmentation, which essentially means partial oversampling.
     
    if opt.fold==2:
        len = len(ori_nf) - 5*len(ori_fl)
    
    else: 
        len = len(ori_nf) - 6*len(ori_fl)

    #Loading FL class to balance the dataset
    bal_set = Balancer(csv_file = csv_file_train, 
                                root_dir = image_dir,
                                len = len,
                                transform = transformations)

    
    if opt.fold==2:
        train_set = ConcatDataset([ori_fl, hr_flip_fl, vr_flip_fl, rotation_fl, hr_flip_fl, ori_nf, bal_set])
        # print(len(ori_nf), len(ori_fl), 5*len(ori_fl)+len(bal_set))
    else:
        train_set = ConcatDataset([ori_fl, hr_flip_fl, vr_flip_fl, rotation_fl, hr_flip_fl, ori_fl, ori_nf, bal_set])
        # print(len(ori_nf), len(ori_fl), 6*len(ori_fl)+len(bal_set))
    

    val_set = MyJP2Dataset(csv_file = csv_file_val, 
                                root_dir = image_dir,
                                transform = transformations)

    train_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, num_workers=4, pin_memory = True, shuffle = True)

    val_loader = DataLoader(dataset=val_set, batch_size=opt.batch_size, num_workers=4, pin_memory = True, shuffle = False)

    return train_loader, val_loader


def visualize_batch_sample(data_loader):
    cmap = plt.get_cmap('Greys_r')
    dataiter = iter(data_loader)
    #dataiter.next()
    images, labels = dataiter.next()
    flare_types = {0: 'Non_flare', 1: 'Flare'}
    fig, axis = plt.subplots(4, 4, figsize=(20, 20))
    for i, ax in enumerate(axis.flat):
        with torch.no_grad():
            image, label = images[i], labels[i]
            ax.imshow(image.permute(1,2,0), cmap=cmap, vmin=0, vmax=1) # add image
            ax.set(title = f"{flare_types[label.item()]}")


def train():
    train_loader, val_loader = dataloading()
    # visualize_batch_sample(train_loader)

    #Model Configuration for two GPUs
    learning_rate = opt.lr
    num_epochs = opt.epochs

    device_ids = [0, 1]

    if opt.attention==1:
        savemodel = f'trained_models/attention/attn_fold{opt.fold}.pth'
        bestmodel = f'trained_models/attention/best_attn_fold{opt.fold}.pth'
        attn = True
    else:
        savemodel = f'trained_models/no_attention/fold_{opt.fold}_no_attn.pth'
        bestmodel = f'trained_models/no_attention/best_fold_{opt.fold}_no_attn.pth'
        attn = False

    net = Attn_Net(im_size=opt.im_size, num_classes=2, attention=attn, init='kaimingUniform').to(device)
    model = nn.DataParallel(net, device_ids=device_ids).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=opt.weight_decay)
    

    # Training Network
    print("Training in Progress..")
    train_loss_values = []
    val_loss_values = []
    train_tss_values = []
    val_tss_values = []
    train_hss_values = []
    val_hss_values = []
    train_time = []
    val_time = []
    learning_rate_values = []
    best_tss = 0
    best_hss = 0
    for epoch in range(1,num_epochs):

        #Dynamically Reducing Learning Rate
        if epoch%3==0:
            for g in optimizer.param_groups:
                g['lr'] = g['lr']/2
        
        #Timer for Training one epoch
        start_train = timeit.default_timer() 
        
        # setting the model to train mode
        model.train()
        train_loss = 0
        train_tss = 0.
        train_hss = 0.
        train_prediction_list = []
        train_target_list = []
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)
            train_target_list.append(targets)
            
            # forward prop
            scores,_,_,_ = model(data)
            loss = criterion(scores, targets)
            _, predictions = torch.max(scores,1)
            train_prediction_list.append(predictions)
            
            # backward prop
            optimizer.zero_grad()
            loss.backward()
            
            # SGD step
            optimizer.step()
            
            # accumulate the training loss
            #print(loss.item())
            train_loss += loss.item()
            #train_acc+= acc.item()
            
        stop_train = timeit.default_timer()
        print(stop_train-start_train)
        train_time.append(stop_train-start_train)

        # Validation: setting the model to eval mode
        model.eval()
        start_val = timeit.default_timer()
        val_loss = 0.
        val_tss = 0.
        val_hss = 0.
        val_prediction_list = []
        val_target_list = []

        # Turning off gradients for validation
        with torch.no_grad():
            for d, t in val_loader:
                # Get data to cuda if possible
                d = d.to(device=device)
                t = t.to(device=device)
                val_target_list.append(t)
                
                # forward pass
                s,_,_,_ = model(d)
                #print("scores", s)
                                    
                # validation batch loss and accuracy
                l = criterion(s, t)
                _, p = torch.max(s,1)
                #print("------------------------------------------------")
                #print(torch.max(s,1))
                #print('final', p)
                val_prediction_list.append(p)
                
                # accumulating the val_loss and accuracy
                val_loss += l.item()
                #val_acc += acc.item()
                del d,t,s,l,p
                torch.cuda.empty_cache()
        stop_val = timeit.default_timer()
        val_time.append(stop_val-start_val)
        learning_rate_values.append(optimizer.param_groups[0]['lr'])

        #Epoch Results
        train_loss /= len(train_loader)
        train_loss_values.append(train_loss)
        val_loss /= len(val_loader)
        val_loss_values.append(val_loss)
        train_tss, train_hss = sklearn_Compatible_preds_and_targets(train_prediction_list, train_target_list)
        train_tss_values.append(train_tss)
        train_hss_values.append(train_hss)
        val_tss, val_hss = sklearn_Compatible_preds_and_targets(val_prediction_list, val_target_list)
        val_tss_values.append(val_tss)
        val_hss_values.append(val_hss)
        if val_tss>best_tss and val_hss>best_hss:
            best_tss = val_tss
            best_hss = val_hss
            best_model = model.module.state_dict()
            torch.save(model.module.state_dict(), bestmodel)
        print(f'Epoch: {epoch}/{num_epochs-1}')
        print(f'Training--> loss: {train_loss:.4f}, TSS: {train_tss:.4f}, HSS2: {train_hss:.4f} | Val--> loss: {val_loss:.4f}, TSS: {val_tss:.4f} | HSS2: {val_hss:.4f} ')
    print(best_tss, best_hss)
    torch.save(model.module.state_dict(), savemodel)


if __name__ == "__main__":
    train()