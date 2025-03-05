import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

import random
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from skimage import io
from PIL import Image
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score
#%%

from easydict import EasyDict
from datetime import datetime
from pytz import timezone
from torch.utils.tensorboard import SummaryWriter  # TensorBoard writer
import yaml


#%%


from load_ISIC2018 import load_ISIC2018_GT
from dataset_from_df import labeled_dataset_from_path, unlabeled_dataset_from_path
import opts
from models.nets import densenet

#%%

def create_densenet_model(pretrained, drop_rate, num_classes):
    backbone = densenet.densenet121(pretrained=pretrained, drop_rate=drop_rate)
    in_features = backbone.classifier.in_features 
    backbone.classifier = nn.Linear(in_features, num_classes)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(backbone)
    return model


def evaluate_model_performance(test_model,TestLoader,device):
    test_model.to(device)
    test_model.eval() 
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():  
        for test_inputs, test_labels, _ in tqdm(TestLoader):
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

            outputs, _ = test_model(test_inputs)

            probs = torch.softmax(outputs, dim=1)  
            _, preds = torch.max(outputs, 1)  
    

            all_labels.extend(np.argmax(test_labels.cpu().numpy(), axis=1))
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays for metric calculation
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    

    test_accuracy = accuracy_score(all_labels, all_preds)
    test_f1_score = f1_score(all_labels, all_preds, average='macro')
    each_test_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average=None)
    avg_test_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    

    
    

    return test_accuracy, test_f1_score, each_test_auc, avg_test_auc


#%%
# 설정 값을 EasyDict로 정의
args = EasyDict({
    "data_root": "/home/sg980429/research/All_data",
    "save_dir": "../save_files",
    'data' : 'ISIC2018',
    "lbl_ratio": 0.1,
    "resize": 224,
    "batch_size": 32,
    "num_workers": 4,
    "gpu_num": 1,
    'gpu': 1,
    'arch': 'DenseNet',
    "drop_rate": 0.5,
    "lr": 1e-3,
    "ft_lr": 1e-3,
    'num_classes':7,
    "num_epochs": 128,
    'RL': 'PPO',
    "episode_size": 16,
    "mini_batch_size": 4,
    "mini_num_epochs": 4,
    "controller_gpu": 2,
    "experiment_num": 789,
    # "experiment_num": datetime.now(timezone('Asia/Seoul')).strftime("%Y%m%d_%H%M%S"),

    "topk" : 50,
    })

#%%
class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
_, val_df, test_df = load_ISIC2018_GT(args.data_root)
val_indexes = np.random.permutation(len(val_df))
test_indexes = np.arange(len(test_df))
num_classes = len(test_df['GT'][0])

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop((args.resize, args.resize), scale=(0.8, 1.0)),  # Crop with a conservative scale to retain lesions
        transforms.RandomHorizontalFlip(),  # Horizontal flip
        transforms.RandomVerticalFlip(),  # Vertical flip
        transforms.RandomRotation(degrees=15),  # Rotate by a small degree
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # Slight color adjustments
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),  # Gentle Gaussian Blur
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)



test_transform = transforms.Compose(
    [
        transforms.Resize((args.resize,args.resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

#%%

val_dataset = labeled_dataset_from_path(val_df, val_indexes, transforms=test_transform)
test_dataset = labeled_dataset_from_path(test_df, test_indexes, transforms=test_transform)

ValLoader = DataLoader(val_dataset, 
                  batch_size=args.batch_size,
                  num_workers=args.num_workers)

TestLoader = DataLoader(test_dataset, 
                  batch_size=args.batch_size,
                  num_workers=args.num_workers)

#%%

# exp_num = [123,234,345,456,567,678,789,890,1111,2222,3333,4444]

args.gpu_num = 3
device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")

exp_num = [123,234,345,456,567,678,789,890]
methods = ['confidence', 'contribution']
threshold = 0.9

exp_num = [123]
exp_num = [234]
exp_num = [345]
exp_num = [456]
exp_num = [567]
exp_num = [678]
exp_num = [789]
exp_num = [890]




#%%

auc_scores = {method : [] for method in methods}

auc_each_class = {} 

model_densenet = create_densenet_model(pretrained=True, drop_rate=args.drop_rate,num_classes=args.num_classes)
ImageNet_PreTrained_weights = copy.deepcopy(model_densenet.state_dict())

criterion = nn.CrossEntropyLoss()

#%%
for method in methods:
    for x in exp_num:
        
        SSL_dir = os.path.join(args.save_dir,"SSL_dir",f"exp_{x}",f"{method}")
        writer = SummaryWriter(SSL_dir) 
        
        model_densenet.load_state_dict(ImageNet_PreTrained_weights)
        
        args.experiment_num = x
        exp_dir = os.path.join(args.save_dir,f'{args.data}_{args.arch}_{args.RL}_{args.experiment_num}')
        
        config_path = os.path.join(SSL_dir,"config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(vars(args), f)

        df_path = os.path.join(exp_dir,"train_df.pkl")
        training_df = pd.read_pickle(df_path)
        
        lbl_df = training_df[training_df['contribution'].isna()].copy()
        ulb_df = training_df[~training_df['contribution'].isna()].copy()
        
        # pseudo-labeled data selection
        ulb_score = np.array(ulb_df[method].tolist())
        mask = ulb_score >= threshold
        ulb_df = ulb_df[mask].reset_index(drop=True)
        
        
        SSL_df = pd.concat([lbl_df, ulb_df], axis=0, ignore_index=True)
        SSL_indexes = np.arange(len(SSL_df))
        SSL_dataset = labeled_dataset_from_path(SSL_df,SSL_indexes,transforms=train_transform)
        
        SSL_TrainLoader = DataLoader(SSL_dataset, 
                  batch_size=args.batch_size,
                  num_workers=args.num_workers)
        
        
        optimizer = torch.optim.Adam(
            model_densenet.parameters(),
            lr=args.ft_lr,
            betas=(0.9, 0.99),
            eps=0.1,
        )
        
        model_densenet.to(device)
        best_loss = 100
        best_acc = 0
        best_F1 = 0
        best_auc = 0
        for epoch in tqdm(range(args.num_epochs)):
        
            running_loss = 0.0
            model_densenet.train() 
            for inputs, labels, idx in SSL_TrainLoader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs, _ = model_densenet(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
                running_loss += loss.item()
            avg_train_loss = running_loss / len(SSL_TrainLoader)
            writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        
            model_densenet.eval()  # Set model to evaluation mode
            val_loss = 0.0
            all_labels = []
            all_preds = []
            all_probs = []
            
            with torch.no_grad():
                for val_inputs, val_labels, _ in ValLoader:
                    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                    val_outputs, _ = model_densenet(val_inputs)
                    loss = criterion(val_outputs, val_labels)
                    val_loss += loss.item()
                    
                    # Get the predicted labels and probabilities
                    probs = torch.softmax(val_outputs, dim=1)
                    _, preds = torch.max(val_outputs, 1)
        
                                    
                    # all_labels.extend(val_labels.cpu().numpy())
                    all_labels.extend(np.argmax(val_labels.cpu().numpy(), axis=1))
                    all_preds.extend(preds.cpu().numpy())
        
                    all_probs.extend(probs.cpu().numpy())
            
        
            all_labels = np.array(all_labels)
            all_preds = np.array(all_preds)
            all_probs = np.array(all_probs)
        
        
            avg_val_loss = val_loss / len(ValLoader)
            val_accuracy = accuracy_score(all_labels, all_preds)
            val_f1_score = f1_score(all_labels, all_preds, average='macro')
            val_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr',average='macro')
        
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
            writer.add_scalar('F1-Score/Validation', val_f1_score, epoch)
            writer.add_scalar('AUC/Validation', val_auc, epoch)
            
            state_dict = {
                "epoch": epoch + 1,
                "weights": model_densenet.state_dict(),
                "val_loss": avg_val_loss,
                "val_accuracy": val_accuracy,
                "val_f1_score": val_f1_score,
                "val_auc": val_auc,
            }
        
        
        
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                # torch.save(state_dict, os.path.join(teacher_model_dir,"best_loss.pth"))
            if val_accuracy >  best_acc:
                best_acc = val_accuracy
                # torch.save(state_dict, os.path.join(teacher_model_dir,"best_acc.pth"))
            if  val_f1_score > best_F1:
                best_F1 = val_f1_score
                # torch.save(state_dict, os.path.join(teacher_model_dir,"best_F1.pth"))
            if val_auc > best_auc:
                best_auc = val_auc
                best_model_wts = copy.deepcopy(model_densenet.state_dict())
                torch.save(state_dict, os.path.join(SSL_dir,"best_auc.pth"))
                
        writer.close()  
        
        model_densenet.load_state_dict(best_model_wts)
        ttest_accuracy, test_f1_score, each_test_auc, avg_test_auc = evaluate_model_performance(model_densenet,TestLoader,device)
        
 
        for i, name in enumerate(class_names):
            auc_each_class[name] = each_test_auc[i]
        auc_each_class["avg_macro"] = avg_test_auc
        
        auc_df = pd.DataFrame(list(auc_each_class.items()), columns=['Class', 'AUC'])
        auc_df.to_excel(os.path.join(SSL_dir,'auc_each_class.xlsx'), index=False)

#%%
teacher_model_dir = os.path.join(exp_dir,"teacher_model")
teacher_model_path = os.path.join(teacher_model_dir,"best_auc.pth")
check_point = torch.load(teacher_model_path)

model_densenet.load_state_dict(check_point['weights'])
# model_densenet.load_state_dict(best_model_wts)
ttest_accuracy, test_f1_score, each_test_auc, avg_test_auc = evaluate_model_performance(model_densenet,TestLoader,device)
print(each_test_auc)
print(avg_test_auc)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        



