#%% import 

import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from skimage import io
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from tqdm import tqdm

# 강화 학습을 위한 openai library
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
from gymnasium import spaces


import copy
import yaml

from torch.utils.tensorboard import SummaryWriter  # TensorBoard writer


#----------------------직접 작성한 파일들------------------
from load_ISIC2018 import load_ISIC2018_GT
from dataset_from_df import labeled_dataset_from_path, unlabeled_dataset_from_path
from model_evaluation import evaluate_model_performance
import opts
from datasets import data_augmentations
from models.nets import densenet
from models.MPS.MPS import MultiFE
from models.MPS import data_contribution



#%% hyperparameters
args = opts.get_opts()

args.experiment_num = 9999
args.gpu_num = 3
args.controller_gpu = 3
args.num_epochs = 256
args.seed = args.experiment_num 
args.lr = 1e-4




#%% experiments

device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")
args.exp_dir = os.path.join(args.save_dir,f'{args.data}_{args.arch}_{args.RL}_{args.experiment_num}')
exp_dir = args.exp_dir

#%% save config file
if not os.path.exists(args.exp_dir):
    os.mkdir(args.exp_dir)
    
config_path = os.path.join(args.exp_dir,"config.yaml")
with open(config_path, 'w') as f:
    yaml.dump(vars(args), f)
print(f"Arguments saved to {config_path}")

#%% data load

train_df, val_df, test_df = load_ISIC2018_GT(args.data_root)
num_classes = len(test_df['GT'][0])
args.num_classes = num_classes

#%% data split for SSL
"""------labeled set과 unlabeled set을 비율대로 random split---------"""
np.random.seed(args.seed)
train_indexes = np.random.permutation(len(train_df))
val_indexes = np.random.permutation(len(val_df))
test_indexes = np.arange(len(test_df))

num_labels = int(len(train_df)*args.lbl_ratio)
lbl_indexes = train_indexes[:num_labels]
ulb_indexes = train_indexes[num_labels:]

train_df.loc[ulb_indexes, 'labels'] = None
train_df['labels'] = train_df['labels'].astype(object)  

#%% data pre-processing

"""--------------- transform을 통해 data augmentation 적용 ---------------------------"""

train_transform, test_transform  = data_augmentations.get_transform(args)

lbl_dataset = labeled_dataset_from_path(train_df,lbl_indexes,transforms=train_transform)
val_dataset = labeled_dataset_from_path(val_df, val_indexes, transforms=test_transform)
test_dataset = labeled_dataset_from_path(test_df, test_indexes, transforms=test_transform)


#%% data loader
"""-------초기 teacher model을 학습하기 위해 labeled dataset으로 구성된 loader------"""
WarmUpTrainLoader = DataLoader(lbl_dataset, 
                  batch_size=args.batch_size,
                  num_workers=args.num_workers)
#------------------------------------------------------------------------
ValLoader = DataLoader(val_dataset, 
                  batch_size=args.batch_size,
                  num_workers=args.num_workers)

TestLoader = DataLoader(test_dataset, 
                  batch_size=args.batch_size,
                  num_workers=args.num_workers)



#%%
"""------------------------ 모델 불러오기------------------------"""
def create_densenet_model(pretrained, drop_rate, num_classes):
    backbone = densenet.densenet121(pretrained=pretrained, drop_rate=drop_rate)
    in_features = backbone.classifier.in_features 
    backbone.classifier = nn.Linear(in_features, num_classes)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(backbone)
    return model


#%% make teacher model

teacher_model_dir = os.path.join(exp_dir,"teacher_model")
SL_log_dir = os.path.join(exp_dir,"SL_log")

#%%

""" -------- labeled dataset으로 초기 teacher model 학습 ------------------------ """
teacher_model = create_densenet_model(pretrained=True, drop_rate=args.drop_rate,num_classes=args.num_classes)

if not os.path.exists(teacher_model_dir):
    os.mkdir(teacher_model_dir)
    
    optimizer = torch.optim.Adam(
                teacher_model.parameters(),
                lr=args.lr,
                betas=(0.9, 0.99),
                eps=0.1,
            )
    
    criterion = nn.CrossEntropyLoss()
    
    writer = SummaryWriter(SL_log_dir)  # TensorBoard SummaryWriter
    teacher_model.to(device)
    best_loss = 100
    best_acc = 0
    best_F1 = 0
    best_auc = 0
    for epoch in tqdm(range(args.num_epochs)):

        running_loss = 0.0
        teacher_model.train() 
        for inputs, labels, idx in WarmUpTrainLoader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs, _ = teacher_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        avg_train_loss = running_loss / len(WarmUpTrainLoader)
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)

        teacher_model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for val_inputs, val_labels, _ in ValLoader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs, _ = teacher_model(val_inputs)
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
        val_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')


        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
        writer.add_scalar('F1-Score/Validation', val_f1_score, epoch)
        writer.add_scalar('AUC/Validation', val_auc, epoch)
        
        state_dict = {
            "epoch": epoch + 1,
            "weights": teacher_model.state_dict(),
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
            torch.save(state_dict, os.path.join(teacher_model_dir,"best_auc.pth"))

    writer.close()  
    

#%%
"""---------------- teacher model evaluation ------------------------"""
metrics = ["loss","acc", "F1", "auc"]
m = 3

teacher_model_path = os.path.join(teacher_model_dir,f"best_{metrics[m]}.pth")
check_point = torch.load(teacher_model_path)

teacher_model.load_state_dict(check_point['weights'])

test_accuracy, test_f1_score, test_auc = evaluate_model_performance(teacher_model,TestLoader,device)


#%%

""" labeled dataset의 feature map과 confidence 값을 teacher model을 통해 뽑고 저장. 
데이터 각각의 feature map은 classifier에서 data point로 취급.
모델은 각 클래스에 대한 확률값 출력 그 중 가장 높은 클래스의 확률 값이 confidence가 된다."""

lbl_test = labeled_dataset_from_path(train_df,lbl_indexes,transforms=test_transform)

lbltest_loader = DataLoader(lbl_test, 
                  batch_size=args.batch_size,
                  num_workers=args.num_workers)

if 'feature_map' not in train_df.columns:
    train_df['feature_map'] = None
train_df['feature_map'] = train_df['feature_map'].astype(object)  

if 'confidence' not in train_df.columns:
    train_df['confidence'] = 0

for inputs,_, idx in tqdm(lbltest_loader):
    inputs = inputs.to(device)
    idx_np = idx.cpu().numpy() 
    outputs, feature_maps = teacher_model(inputs)


    probabilities = F.softmax(outputs, dim=1)
    confidence, predicted_class = torch.max(probabilities, dim=1)
    confidence_np = confidence.detach().cpu().numpy()
    
    
    feature_maps = feature_maps.view(feature_maps.size(0), -1).detach().cpu().numpy() 
    
    for i, idx in enumerate(idx_np):
        train_df.at[idx, 'feature_map'] = feature_maps[i]
        train_df.at[idx, 'confidence'] = confidence_np[i]


#%% pseudo-labeling
"""unlabeled data에 대한 feature map, confidence, pseudo-label 저장"""
PL_train_df = train_df.copy()

ulb_dataset = unlabeled_dataset_from_path(PL_train_df,ulb_indexes,transforms=test_transform)
UlbLoader = DataLoader(ulb_dataset, 
                  batch_size=args.batch_size,
                  num_workers=args.num_workers)

if 'feature_map' not in PL_train_df.columns:
    PL_train_df['feature_map'] = None
PL_train_df['feature_map'] = PL_train_df['feature_map'].astype(object)  

if 'confidence' not in PL_train_df.columns:
    PL_train_df['confidence'] = 0

for ulb_inputs, ulb_idx in tqdm(UlbLoader):
    ulb_inputs = ulb_inputs.to(device)
    ulb_idx_np = ulb_idx.cpu().numpy() 
    ulb_outputs, feature_maps = teacher_model(ulb_inputs)
    predicted_labels = ulb_outputs.argmax(dim=1)

    probabilities = F.softmax(ulb_outputs, dim=1)
    confidence, predicted_class = torch.max(probabilities, dim=1)
    confidence_np = confidence.detach().cpu().numpy()
    

    one_hot_encoded = F.one_hot(predicted_labels, num_classes=num_classes)  
    one_hot_encoded_np = one_hot_encoded.cpu().numpy()
    
    
    feature_maps = feature_maps.view(feature_maps.size(0), -1).detach().cpu().numpy() 
    
    for i, idx in enumerate(ulb_idx_np):
        PL_train_df.at[idx, 'labels'] = one_hot_encoded_np[i]
        PL_train_df.at[idx, 'feature_map'] = feature_maps[i]
        PL_train_df.at[idx, 'confidence'] = confidence_np[i]


#%% 


task_predictor_dir = os.path.join(exp_dir,"task_predictor")

if not os.path.exists(task_predictor_dir):
    os.mkdir(task_predictor_dir)


#%% task predictor
"""task predictor는  '모델의 성능을 올리는 데이터' 에서 '모델'을 담당"""
predictor = create_densenet_model(pretrained=False, drop_rate=args.drop_rate,num_classes=args.num_classes)


#%%

if not os.path.exists(task_predictor_dir):
    os.mkdir(task_predictor_dir)
        

df_save = os.path.join(exp_dir,"train_df.pkl")
RL_save = os.path.join(exp_dir,"agent")
RL_log_dir = os.path.join(exp_dir,"RL_log")

# env = data_contribution.PL_data_valuation_env(args, PL_train_df, ulb_indexes, ValLoader, predictor)
""" 
******** pseudo-labeled 데이터의 가치를 측정하는 강화학습 환경(environment) *********
models/MPS/data_contribution.py 파일에서 강화학습 환경 참고
"""
env = data_contribution.Contribuion_Evaluation(args, PL_train_df, ulb_indexes, ValLoader, predictor)

my_env = DummyVecEnv([lambda: env])

# policy_kwargs = dict(
#     features_extractor_class=MultiFE,
#             features_extractor_kwargs={
#               "features_dim":1024, 
#               "nb_classes": 7,
#               "net_size":[128, 128, 64, 32, 16],
#               "in_stride": 1, 
#               "in_padding": 0,
#             }
#             )

# agent = PPO(
#           env=env,
#           learning_rate=args.RL_lr,
#           policy="MultiInputPolicy",
#           policy_kwargs= policy_kwargs,
#           tensorboard_log=RL_log_dir,
#           device=f"cuda:{controller_gpu}",
#           verbose=2
#         )
""" 
********데이터 가치를 측정하는 강화학습 알고리즘 ******** 
알고리즘에는 DQN,DDPG,PPO 등이 있으며, 현재까지 알기로는 PPO가 가장 안정적
강화학습 알고리즘에 대한 비교도 필요할까?
"""
agent = PPO(
          env=env,
          learning_rate=args.RL_lr,
          policy="MlpPolicy",
          tensorboard_log=RL_log_dir,
          device=f"cuda:{args.controller_gpu}",
          verbose=2
        )
RL_steps = len(ulb_indexes)*(args.num_epochs//args.mini_num_epochs)
agent.learn(total_timesteps=RL_steps) 
agent.save(RL_save)
env.train_df.to_pickle(df_save)

predictor_weights = copy.deepcopy(env.predictor.state_dict())
task_predictor_path = os.path.join(exp_dir,"task_predictor.pth")
torch.save(predictor_weights, task_predictor_path)

#%%


