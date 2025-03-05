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
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
from gymnasium import spaces
import copy

from torch.utils.tensorboard import SummaryWriter  # TensorBoard writer

from load_ISIC2018 import load_ISIC2018_GT
from dataset_from_df import labeled_dataset_from_path, unlabeled_dataset_from_path
from model_evaluation import evaluate_model_performance
import opts

args = opts.get_opts()

#%% hyperparameters

data_root = args.data_root
save_dir = args.save_dir

lbl_ratio = args.lbl_ratio
resize = args.resize
batch_size = args.batch_size
num_workers = args.num_workers
gpu_num = args.gpu_num
lr = args.lr
num_epochs = args.num_epochs
episode_size = args.episode_size
mini_batch_size = args.mini_batch_size
mini_num_epochs = args.mini_num_epochs

controller_gpu = args.controller_gpu

#%% experiments

experiment_num = args.experiment_num
device = torch.device(f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu")

exp_dir = os.path.join(save_dir,f'{"isic"}_{"vgg"}_{"PPO"}_{experiment_num}')
if not os.path.exists(exp_dir):
    os.mkdir(exp_dir)
#%% data load


train_df, val_df, test_df = load_ISIC2018_GT(data_root)
num_classes = len(test_df['GT'][0])

#%% data split for SSL
np.random.seed(123)
train_indexes = np.random.permutation(len(train_df))
val_indexes = np.random.permutation(len(val_df))
test_indexes = np.arange(len(test_df))

num_labels = int(len(train_df)*lbl_ratio)
lbl_indexes = train_indexes[:num_labels]
ulb_indexes = train_indexes[num_labels:]

train_df.loc[ulb_indexes, 'labels'] = None
train_df['labels'] = train_df['labels'].astype(object)  

#%% data pre-processing

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop((resize,resize), scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((resize,resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)


lbl_dataset = labeled_dataset_from_path(train_df,lbl_indexes,transforms=train_transform)

val_dataset = labeled_dataset_from_path(val_df, val_indexes, transforms=test_transform)
test_dataset = labeled_dataset_from_path(test_df, test_indexes, transforms=test_transform)


#%% data loader
WarmUpTrainLoader = DataLoader(lbl_dataset, 
                  batch_size=batch_size,
                  num_workers=num_workers)

ValLoader = DataLoader(val_dataset, 
                  batch_size=batch_size,
                  num_workers=num_workers)

TestLoader = DataLoader(test_dataset, 
                  batch_size=batch_size,
                  num_workers=num_workers)


#%% make teacher model

teacher_model_dir = os.path.join(exp_dir,"teacher_model")
SL_log_dir = os.path.join(exp_dir,"SL_log")

if not os.path.exists(teacher_model_dir):
    os.mkdir(teacher_model_dir)
    
    
    backbone = models.vgg19(weights=True)
    dim_feature = backbone.classifier[-1].in_features
    backbone.classifier[-1] = nn.Linear(in_features=dim_feature, out_features=num_classes)

    optimizer = torch.optim.Adam(
                backbone.parameters(),
                lr=lr,
                betas=(0.9, 0.99),
                eps=0.1,
            )
    
    criterion = nn.CrossEntropyLoss()
    
    writer = SummaryWriter(SL_log_dir)  # TensorBoard SummaryWriter
    backbone.to(device)
    best_loss = 100
    best_acc = 0
    best_F1 = 0
    best_auc = 0
    for epoch in tqdm(range(num_epochs)):

        running_loss = 0.0
        backbone.train() 
        for inputs, labels, idx in WarmUpTrainLoader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = backbone(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        avg_train_loss = running_loss / len(WarmUpTrainLoader)
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)

        backbone.eval()  # Set model to evaluation mode
        val_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for val_inputs, val_labels, _ in ValLoader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = backbone(val_inputs)
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
            "weights": backbone.state_dict(),
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
            "val_f1_score": val_f1_score,
            "val_auc": val_auc,
        }


  
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(state_dict, os.path.join(teacher_model_dir,"best_loss.pth"))
        if val_accuracy >  best_acc:
            best_acc = val_accuracy
            torch.save(state_dict, os.path.join(teacher_model_dir,"best_acc.pth"))
        if  val_f1_score > best_F1:
            best_F1 = val_f1_score
            torch.save(state_dict, os.path.join(teacher_model_dir,"best_F1.pth"))
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(state_dict, os.path.join(teacher_model_dir,"best_auc.pth"))

    writer.close()  
    
#%% pre-trained teacher model test

teacher_model = models.vgg19(weights=False)
dim_feature = teacher_model.classifier[-1].in_features
teacher_model.classifier[-1] = nn.Linear(in_features=dim_feature, out_features=num_classes)

#%%
metrics = ["loss","acc", "F1", "auc"]
m = 3

teacher_model_path = os.path.join(teacher_model_dir,f"best_{metrics[m]}.pth")
check_point = torch.load(teacher_model_path)

teacher_model.load_state_dict(check_point['weights'])

test_accuracy, test_f1_score, test_auc = evaluate_model_performance(teacher_model,TestLoader,device)


#%%

lbl_test = labeled_dataset_from_path(train_df,lbl_indexes,transforms=test_transform)

lbltest_loader = DataLoader(lbl_test, 
                  batch_size=batch_size,
                  num_workers=num_workers)

if 'feature_map' not in train_df.columns:
    train_df['feature_map'] = None
train_df['feature_map'] = train_df['feature_map'].astype(object)  

if 'confidence' not in train_df.columns:
    train_df['confidence'] = 0

for inputs,_, idx in tqdm(lbltest_loader):
    inputs = inputs.to(device)
    idx_np = idx.cpu().numpy() 
    outputs = teacher_model(inputs)


    probabilities = F.softmax(outputs, dim=1)
    confidence, predicted_class = torch.max(probabilities, dim=1)
    confidence_np = confidence.detach().cpu().numpy()
    
    
    feature_maps = teacher_model.features(inputs)
    feature_maps = feature_maps.view(feature_maps.size(0), -1).detach().cpu().numpy() 
    
    for i, idx in enumerate(idx_np):
        train_df.at[idx, 'feature_map'] = feature_maps[i]
        train_df.at[idx, 'confidence'] = confidence_np[i]


#%% pseudo-labeling

PL_train_df = train_df.copy()

ulb_dataset = unlabeled_dataset_from_path(PL_train_df,ulb_indexes,transforms=test_transform)
UlbLoader = DataLoader(ulb_dataset, 
                  batch_size=batch_size,
                  num_workers=num_workers)

if 'feature_map' not in PL_train_df.columns:
    PL_train_df['feature_map'] = None
PL_train_df['feature_map'] = PL_train_df['feature_map'].astype(object)  

if 'confidence' not in PL_train_df.columns:
    PL_train_df['confidence'] = 0

for ulb_inputs, ulb_idx in tqdm(UlbLoader):
    ulb_inputs = ulb_inputs.to(device)
    ulb_idx_np = ulb_idx.cpu().numpy() 
    ulb_outputs = teacher_model(ulb_inputs)
    predicted_labels = ulb_outputs.argmax(dim=1)

    probabilities = F.softmax(ulb_outputs, dim=1)
    confidence, predicted_class = torch.max(probabilities, dim=1)
    confidence_np = confidence.detach().cpu().numpy()
    

    one_hot_encoded = F.one_hot(predicted_labels, num_classes=num_classes)  
    one_hot_encoded_np = one_hot_encoded.cpu().numpy()
    
    
    feature_maps = teacher_model.features(ulb_inputs)
    feature_maps = feature_maps.view(feature_maps.size(0), -1).detach().cpu().numpy() 
    
    for i, idx in enumerate(ulb_idx_np):
        PL_train_df.at[idx, 'labels'] = one_hot_encoded_np[i]
        PL_train_df.at[idx, 'feature_map'] = feature_maps[i]
        PL_train_df.at[idx, 'confidence'] = confidence_np[i]


#%% contribution evaluating


task_predictor_dir = os.path.join(exp_dir,"task_predictor")

if not os.path.exists(task_predictor_dir):
    os.mkdir(task_predictor_dir)


class PL_data_valuation_env(gym.Env):
    def __init__(self, train_df, ulb_indexes, val_loader, predictor):
        super(PL_data_valuation_env,self).__init__()
        self.train_df = train_df.copy()
        self.train_df['contributions'] = [[] for _ in range(len(self.train_df))]
        
        self.ulb_indexes = ulb_indexes

        self.num_classes = len(self.train_df.loc[ulb_indexes[0],'labels'])
        self.feature_map_size = len(self.train_df.loc[ulb_indexes[0],'feature_map'])
        
        self.observation_space = spaces.Box(low=0,high=1,shape=(self.feature_map_size+self.num_classes,),dtype=np.float32)
        
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        

        self.predictor = predictor
        self.optimizer = torch.optim.Adam(
                    self.predictor.parameters(),
                    lr=lr,
                    betas=(0.9, 0.99),
                    eps=0.1,
                )
        # self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        
        self.val_loader = val_loader
        
        self.num_mean = 1
        self.val_metric_list = [0.5]*self.num_mean
        
        
        self.best_loss = 100
        self.best_acc = 0
        self.best_F1 = 0
        self.best_auc = 0

        self.time_steps = 0
        
    def reset(self,seed=None, options=None):

        shuffled_idx = self.ulb_indexes.copy()
        random.shuffle(shuffled_idx)
        ulb_idx = shuffled_idx[:episode_size]
        self.training_idx = ulb_idx

        
        
        pseudo_labels = torch.FloatTensor(self.train_df.loc[self.training_idx ,'labels'].tolist())
        feature_maps = torch.FloatTensor(self.train_df.loc[self.training_idx ,'feature_map'].tolist())
        self.all_feature_maps = torch.cat((feature_maps, pseudo_labels), dim=1)

        
              
        self.actions_list = []
        self.num_count = 0
        info = {}
        obs = self.all_feature_maps[self.num_count]


        return obs, info
    
    def step(self, action):
        self.actions_list.append(action)
        self.num_count += 1
        self.time_steps += 1
        truncated, info = False,{}
        if len(self.actions_list) < len(self.all_feature_maps):
            reward = 0
            done = False
            obs = self.all_feature_maps[self.num_count]
            return obs, reward, done, truncated, info
        else:
            self.val_metric_list = self.val_metric_list[-self.num_mean:]
            moving_avg = np.mean(self.val_metric_list)  
            
            self.train_df.loc[self.training_idx,'contribution'] = self.actions_list
            for i, idx in enumerate(self.training_idx):
                self.train_df.at[idx, 'contributions'].append(self.actions_list[i].item())
            
            pl_train = labeled_dataset_from_path(self.train_df, self.training_idx, transforms=train_transform)
            pl_loader = DataLoader(pl_train, batch_size=mini_batch_size)
            
            
            self.predictor.to(device)
            self.predictor.train() 
            for epoch in range(mini_num_epochs):     
                for inputs, pseudo_labels, idx in pl_loader:
                    inputs, pseudo_labels = inputs.to(device), pseudo_labels.to(device)


                    self.optimizer.zero_grad()
                    outputs = self.predictor(inputs)
                    loss_per_sample = self.criterion(outputs, pseudo_labels)

                    data_weight = torch.tensor(self.train_df.loc[idx.tolist(), 'contribution'].tolist(), device=device)
                    weighted_loss = loss_per_sample * data_weight  
     
                    loss = weighted_loss.mean()
                
                    
            
                    loss.backward()
                    self.optimizer.step()
                    
            
            self.predictor.eval() 
            val_loss = 0.0
            all_labels = []
            all_preds = []
            all_probs = []
            
            with torch.no_grad():
                for val_inputs, val_labels, _ in self.val_loader:
                    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                    val_outputs = self.predictor(val_inputs)
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
            
    
            if num_classes > 2:
                val_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
            else:
                val_auc = roc_auc_score(all_labels, all_probs[:, 1])  
                               
                
            weightes = copy.deepcopy(self.predictor.state_dict())
            state_dict = {
                "steps": self.time_steps,
                "weights": weightes,
                "val_loss": avg_val_loss,
                "val_accuracy": val_accuracy,
                "val_f1_score": val_f1_score,
                "val_auc": val_auc,
            }
    
      
            if avg_val_loss < self.best_loss:
                self.best_loss = avg_val_loss
                torch.save(state_dict, os.path.join(task_predictor_dir,"best_loss.pth"))
            if val_accuracy >  self.best_acc:
                self.best_acc = val_accuracy
                torch.save(state_dict, os.path.join(task_predictor_dir,"best_acc.pth"))
            if  val_f1_score > self.best_F1:
                self.best_F1 = val_f1_score
                torch.save(state_dict, os.path.join(task_predictor_dir,"best_F1.pth"))
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                torch.save(state_dict, os.path.join(task_predictor_dir,"best_auc.pth"))
                
            
            
            # score = -avg_val_loss
            # score = val_accuracy
            # score = val_f1_score
            score = val_auc
            
            reward = score - moving_avg
            self.val_metric_list.append(score)    
           
            # print(avg_val_loss,val_accuracy,val_f1_score,val_auc)
            
            done = True
            obs = np.random.randn(*self.all_feature_maps[0].shape)           

            return obs, reward, done, truncated, info

#%% task predictor
predictor = models.vgg19(weights=True)
predictor.classifier[-1] = nn.Linear(in_features=dim_feature, out_features=num_classes)     



#%%
df_save = os.path.join(exp_dir,"train_df.pkl")
RL_save = os.path.join(exp_dir,"agent")
RL_log_dir = os.path.join(exp_dir,"RL_log")

env = PL_data_valuation_env(PL_train_df, ulb_indexes, ValLoader, predictor)
my_env = DummyVecEnv([lambda: env])

agent = PPO("MlpPolicy", my_env, tensorboard_log=RL_log_dir, verbose=2, device=f"cuda:{controller_gpu}",learning_rate=0.0001)

RL_steps = len(ulb_indexes)*(num_epochs//mini_num_epochs)
agent.learn(total_timesteps=RL_steps) 
agent.save(RL_save)
env.train_df.to_pickle(df_save)

predictor_weights = copy.deepcopy(env.predictor.state_dict())
task_predictor_path = os.path.join(task_predictor_dir,"final_task_predictor.pth")
torch.save(predictor_weights, task_predictor_path)


#%%


