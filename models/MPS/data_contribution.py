import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import gymnasium as gym
from gymnasium import spaces
import copy

from datasets.dataset import labeled_dataset_from_path
from datasets import data_augmentations

import time


""" pseudo-labeled data의 가치를 측정하는 강화학습 환경(environment)
gym에서 env는 크게 reset, step으로 이루어져 있다.
reset에서는 학습하는 경로(trajectory)를 정의한다.
step, 즉 각각의 state마다 observed state(다음 state), reward를 정의해줘야 한다.
"""
class Contribuion_Evaluation(gym.Env):
    def __init__(self, args, train_df, ulb_indexes, val_loader, predictor):
        super(Contribuion_Evaluation,self).__init__()
        self.args = args
        self.device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")
        self.train_transform, self.test_transform = data_augmentations.get_transform(args)

        self.train_df = train_df.copy()
        self.train_df['contributions'] = [[] for _ in range(len(self.train_df))]
        self.feature_map_size = len(self.train_df.loc[ulb_indexes[0],'feature_map'])
        
        self.ulb_indexes = ulb_indexes

        self.num_classes = len(self.train_df.loc[ulb_indexes[0],'labels'])


        # state의 shape을 정의한다
        self.observation_space = spaces.Box(low=0,high=1,shape=(self.feature_map_size+self.num_classes,),dtype=np.float32)
        # action의 shape을 정의한다
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        # 모델 불러오기
        self.predictor = predictor
        self.optimizer = torch.optim.Adam(
                    self.predictor.parameters(),
                    lr=self.args.lr,
                    betas=(0.9, 0.99),
                    eps=0.1,
                )
        self.criterion = nn.CrossEntropyLoss(reduction="none")

        self.val_loader = val_loader

        self.task_predictor_dir = os.path.join(args.exp_dir,"task_predictor")

        
        self.num_window = 10
        self.val_metric_list = [0.5]*self.num_window
        
        
        
        self.best_loss = 100
        self.best_acc = 0
        self.best_F1 = 0
        self.best_auc = 0

        self.time_steps = 0
        


    def step(self, action):
        

        # action(데이터를 학습에 넣는다 or 넣지 않는다)을 저장
        self.actions_list.append(action)
        self.num_count += 1
        self.time_steps += 1
        truncated, info = False,{}

        if len(self.actions_list) < len(self.training_idx):
            # action의 수가 학습에 필요한 정도의 수가 아니라면 reward를 0으로 하고 다음 스텝으로 넘어간다.
             
            reward = 0
            done = False
            obs = self.all_feature_maps[self.num_count]
                    
            return obs, reward, done, truncated, info
        else:
            """action의 수가 학습하려는 데이터의 수와 같은 경우
              해당 action값을 통해 데이터를 선별하여 학습에 사용한다."""
            self.val_metric_list = self.val_metric_list[-self.num_window:]
            moving_avg = np.mean(self.val_metric_list)  
            
            self.train_df.loc[self.training_idx,'contribution'] = self.actions_list
            for i, idx in enumerate(self.training_idx):
                self.train_df.at[idx, 'contributions'].append(self.actions_list[i].item())
            
            pl_train = labeled_dataset_from_path(self.train_df, self.training_idx, transforms= self.train_transform)
            pl_loader = DataLoader(pl_train, batch_size=self.args.mini_batch_size)
            

            ############################train############################################
            self.predictor.to(self.device)
            self.predictor.train() 
            for epoch in range(self.args.mini_num_epochs):     
                for inputs, pseudo_labels, idx in pl_loader:
                    data_contribution = torch.tensor(self.train_df.loc[idx.tolist(), 'contribution'].tolist(), device=self.device)
                    # action_mask = torch.bernoulli(data_contribution).reshape(-1, 1, 1) 
                    action_mask = torch.bernoulli(data_contribution)
                    # print(torch.bernoulli(data_contribution).shape)

     
                    
                    inputs, pseudo_labels = inputs.to(self.device), pseudo_labels.to(self.device)
                    inputs = inputs
                    pseudo_labels = pseudo_labels


                    self.optimizer.zero_grad()
                    outputs, _ = self.predictor(inputs)
                    loss_per_sample = self.criterion(outputs, pseudo_labels)


                   
                    
                    weighted_loss = loss_per_sample*action_mask

     
                    loss = weighted_loss.mean()
                
                    
            
                    loss.backward()
                    self.optimizer.step()
                                     
            ########################eval####################################################
            self.predictor.eval() 
            val_loss = 0.0
            all_labels = []
            all_preds = []
            all_probs = []
            
            with torch.no_grad():
                for val_inputs, val_labels, _ in self.val_loader:
                    val_inputs, val_labels = val_inputs.to(self.device), val_labels.to(self.device)
                    val_outputs, _ = self.predictor(val_inputs)
                    loss = self.criterion(val_outputs, val_labels)
                    val_loss += loss.mean().item()
                    
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
    
    
            avg_val_loss = val_loss 
            val_accuracy = accuracy_score(all_labels, all_preds)
            val_f1_score = f1_score(all_labels, all_preds, average='macro')
            
    
            if self.num_classes > 2:
                val_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
            else:
                val_auc = roc_auc_score(all_labels, all_probs[:, 1])  
                               
                    
            ####################check point ################################
            weightes = copy.deepcopy(self.predictor.state_dict())
            checkpoint = {
                "steps": self.time_steps,
                "weights": weightes,
                "val_loss": avg_val_loss,
                "val_accuracy": val_accuracy,
                "val_f1_score": val_f1_score,
                "val_auc": val_auc,
            }
    
      
            if avg_val_loss < self.best_loss:
                self.best_loss = avg_val_loss
                # torch.save(checkpoint, os.path.join(self.task_predictor_dir,"best_loss.pth"))
            if val_accuracy >  self.best_acc:
                self.best_acc = val_accuracy
                # torch.save(checkpoint, os.path.join(self.task_predictor_dir,"best_acc.pth"))
            if  val_f1_score > self.best_F1:
                self.best_F1 = val_f1_score
                # torch.save(checkpoint, os.path.join(self.task_predictor_dir,"best_F1.pth"))
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                torch.save(checkpoint, os.path.join(self.task_predictor_dir,"best_auc.pth"))
                
            
####################################################################
            # score = -avg_val_loss
            # score = val_accuracy
            # score = val_f1_score
            score = val_auc
            
            reward = score - moving_avg
            self.val_metric_list.append(score)    
            """*************reward의 경우 'validation set에서의 AUC score의 향상하는 정도'로 디자인 한 것이다.
            이 reward를 어떻게 디자인할 지... 고민해봐야 함.***********"""
   
            done = True   
            obs = np.random.randn(*self.all_feature_maps[0].shape) 
            
            
            return obs, reward, done, truncated, info



    def reset(self,seed=None, options=None):        
        shuffled_idx = self.ulb_indexes.copy()
        random.shuffle(shuffled_idx)
        ulb_idx = shuffled_idx[:self.args.episode_size]
        self.training_idx = ulb_idx
                
        
        pseudo_labels = torch.FloatTensor(self.train_df.loc[self.training_idx ,'labels'].tolist())
        feature_maps = torch.FloatTensor(self.train_df.loc[self.training_idx ,'feature_map'].tolist())
        self.all_feature_maps = torch.cat((feature_maps, pseudo_labels), dim=1)

        
              
        self.actions_list = []
        self.num_count = 0
        info = {}
        obs = self.all_feature_maps[self.num_count]


        return obs, info        

    def render(self):
        raise NotImplemented

    def close(self):
        raise NotImplemented




class PL_data_valuation_env(gym.Env):
    def __init__(self, args, train_df, ulb_indexes, val_loader, predictor):
        super(PL_data_valuation_env,self).__init__()
        self.args = args
        self.device = torch.device(f"cuda:{args.controller_gpu}" if torch.cuda.is_available() else "cpu")
        self.train_transform, self.test_transform = data_augmentations.get_transform(args)

        self.train_df = train_df.copy()
        self.train_df['contributions'] = [[] for _ in range(len(self.train_df))]
        
        self.ulb_indexes = ulb_indexes

        self.num_classes = len(self.train_df.loc[ulb_indexes[0],'labels'])


        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        # reset() now returns a tuple: the state and information
        self.observation_shape = self.reset()[0]["image"].shape
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=-1., high=1., shape=self.observation_shape, dtype=np.float32),
            "label" : spaces.MultiBinary(self.num_classes)
        })

        self.predictor = predictor
        self.optimizer = torch.optim.Adam(
                    self.predictor.parameters(),
                    lr=self.args.lr,
                    betas=(0.9, 0.99),
                    eps=0.1,
                )
        self.criterion = nn.CrossEntropyLoss(reduction="none")

        self.val_loader = val_loader

        exp_dir = os.path.join(args.save_dir,f'{args.data}_{args.arch}_{args.RL}_{args.experiment_num}')
        self.task_predictor_dir = os.path.join(exp_dir,"task_predictor")

        
        self.num_window = 1
        self.val_metric_list = [0.5]*self.num_window
        
        
        
        self.best_loss = 100
        self.best_acc = 0
        self.best_F1 = 0
        self.best_auc = 0

        self.time_steps = 0


        
    def step(self, action):
        

        self.actions_list.append(action)
        self.num_count += 1
        self.time_steps += 1
        truncated, info = False,{}
        if len(self.actions_list) < self.pl_eval.__len__() :
            
            reward = 0
            done = False
            obs = {"image": self.pl_eval[self.num_count][0], "label": self.pl_eval[self.num_count][1]}

            return obs, reward, done, truncated, info
        else:
            self.val_metric_list = self.val_metric_list[-self.num_window:]
            moving_avg = np.mean(self.val_metric_list)  
            
            self.train_df.loc[self.training_idx,'contribution'] = self.actions_list
            for i, idx in enumerate(self.training_idx):
                self.train_df.at[idx, 'contributions'].append(self.actions_list[i].item())
            
            pl_train = labeled_dataset_from_path(self.train_df, self.training_idx, transforms= self.train_transform)
            pl_loader = DataLoader(pl_train, batch_size=self.args.mini_batch_size)
            

            ############################train############################################
            self.predictor.to(self.device)
            self.predictor.train() 
            for epoch in range(self.args.mini_num_epochs):     
                for inputs, pseudo_labels, idx in pl_loader:
                    data_contribution = torch.tensor(self.train_df.loc[idx.tolist(), 'contribution'].tolist(), device=self.device)
                    # action_mask = torch.bernoulli(data_contribution).reshape(-1, 1, 1) 
                    action_mask = torch.bernoulli(data_contribution)
                    # print(torch.bernoulli(data_contribution).shape)

     
                    
                    inputs, pseudo_labels = inputs.to(self.device), pseudo_labels.to(self.device)
                    inputs = inputs
                    pseudo_labels = pseudo_labels


                    self.optimizer.zero_grad()
                    outputs, _ = self.predictor(inputs)
                    loss_per_sample = self.criterion(outputs, pseudo_labels)


                   
                    
                    weighted_loss = loss_per_sample*action_mask
     
                    loss = weighted_loss.mean()
                
                    
            
                    loss.backward()
                    self.optimizer.step()

            ########################eval####################################################
            self.predictor.eval() 
            val_loss = 0.0
            all_labels = []
            all_preds = []
            all_probs = []
            
            with torch.no_grad():
                for val_inputs, val_labels, _ in self.val_loader:
                    val_inputs, val_labels = val_inputs.to(self.device), val_labels.to(self.device)
                    val_outputs, _ = self.predictor(val_inputs)
                    loss = self.criterion(val_outputs, val_labels)
                    val_loss += loss.mean().item()
                    
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
    
    
            avg_val_loss = val_loss 
            val_accuracy = accuracy_score(all_labels, all_preds)
            val_f1_score = f1_score(all_labels, all_preds, average='macro')
            
    
            if self.num_classes > 2:
                val_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
            else:
                val_auc = roc_auc_score(all_labels, all_probs[:, 1])  
                               
                

        ####################check point ################################
            weightes = copy.deepcopy(self.predictor.state_dict())
            checkpoint = {
                "steps": self.time_steps,
                "weights": weightes,
                "val_loss": avg_val_loss,
                "val_accuracy": val_accuracy,
                "val_f1_score": val_f1_score,
                "val_auc": val_auc,
            }
    
      
            if avg_val_loss < self.best_loss:
                self.best_loss = avg_val_loss
                # torch.save(checkpoint, os.path.join(self.task_predictor_dir,"best_loss.pth"))
            if val_accuracy >  self.best_acc:
                self.best_acc = val_accuracy
                # torch.save(checkpoint, os.path.join(self.task_predictor_dir,"best_acc.pth"))
            if  val_f1_score > self.best_F1:
                self.best_F1 = val_f1_score
                # torch.save(checkpoint, os.path.join(self.task_predictor_dir,"best_F1.pth"))
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                torch.save(checkpoint, os.path.join(self.task_predictor_dir,"best_auc.pth"))
                
            
####################################################################
            # score = -avg_val_loss
            # score = val_accuracy
            # score = val_f1_score
            score = val_auc
            
            reward = score - moving_avg
            self.val_metric_list.append(score)    
           
            # print(avg_val_loss,val_accuracy,val_f1_score,val_auc)
   
            done = True   
            obs = {"image": np.random.randn(*self.pl_eval[0][0].shape), "label": np.random.randn(*self.pl_eval[0][1].shape)}      

            return obs, reward, done, truncated, info



    def reset(self,seed=None, options=None):

        shuffled_idx = self.ulb_indexes.copy()
        random.shuffle(shuffled_idx)
        ulb_idx = shuffled_idx[:self.args.episode_size]
        self.training_idx = ulb_idx
        
        self.pl_eval = labeled_dataset_from_path(self.train_df, self.training_idx, transforms=self.test_transform)

              
        self.actions_list = []
        self.num_count = 0
        info = {}
        obs = {"image": self.pl_eval[self.num_count][0], "label": self.pl_eval[self.num_count][1]}


        return obs, info
        


    def render(self):
        raise NotImplemented

    def close(self):
        raise NotImplemented
    


class Contribuion_Evaluation_training_time(gym.Env):
    def __init__(self, args, train_df, ulb_indexes, val_loader, predictor):
        super(Contribuion_Evaluation,self).__init__()
        self.args = args
        self.device = torch.device(f"cuda:{args.controller_gpu}" if torch.cuda.is_available() else "cpu")
        self.train_transform, self.test_transform = data_augmentations.get_transform(args)

        self.train_df = train_df.copy()
        self.train_df['contributions'] = [[] for _ in range(len(self.train_df))]
        self.feature_map_size = len(self.train_df.loc[ulb_indexes[0],'feature_map'])
        
        self.ulb_indexes = ulb_indexes

        self.num_classes = len(self.train_df.loc[ulb_indexes[0],'labels'])



        self.observation_space = spaces.Box(low=0,high=1,shape=(self.feature_map_size+self.num_classes,),dtype=np.float32)
        
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        self.predictor = predictor
        self.optimizer = torch.optim.Adam(
                    self.predictor.parameters(),
                    lr=self.args.lr,
                    betas=(0.9, 0.99),
                    eps=0.1,
                )
        self.criterion = nn.CrossEntropyLoss(reduction="none")

        self.val_loader = val_loader

        exp_dir = os.path.join(args.save_dir,f'{args.data}_{args.arch}_{args.RL}_{args.experiment_num}')
        self.task_predictor_dir = os.path.join(exp_dir,"task_predictor")

        
        self.num_window = 10
        self.val_metric_list = [0.5]*self.num_window
        
        
        
        self.best_loss = 100
        self.best_acc = 0
        self.best_F1 = 0
        self.best_auc = 0

        self.time_steps = 0
        
        self.time_init = time.time()
        self.time_episode_step_end = time.time()

    def step(self, action):
        
        self.time_step_start = time.time()
        
        self.actions_list.append(action)
        self.num_count += 1
        self.time_steps += 1
        truncated, info = False,{}
        # if len(self.actions_list) < self.pl_eval.__len__() :
        if len(self.actions_list) < len(self.training_idx):
             
            reward = 0
            done = False
            obs = self.all_feature_maps[self.num_count]
            
            self.time_step_with_no_training = time.time()
            print(f"time no training: {self.time_step_with_no_training-self.time_step_start} sec")

            
            return obs, reward, done, truncated, info
        else:
            self.val_metric_list = self.val_metric_list[-self.num_window:]
            moving_avg = np.mean(self.val_metric_list)  
            
            self.train_df.loc[self.training_idx,'contribution'] = self.actions_list
            for i, idx in enumerate(self.training_idx):
                self.train_df.at[idx, 'contributions'].append(self.actions_list[i].item())
            
            pl_train = labeled_dataset_from_path(self.train_df, self.training_idx, transforms= self.train_transform)
            pl_loader = DataLoader(pl_train, batch_size=self.args.mini_batch_size)
            
            self.time_training_start = time.time()  
            ############################train############################################
            self.predictor.to(self.device)
            self.predictor.train() 
            for epoch in range(self.args.mini_num_epochs):     
                for inputs, pseudo_labels, idx in pl_loader:
                    data_contribution = torch.tensor(self.train_df.loc[idx.tolist(), 'contribution'].tolist(), device=self.device)
                    # action_mask = torch.bernoulli(data_contribution).reshape(-1, 1, 1) 
                    action_mask = torch.bernoulli(data_contribution)
                    # print(torch.bernoulli(data_contribution).shape)

     
                    
                    inputs, pseudo_labels = inputs.to(self.device), pseudo_labels.to(self.device)
                    inputs = inputs
                    pseudo_labels = pseudo_labels


                    self.optimizer.zero_grad()
                    outputs, _ = self.predictor(inputs)
                    loss_per_sample = self.criterion(outputs, pseudo_labels)


                   
                    
                    weighted_loss = loss_per_sample*action_mask

     
                    loss = weighted_loss.mean()
                
                    
            
                    loss.backward()
                    self.optimizer.step()
                    
            self.time_training_end = time.time()
            print(f"time for training: {self.time_training_end-self.time_training_start} sec")                     
            ########################eval####################################################
            self.predictor.eval() 
            val_loss = 0.0
            all_labels = []
            all_preds = []
            all_probs = []
            
            with torch.no_grad():
                for val_inputs, val_labels, _ in self.val_loader:
                    val_inputs, val_labels = val_inputs.to(self.device), val_labels.to(self.device)
                    val_outputs, _ = self.predictor(val_inputs)
                    loss = self.criterion(val_outputs, val_labels)
                    val_loss += loss.mean().item()
                    
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
    
    
            avg_val_loss = val_loss 
            val_accuracy = accuracy_score(all_labels, all_preds)
            val_f1_score = f1_score(all_labels, all_preds, average='macro')
            
    
            if self.num_classes > 2:
                val_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
            else:
                val_auc = roc_auc_score(all_labels, all_probs[:, 1])  
                               
                
            self.time_evaluation_end = time.time()
            print(f"time for evaluation: {self.time_evaluation_end-self.time_training_end} sec")
        
            ####################check point ################################
            weightes = copy.deepcopy(self.predictor.state_dict())
            checkpoint = {
                "steps": self.time_steps,
                "weights": weightes,
                "val_loss": avg_val_loss,
                "val_accuracy": val_accuracy,
                "val_f1_score": val_f1_score,
                "val_auc": val_auc,
            }
    
      
            if avg_val_loss < self.best_loss:
                self.best_loss = avg_val_loss
                # torch.save(checkpoint, os.path.join(self.task_predictor_dir,"best_loss.pth"))
            if val_accuracy >  self.best_acc:
                self.best_acc = val_accuracy
                # torch.save(checkpoint, os.path.join(self.task_predictor_dir,"best_acc.pth"))
            if  val_f1_score > self.best_F1:
                self.best_F1 = val_f1_score
                # torch.save(checkpoint, os.path.join(self.task_predictor_dir,"best_F1.pth"))
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                torch.save(checkpoint, os.path.join(self.task_predictor_dir,"best_auc.pth"))
                
            
####################################################################
            # score = -avg_val_loss
            # score = val_accuracy
            # score = val_f1_score
            score = val_auc
            
            reward = score - moving_avg
            self.val_metric_list.append(score)    
           
            # print(avg_val_loss,val_accuracy,val_f1_score,val_auc)
   
            done = True   
            obs = np.random.randn(*self.all_feature_maps[0].shape) 
            
            self.time_episode_step_end = time.time()
            print(f"time for last episode step: {self.time_episode_step_end-self.time_step_start} sec") 
            
            return obs, reward, done, truncated, info



    def reset(self,seed=None, options=None):
        self.time_reset = time.time()
        print(f"time for reset: {self.time_reset-self.time_episode_step_end} sec")
        
        shuffled_idx = self.ulb_indexes.copy()
        random.shuffle(shuffled_idx)
        ulb_idx = shuffled_idx[:self.args.episode_size]
        self.training_idx = ulb_idx
                
        
        pseudo_labels = torch.FloatTensor(self.train_df.loc[self.training_idx ,'labels'].tolist())
        feature_maps = torch.FloatTensor(self.train_df.loc[self.training_idx ,'feature_map'].tolist())
        self.all_feature_maps = torch.cat((feature_maps, pseudo_labels), dim=1)

        
              
        self.actions_list = []
        self.num_count = 0
        info = {}
        obs = self.all_feature_maps[self.num_count]


        return obs, info        




    def render(self):
        raise NotImplemented

    def close(self):
        raise NotImplemented