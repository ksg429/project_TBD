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
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from stable_baselines3 import PPO
from easydict import EasyDict
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
import umap
#%%
from load_ISIC2018 import load_ISIC2018_GT
from dataset_from_df import labeled_dataset_from_path, unlabeled_dataset_from_path
from model_evaluation import evaluate_model_performance
import opts
from models.nets import densenet
from datasets import data_augmentations

#%%
def create_densenet_model(pretrained, drop_rate, num_classes):
    backbone = densenet.densenet121(pretrained=pretrained, drop_rate=drop_rate)
    in_features = backbone.classifier.in_features 
    backbone.classifier = nn.Linear(in_features, num_classes)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(backbone)
    return model

def Cls_dist_visualizing(result_df, class_names, name):

    if name == 'pseudo label':
        all_labels = result_df['labels']
    else:
        all_labels = result_df['GT']
    classes_num_data = np.sum(np.array(all_labels), axis=0)

    # Figure와 Axes 객체 생성
    fig, ax = plt.subplots(figsize=(10, 6))

    # Seaborn 막대 그래프 생성
    sns.barplot(x=class_names, y=classes_num_data, color='skyblue', ax=ax)

    # 그래프 요소 설정
    ax.set_title(f'Training Data Class Distribution :: {name}', fontsize=16)
    ax.set_xlabel('Class Names', fontsize=14)
    ax.set_ylabel('Number of Instances', fontsize=14)
    ax.set_xticklabels(class_names, rotation=-45)  # x축 레이블 회전

    # 막대 위에 수치를 표시
    for index, value in enumerate(classes_num_data):
        ax.text(index, value, f'{value}', ha='center', va='bottom')

    # y축 격자 설정
    ax.grid(axis='y', linestyle='--', alpha=0.5)  # y축에 투명한 격자 추가

    # 그래프 출력
    plt.show()
    
def mask_generate(ulb_df,filter,threshold_score=None):
    if filter == 'nothing':
        mask = np.array(range(len(ulb_df)))

    elif filter == 'confidence':
        ulb_score= np.array(ulb_df['confidence'].tolist())
        mask = ulb_score >= threshold_score

    elif filter == 'uncertainty':
        ulb_score= np.array(ulb_df['uncertainty'].tolist())
        mask = ulb_score <= threshold_score

    elif filter == 'energy':
        ulb_score= np.array(ulb_df['energy'].tolist())   
        mask = ulb_score <= threshold_score

    elif filter == 'contribution':
        ulb_score= np.array(ulb_df['contribution'].tolist())
        mask = ulb_score >= threshold_score

    elif filter == 'action':
        ulb_score= np.array(ulb_df['action'].tolist())
        mask = ulb_score >= threshold_score
        

    elif filter == 'contributions':
        ulb_score= np.array(ulb_df['contributions'].apply(lambda x : np.mean(np.array(x[:-3]),axis=0)).tolist())
        mask = ulb_score >= threshold_score
        
    return mask, threshold_score


#%%
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
    "num_epochs": 128,
    'RL': 'PPO',
    "episode_size": 16,
    "mini_batch_size": 4,
    "mini_num_epochs": 4,
    "controller_gpu": 2,
    "experiment_num": 123,
    # "experiment_num": datetime.now(timezone('Asia/Seoul')).strftime("%Y%m%d_%H%M%S"),

    "topk" : 50,
    })


#%%

device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")


#%%
train_df, val_df, test_df = load_ISIC2018_GT(args.data_root)
num_classes = len(test_df['GT'][0])
args.num_classes = num_classes

#%% data split for SSL
args.experiment_num = 890
args.seed = args.experiment_num

exp_dir = os.path.join(args.save_dir,f'{args.data}_{args.arch}_{args.RL}_{args.experiment_num}')

teacher_model_dir = os.path.join(exp_dir,"teacher_model")
task_predictor_dir = os.path.join(exp_dir,"task_predictor")
df_path = os.path.join(exp_dir,"train_df.pkl")

np.random.seed(args.seed)
train_indexes = np.random.permutation(len(train_df))
val_indexes = np.random.permutation(len(val_df))
test_indexes = np.arange(len(test_df))

num_labels = int(len(train_df)*args.lbl_ratio)
lbl_indexes = train_indexes[:num_labels]
ulb_indexes = train_indexes[num_labels:]

train_df.loc[ulb_indexes, 'labels'] = None
train_df['labels'] = train_df['labels'].astype(object)  


train_transform, test_transform  = data_augmentations.get_transform(args)

val_dataset = labeled_dataset_from_path(val_df, val_indexes, transforms=test_transform)
test_dataset = labeled_dataset_from_path(test_df, test_indexes, transforms=test_transform)

ValLoader = DataLoader(val_dataset, 
                  batch_size=args.batch_size,
                  num_workers=args.num_workers)

TestLoader = DataLoader(test_dataset, 
                  batch_size=args.batch_size,
                  num_workers=args.num_workers)

#%%
backbonemodel = create_densenet_model(pretrained=True, drop_rate=args.drop_rate,num_classes=args.num_classes)
#%%
metrics = ["loss","acc", "F1", "auc"]
m = 3
#%%
task_predictor_path = os.path.join(task_predictor_dir,f"best_{metrics[m]}.pth")
check_point = torch.load(task_predictor_path)
backbonemodel.load_state_dict(check_point['weights'])
#%%
test_accuracy, test_f1_score, test_auc = evaluate_model_performance(backbonemodel,TestLoader,device)
print(f"\ntask predictor performance \n accuracy: {test_accuracy} \n F1 score: {test_f1_score} \n AUC: {test_auc}")
print(check_point['val_auc'])
#%%
teacher_model_path = os.path.join(teacher_model_dir,f"best_{metrics[m]}.pth")
check_point = torch.load(teacher_model_path)
backbonemodel.load_state_dict(check_point['weights'])
#%%
test_accuracy, test_f1_score, test_auc = evaluate_model_performance(backbonemodel,TestLoader,device)
print(f"\nteacher model performance \n accuracy: {test_accuracy} \n F1 score: {test_f1_score} \n AUC: {test_auc}")
print(check_point['val_auc'])
#%%
training_df = pd.read_pickle(df_path)

#%%
ups_df = training_df.copy()

ulb_dataset = unlabeled_dataset_from_path(train_df,ulb_indexes,transforms=test_transform)
UlbLoader = DataLoader(ulb_dataset, 
                  batch_size=args.batch_size,
                  num_workers=args.num_workers)

f_pass = 5
backbonemodel.train()
with torch.no_grad():
    for batch_idx, (inputs, indexs) in enumerate(UlbLoader):
        inputs = inputs.to(device)
        out_prob = []

        for _ in range(f_pass):
            outputs, _ = backbonemodel(inputs)
            out_prob.append(F.softmax(outputs, dim=1)) #for selecting positive pseudo-labels
        out_prob = torch.stack(out_prob)

        out_std = torch.std(out_prob, dim=0)
        out_prob = torch.mean(out_prob, dim=0)
        max_value, max_idx = torch.max(out_prob, dim=1)
        max_std = out_std.gather(1, max_idx.view(-1,1))


        confidence = max_value.cpu()
        uncertainty = max_std.squeeze(1).cpu().detach().numpy()
        ups_df.loc[indexs, 'uncertainty'] = uncertainty


#%%
energy_df = ups_df.copy()
ulb_dataset = unlabeled_dataset_from_path(train_df,ulb_indexes,transforms=train_transform)
UlbLoader = DataLoader(ulb_dataset, 
                  batch_size=args.batch_size,
                  num_workers=args.num_workers)

backbonemodel.eval()

with torch.no_grad():
    for batch_idx, (inputs, indexs) in enumerate(UlbLoader):
        inputs = inputs.to(device)
        outputs, features = backbonemodel(inputs)
        energy = -torch.logsumexp(outputs, dim=1).cpu().detach().numpy()
        # energy = -torch.logsumexp(features, dim=1).cpu().detach().numpy()

        # mask_raw = energy.le(-8)
        # break
        energy_df.loc[indexs, 'energy'] = energy


#%%
action_df = energy_df.copy()
ulb_df = action_df.loc[ulb_indexes].copy()
ulb_features_np = np.array(ulb_df['feature_map'].tolist())
pseudo_labels = ulb_df.loc[list(ulb_indexes),'labels']


agent_path = os.path.join(exp_dir,"agent.zip")
lb = np.array(list(pseudo_labels))
obs = np.concatenate([ulb_features_np,lb],axis=1)
agent = PPO.load(agent_path)
actions, _ = agent.predict(obs) 
actions = actions.squeeze(-1)
action_df.loc[ulb_indexes,'action'] = actions
#%%
class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
result_df = action_df.copy()
ulb_df = result_df.loc[ulb_indexes].copy()
lbl_df = result_df.loc[lbl_indexes].copy()
#%%
Cls_dist_visualizing(lbl_df,class_names,"small amount of labeled data")
Cls_dist_visualizing(ulb_df,class_names,"pseudo label")
#%% class distribution

Cls_dist_visualizing(result_df,class_names,"all")
    


#%%

lbl_labels_one_hot_np = np.array(lbl_df['labels'].tolist())
pseudo_labels_one_hot_np = np.array(ulb_df['labels'].tolist())

lbl_labels_np = np.argmax(list(lbl_labels_one_hot_np), axis=1)
pseudo_labels_np = np.argmax(list(pseudo_labels_one_hot_np), axis=1)


#%%
lbl_features_np = np.array(lbl_df['feature_map'].tolist())
ulb_features_np = np.array(ulb_df['feature_map'].tolist())
combined_features = np.concatenate([lbl_features_np, ulb_features_np], axis=0)
reducers = {"UMAP":umap.UMAP(n_components=2, random_state=42),"TSNE": TSNE(n_components=2, random_state=30)}

reducer_name = "TSNE"
reducer = reducers[reducer_name]

features_2d = reducer.fit_transform(combined_features)
lbl_features_2d = features_2d[:len(lbl_features_np)]
ulb_features_2d = features_2d[len(lbl_features_np):]
#%%
classes = np.unique(lbl_labels_np)


k_knn = 3
knn = KNeighborsClassifier(n_neighbors=k_knn)
knn.fit(lbl_features_2d, lbl_labels_np)


x_min, x_max = lbl_features_2d[:, 0].min() - 1, lbl_features_2d[:, 0].max() + 1
y_min, y_max = lbl_features_2d[:, 1].min() - 1, lbl_features_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))


Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
#%%

# Figure와 Axes 객체 생성
fig, ax = plt.subplots(figsize=(10, 6))
sns.set_style('whitegrid')

# 결정 경계 배경 표시 (Axes 객체 사용)
# contour = ax.contourf(xx, yy, Z, alpha=0.85, cmap=plt.cm.coolwarm)
contour = ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.turbo)
colors = plt.cm.turbo(np.linspace(0, 1, args.num_classes))

# 원본 데이터 포인트 시각화 (Axes 객체 사용)
for cls, color in zip(classes, colors):
    cls_features = lbl_features_2d[lbl_labels_np == cls]
    ax.scatter(cls_features[:, 0], cls_features[:, 1], label=f'{class_names[cls]}', alpha=1, color=color)

# 범례 설정 (Axes 객체 사용)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0), ncol=7, frameon=False)

# 축 숨기기
ax.axis('off')

# 그래프 제목 설정
ax.set_title(f'labeled data, num of data: {len(lbl_labels_np)}', fontsize=16)

# 그래프 출력
fig.tight_layout()
fig.show()



#%%



#%%
methods = ['nothing', 'confidence','uncertainty','energy','contribution','contributions','action']
threshold_scores = [None,0.9, 0.005, -7, 0.85, 0.5, 0.9]


#%%
l = 4
f = methods[l]
mask, threshold_score = mask_generate(ulb_df,methods[l],threshold_scores[l])

# Figure와 Axes 객체 생성
fig, ax = plt.subplots(figsize=(10, 6))
sns.set_style('whitegrid')

# 결정 경계 배경 표시 (Axes 객체 사용)
# contour = ax.contourf(xx, yy, Z, alpha=0.85, cmap=plt.cm.coolwarm)
contour = ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.turbo)
colors = plt.cm.turbo(np.linspace(0, 1, args.num_classes))

# 원본 데이터 포인트 시각화 (Axes 객체 사용)
for cls, color in zip(classes, colors):
    ulb_features = ulb_features_2d[mask][pseudo_labels_np[mask] == cls]
    ax.scatter(ulb_features[:, 0], ulb_features[:, 1], label=f'{class_names[cls]}', alpha=1, color=color)

# 범례 설정 (Axes 객체 사용)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0), ncol=7, frameon=False)

# 축 숨기기
ax.axis('off')

# 그래프 제목 설정
if threshold_score == None:
    ax.set_title(f'{f}, num of data: {len(mask)}', fontsize=16)
else:
    ax.set_title(f'{f}:{threshold_score},num of data: {sum(mask)}', fontsize=16)

# 그래프 출력
fig.tight_layout()
plt.show()





#%%









