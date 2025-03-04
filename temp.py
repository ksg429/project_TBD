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

experiment_num = args.experiment_num

#%%

experiment_num = 4
gpu_num = 3

#%% experiments


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

#%%
len(ulb_indexes)*(num_epochs//mini_num_epochs)

#%% data pre-processing

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

test_transform = transforms.Compose(
    [
        transforms.Resize((resize,resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)


test_dataset = labeled_dataset_from_path(test_df, test_indexes, transforms=test_transform)


TestLoader = DataLoader(test_dataset, 
                  batch_size=batch_size,
                  num_workers=num_workers)


#%% pre-trained teacher model test

teacher_model = models.vgg19(weights=False)
dim_feature = teacher_model.classifier[-1].in_features
teacher_model.classifier[-1] = nn.Linear(in_features=dim_feature, out_features=num_classes)



#%% test teacher model

teacher_model_dir = os.path.join(exp_dir,"teacher_model")
metrics = ["loss","acc", "F1", "auc"]
m = 3

teacher_model_path = os.path.join(teacher_model_dir,f"best_{metrics[m]}.pth")
check_point = torch.load(teacher_model_path)

teacher_model.load_state_dict(check_point['weights'])

test_accuracy, test_f1_score, test_auc = evaluate_model_performance(teacher_model,TestLoader,device)

print(f"\nteacher_model_performance_best_{metrics[m]}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test F1 Score: {test_f1_score:.4f}")
print(f"Test AUC: {test_auc:.4f}")


#%%


task_predictor_dir = os.path.join(exp_dir,"task_predictor")

metrics = ["loss","acc", "F1", "auc"]
m=m

task_predictor_model_path = os.path.join(task_predictor_dir,f"best_{metrics[m]}.pth")

check_point = torch.load(task_predictor_model_path)

teacher_model.load_state_dict(check_point['weights'])


test_accuracy, test_f1_score, test_auc = evaluate_model_performance(teacher_model,TestLoader,device)

print(f"\ntask_predictor_performance_best_{metrics[m]}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test F1 Score: {test_f1_score:.4f}")
print(f"Test AUC: {test_auc:.4f}")



#%%
df_save = os.path.join(exp_dir,"train_df.pkl")
train_df = pd.read_pickle(df_save)


#%%
# train_df = pd.read_pickle(df_save_path)
#%%
training_df = train_df.copy()
ulb_df = training_df.loc[ulb_indexes].copy()
lbl_df = training_df.loc[lbl_indexes].copy()



#%%
from sklearn.manifold import TSNE
import umap


# print(len(training_df[training_df['confidence'] > 0.99]), len(training_df[training_df['contribution'] > 0.5]))

lbl_features_np = np.array(lbl_df['feature_map'].tolist())
ulb_features_np = np.array(ulb_df['feature_map'].tolist())
combined_features = np.concatenate([lbl_features_np, ulb_features_np], axis=0)



reducers = {"UMAP":umap.UMAP(n_components=2, random_state=42),"TSNE": TSNE(n_components=2, random_state=42)}

reducer_name = "TSNE"
reducer = reducers[reducer_name]

features_2d = reducer.fit_transform(combined_features)
#%%
import matplotlib.pyplot as plt


lbl_features_2d = features_2d[:len(lbl_features_np)]
ulb_features_2d = features_2d[len(lbl_features_np):]


compare = ['confidence','contribution','contributions']

i = 0

threshold_score = .9999999

if compare[i] == 'confidence':
    ulb_score= np.array(ulb_df['confidence'].tolist())
elif compare[i] == 'contribution':
    ulb_score= np.array(ulb_df['contribution'].tolist())
elif compare[i] == 'contributions':
    ulb_score= np.array(ulb_df['contributions'].apply(lambda x : np.mean(np.array(x[:-5]),axis=0)).tolist())

#%% Plot the 2D features
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.cm as cm

plt.figure(figsize=(10, 6))

labels = lbl_df.loc[list(lbl_indexes),'labels']
pseudo_labels = ulb_df.loc[list(ulb_indexes),'labels']

lbl_labels_one_hot_np = labels 
pseudo_labels_one_hot_np = pseudo_labels 

lbl_labels_np = np.argmax(list(lbl_labels_one_hot_np), axis=1)
pseudo_labels_np = np.argmax(list(pseudo_labels_one_hot_np), axis=1)

classes = np.unique(lbl_labels_np)
colors = plt.cm.rainbow(np.linspace(0, 1, len(classes)))

k_knn = 3
knn = KNeighborsClassifier(n_neighbors=k_knn)
knn.fit(lbl_features_2d, lbl_labels_np)


x_min, x_max = lbl_features_2d[:, 0].min() - 1, lbl_features_2d[:, 0].max() + 1
y_min, y_max = lbl_features_2d[:, 1].min() - 1, lbl_features_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))


Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


plt.contourf(xx, yy, Z, alpha=0.85, cmap=plt.cm.coolwarm)
# plt.colorbar(label=classes)
for cls, color in zip(classes, colors):
    # cls_features = lbl_features_2d[lbl_labels_np == cls]
    # plt.scatter(cls_features[:, 0], cls_features[:, 1], label=f'Class {cls}', alpha=0.75, color=color)
    
    ulb_features = ulb_features_2d[(pseudo_labels_np == cls) & (ulb_score >= threshold_score)]
    plt.scatter(ulb_features[:, 0], ulb_features[:, 1],label=f'Class {cls}', alpha=0.75, color=color)
    
ulb_features = ulb_features_2d[(ulb_score >= threshold_score)]
plt.scatter(ulb_features[:, 0], ulb_features[:, 1], cmap = cm.viridis, alpha=0.7)

# plt.title('unlabeled data featuremap')
plt.title(f'{compare[i]}:{threshold_score}')
plt.axis('off')
plt.show()



#%%
filted_df = ulb_df[(ulb_score >= threshold_score)]
print(f"number of data {compare[i]}:{len(filted_df)}")

y_true = filted_df['GT'].apply(lambda x: np.argmax(x)) 
y_pred = filted_df['labels'].apply(lambda x: np.argmax(x))  


accuracy = accuracy_score(y_true, y_pred)


f1 = f1_score(y_true, y_pred, average='macro')  

print(f"{compare[i]}:{threshold_score}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
#%%

y_true = ulb_df['GT'].apply(lambda x: np.argmax(x))  
y_pred = ulb_df['labels'].apply(lambda x: np.argmax(x))  


accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='macro')  


print("total")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

