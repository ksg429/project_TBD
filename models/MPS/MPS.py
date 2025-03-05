
import torch.nn as nn
import os
import torch
import torch.nn as nn
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
from gymnasium import spaces
import copy
import gymnasium as gym
from torch import nn
import numpy as np
from typing import Dict, Any  
import gymnasium
from stable_baselines3.common.env_util import make_vec_env


from models.nets import densenet






from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class MultiFE(BaseFeaturesExtractor):
    def __init__(
            self, 
            observation_space: gym.spaces.Dict, 
            features_dim: int = 1024, 
            nb_classes: int = 7,
            net_size: Dict[str, Any] = [64, 32, 32], 
            in_stride: int = 1, 
            in_padding: int = 0
        ):
        super().__init__(observation_space,features_dim+nb_classes)
        # We assume CxHxW images (channels first)
        n_input_channels, H_in, W_in = observation_space["image"].shape
        layers = []

        C_in, C_out = n_input_channels, net_size[0]
        H_out, W_out = self.conv_output_shape((H_in, W_in), kernel_size=3)
        layer_0 = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU()
        )
        layers.append(layer_0)

        for layer in net_size[1:]:
            C_in, C_out = C_out, layer
            H_out, W_out = self.conv_output_shape((H_out, W_out), kernel_size=3)
            layer_ = nn.Sequential(
                nn.Conv2d(C_in, C_out, kernel_size=3, stride=1, padding=0),
                nn.LeakyReLU()
            )
            layers.append(layer_)
        
        self.cnn = nn.Sequential(*layers,nn.Flatten())

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space["image"].sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.LeakyReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        images_enc = self.linear(self.cnn(observations["image"]))
        return torch.cat([images_enc, observations["label"]], dim=1)
    
    def conv_output_shape(self, h_w, kernel_size=1, stride=1, pad=0, dilation=1):
        from math import floor
        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size, kernel_size)
        h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
        w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
        return h, w














#%%
# df_save = os.path.join(exp_dir,"train_df.pkl")
# RL_save = os.path.join(exp_dir,"agent")
# RL_log_dir = os.path.join(exp_dir,"RL_log")

# env = PL_data_valuation_env(PL_train_df, ulb_indexes, ValLoader, predictor)
# my_env = DummyVecEnv([lambda: env])

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
#           learning_rate=1e-5,
#           policy="MultiInputPolicy",
#           policy_kwargs= policy_kwargs,
#           tensorboard_log=RL_log_dir,
#           device=f"cuda:{controller_gpu}",
#           verbose=2
#         )

# # agent = PPO("MlpPolicy", my_env, tensorboard_log=RL_log_dir, verbose=2, device=f"cuda:{controller_gpu}",learning_rate=0.0001)

# RL_steps = len(ulb_indexes)*(num_epochs//mini_num_epochs)
# agent.learn(total_timesteps=RL_steps) 
# agent.save(RL_save)
# env.train_df.to_pickle(df_save)

# predictor_weights = copy.deepcopy(env.predictor.state_dict())
# task_predictor_path = os.path.join(exp_dir,"task_predictor.pth")
# torch.save(predictor_weights, task_predictor_path)
