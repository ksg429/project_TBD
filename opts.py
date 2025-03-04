import argparse
from datetime import datetime
from pytz import timezone

datetime.now(timezone('Asia/Seoul'))
def get_opts():
    # general setting
    parser = argparse.ArgumentParser(description="MPS")
    
    path_group = parser.add_argument_group("About path")
    path_group.add_argument("--data_root", metavar="DIR", default="/TBD/all_data", help="path to dataset")
    path_group.add_argument("--save_dir", default="../save_files", help="path to save files")
    path_group.add_argument("--experiment_num", type=int, default=datetime.now(timezone('Asia/Seoul')).strftime("%Y%m%d_%H%M%S"), help="experiments number")

    
    data_group = parser.add_argument_group("About data")
    data_group.add_argument("--data", type=str, choices=["ISIC2018"],  
                        default="ISIC2018",help="dataset to use.")
    data_group.add_argument("--seed", type=int, 
                        default=123, help="random seed")
    data_group.add_argument("--lbl_ratio", type=float, 
                        default=0.1, help="labeled data ratio")
    data_group.add_argument("--resize", type=int, 
                        default=224, help="image resize dimension")
    data_group.add_argument("--batch_size", type=int, 
                        default=32, help="batch size")
    data_group.add_argument("--num_classes", type=int, 
                        default=7, help="number of classes for data")
    data_group.add_argument("--num_workers", type=int, 
                        default=4, help="number of workers for data loading")
    

    model_group = parser.add_argument_group("About model")
    model_group.add_argument("--arch", type=str, choices=["DenseNet"],  
                        default="DenseNet",help="architecture to use.")
    model_group.add_argument("--drop_rate", type=float, 
                        default=0.5, help="dropout rate")


    training_group = parser.add_argument_group("About SL training")
    training_group.add_argument("--gpu_num", type=int, 
                        default=0, help="GPU number to use")
    training_group.add_argument("--lr", type=float, 
                        default=1e-3, help="learning rate")
    training_group.add_argument("--num_epochs", type=int, 
                        default=128, help="number of epochs")

    
    reinforcement_learning_group = parser.add_argument_group("About data contribution evaluating")
    reinforcement_learning_group.add_argument("--RL", type=str, choices=["PPO", "TD3", "SAC", "DDPG"],  
                        default="PPO",help="Reinforcement learning algorithm to use.")
    reinforcement_learning_group.add_argument("--RL_lr", type=float, 
                        default=1e-4, help="learning rate")
    reinforcement_learning_group.add_argument("--episode_size", type=int, 
                        default=16, help="episode size")
    reinforcement_learning_group.add_argument("--mini_batch_size", type=int, 
                        default=4, help="mini batch size")
    reinforcement_learning_group.add_argument("--mini_num_epochs", type=int, 
                        default=4, help="number of mini-batch epochs")
    reinforcement_learning_group.add_argument("--controller_gpu", type=int, 
                        default=2, help="controller GPU number")


    args = parser.parse_args()
    return args