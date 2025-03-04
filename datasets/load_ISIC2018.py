import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from skimage import io




def load_ISIC2018_GT(data_root):
    root = os.path.join(data_root,'ISIC2018')
    
    train_GT = "ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv"
    val_GT = "ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv"
    test_GT = "ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv"
    
    
    train_GT_path = os.path.join(root,train_GT)
    val_GT_path = os.path.join(root,val_GT)
    test_GT_path = os.path.join(root,test_GT)
    
    
    train_GT = pd.read_csv(train_GT_path)
    val_GT = pd.read_csv(val_GT_path)
    test_GT = pd.read_csv(test_GT_path)

    
    train_df = create_ISIC2018_dataframe(train_GT, data_root, splits ="Training")
    val_df = create_ISIC2018_dataframe(val_GT, data_root, splits ="Validation")
    test_df = create_ISIC2018_dataframe(test_GT, data_root, splits ="Test")


    
    return train_df, val_df, test_df



def create_ISIC2018_dataframe(df, data_root, splits ="Training" ):
    root = os.path.join(data_root,'ISIC2018')
    new_data = {
        'image_path': [],
        'GT': []
    }
    
    for index, row in df.iterrows():
        image_path = os.path.join(root, f"ISIC2018_Task3_{splits }_Input", row[0] + '.jpg')
        label = np.array(row[1:].map(int).tolist()) 
        
        new_data['image_path'].append(image_path)
        new_data['GT'].append(label)
    ISIC2018_dataframe = pd.DataFrame(new_data)
    ISIC2018_dataframe['labels'] = ISIC2018_dataframe['GT']
    return ISIC2018_dataframe
