import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from tqdm import tqdm

#%%

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
    test_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')

    
    

    return test_accuracy, test_f1_score, test_auc

