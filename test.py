import torch
import random
import numpy as np
from dataset import SoilActDataset
from models import CNNClassifier
from data_process import generate_data
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

def model_test(model, dataloader):
    model.eval()
    test_acc = 0.0
    test_num = len(dataloader)
    
    with torch.no_grad():
        for i, (sig, label) in enumerate(dataloader):
            sig, label = sig.float().to(device), label.long().to(device)
            pred = model(sig)
            acc = accuracy_score(torch.argmax(pred, dim=1).cpu(), label.cpu())
            test_acc += acc.item()
        
        test_acc /= test_num
    
    return test_acc

batchSize = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Define TestDataset """
syf2, _, _ = generate_data('E:/研一/嗑盐/土壤扰动/dataset/syf2', factor=0)
yqcc2, _, _ = generate_data('E:/研一/嗑盐/土壤扰动/dataset/yqcc2_md', factor=0)
zwy2, _, _ = generate_data('E:/研一/嗑盐/土壤扰动/dataset/zwy_d1', factor=0)

test_data = {'syf': syf2, 'yqcc': yqcc2, 'zwy': zwy2}

""" Define Model """
soil_model = CNNClassifier()
soil_model.load_state_dict(torch.load('state_dicts/Area3ClassModel.pth'))
soil_model = soil_model.to(device)

act_model = CNNClassifier()
act_model.load_state_dict(torch.load('state_dicts/Act3ClassModel.pth'))
act_model = act_model.to(device)


""" Activity Test """
act_acc = {}
for key, value in test_data.items():
    
    dataset = SoilActDataset(value, mode='origin', use_soil=False)
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=False)
    act_acc[key] = model_test(act_model, dataloader)

""" Soil Test """
soil_acc = {}
for key, value in test_data.items():

    dataset = SoilActDataset(value, mode='origin', use_soil=True)
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=False)
    soil_acc[key] = model_test(soil_model, dataloader)

print(act_acc)
print(soil_acc)

