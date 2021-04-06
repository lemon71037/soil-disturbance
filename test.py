import torch
import random
import numpy as np
from dataset import SoilActDataset
from models import CNNClassifier, CNN2DClassifier
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
syf, _, _ = generate_data('E:/研一/嗑盐/土壤扰动/dataset/syf', factor=0)
syf2, _, _ = generate_data('E:/研一/嗑盐/土壤扰动/dataset/syf2', factor=0)
yqcc, _, _ = generate_data('E:/研一/嗑盐/土壤扰动/dataset/yqcc2', factor=0)
yqcc2, _, _ = generate_data('E:/研一/嗑盐/土壤扰动/dataset/yqcc2_md', factor=0)
zwy, _, _ = generate_data('E:/研一/嗑盐/土壤扰动/dataset/zwy', factor=0)
zwy2, _, _ = generate_data('E:/研一/嗑盐/土壤扰动/dataset/zwy_d1', factor=0)
j11, _, _ = generate_data('E:/研一/嗑盐/土壤扰动/dataset/j11', factor=0, by_txt=False)
j11_2, _, _ = generate_data('E:/研一/嗑盐/土壤扰动/dataset/j11_md', factor=0, by_txt=False)
zyq, _, _ = generate_data('E:/研一/嗑盐/土壤扰动/dataset/zyq', factor=0, by_txt=False)
zyq2, _, _ = generate_data('E:/研一/嗑盐/土壤扰动/dataset/zyq_d1', factor=0, by_txt=False)

# test_data = {'syf': syf2, 'yqcc': yqcc, 'zwy': zwy2, 'j11': j11, 'zyq': zyq2}

test_data = {'syf': syf}
# test_data = {'j11': j11, 'j11_2': j11_2, 'zyq': zyq, 'zyq2': zyq2}
act_class = 3
soil_class = 5
data_mode = 'combine'
classifer = CNN2DClassifier if data_mode=='wavelet' else CNNClassifier

""" Define Model """
soil_model = classifer(n_class=soil_class)
soil_model.load_state_dict(torch.load('state_dicts/Soil5ClassModel.pth'))
soil_model = soil_model.to(device)

act_model = classifer(n_class=act_class)
act_model.load_state_dict(torch.load('state_dicts/Act3ClassModel.pth'))
act_model = act_model.to(device)


""" Activity Test """
act_acc = {}
for key, value in test_data.items():
    
    dataset = SoilActDataset(value, mode=data_mode, use_soil=False)
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=False)
    act_acc[key] = model_test(act_model, dataloader)

""" Soil Test """
soil_acc = {}
for key, value in test_data.items():
    dataset = SoilActDataset(value, mode=data_mode, use_soil=True)
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=False)
    soil_acc[key] = model_test(soil_model, dataloader)

print(act_acc)
print(soil_acc)

