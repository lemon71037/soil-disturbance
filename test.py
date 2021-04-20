from numpy.lib.function_base import average
import torch
import random
import numpy as np
from dataset import SoilActDataset
from models import CNNClassifier, CNN2DClassifier
from data_process import generate_data
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score

def test_eachfile(model, data, filename, area, data_mode, n_class=3, use_soil=False):
    file_result = []
    for name in filename:
        file_dataset = []

        for item in data:
            if item['file_name'] == name:
                file_dataset.append(item)
        
        dataset = SoilActDataset(file_dataset, mode=data_mode, use_soil=use_soil)
        dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=False)
        
        test_acc, test_precision = model_test_detail(model, dataloader, n_class)
        file_result.apend({'area':area, 'name': name, 'acc': test_acc, 'precision': test_precision})
    
    return file_result

def model_test_detail(model, dataloader, n_class=3):
    model.eval()
    test_acc = 0.0
    test_presion = [0.0] * n_class
    test_num = len(dataloader)
    
    with torch.no_grad():
        for i, (sig, label) in enumerate(dataloader):
            sig, label = sig.float().to(device), label.long().to(device)

            pred = torch.argmax(model(sig), dim=1).cpu()
            acc = accuracy_score(label.cpu(), pred)
            precision = precision_score(label, pred, average=None)
            test_presion = [(test_presion[i] + precision[i]) for i in range(n_class)]
            test_acc += acc
        
        test_acc /= test_num
        test_presion /= test_num
    
    return test_acc, test_presion

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
# syf, _, _, _ = generate_data('E:/研一/嗑盐/土壤扰动/dataset/syf', factor=0)
# syf2, _, _, _ = generate_data('E:/研一/嗑盐/土壤扰动/dataset/syf2', factor=0)
_, _, _, yqcc_dict = generate_data('E:/研一/嗑盐/土壤扰动/dataset/yqcc2', factor=0)
_, _, _, yqcc2_dict = generate_data('E:/研一/嗑盐/土壤扰动/dataset/yqcc2_md', factor=0)
# zwy, _, _, _ = generate_data('E:/研一/嗑盐/土壤扰动/dataset/zwy', factor=0)
# zwy2, _, _, _ = generate_data('E:/研一/嗑盐/土壤扰动/dataset/zwy_d1', factor=0)
# j11, _, _, _ = generate_data('E:/研一/嗑盐/土壤扰动/dataset/j11', factor=0, by_txt=False)
# j11_2, _, _, _ = generate_data('E:/研一/嗑盐/土壤扰动/dataset/j11_md', factor=0, by_txt=False)
# j11_md, _, _, _ = generate_data('E:/研一/嗑盐/土壤扰动/dataset/j11_49', factor=0, by_txt=False)
_, _, _, zyq_dict = generate_data('E:/研一/嗑盐/土壤扰动/dataset/zyq', factor=0, by_txt=False)
_, _, _, zyq2_dict = generate_data('E:/研一/嗑盐/土壤扰动/dataset/zyq_d1', factor=0, by_txt=False)

# test_data = {'syf': syf2, 'yqcc': yqcc, 'zwy': zwy2, 'j11': j11, 'zyq': zyq2}

# test_data = {'syf': syf, 'syf2': syf2, 'yqcc': yqcc, 'yqcc2': yqcc2, 'zwy': zwy, 'zwy2': zwy2, 'j11': j11, \
    # 'j11_2': j11_2, 'j11_md': j11_md, 'zyq': zyq, 'zyq2': zyq2}
# test_data = {'j11': j11, 'j11_2': j11_2, 'zyq': zyq, 'zyq2': zyq2}
test_data = yqcc2_dict

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

