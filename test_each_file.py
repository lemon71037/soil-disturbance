from numpy.lib.function_base import average
import torch
import random
import numpy as np
import pandas as pd
from dataset import SoilActDataset
from models import CNNClassifier, CNN2DClassifier
from data_process import generate_data
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score

def model_test(model, dataloader, n_class=3):
    model.eval()
    test_acc = [0.0] * n_class
    test_num  = len(dataloader)
    
    with torch.no_grad():
        for i, (sig, label) in enumerate(dataloader):
            sig, label = sig.float().to(device), label.long().to(device)
            pred = torch.argmax(model(sig), dim=1).cpu()

            acc = [(pred==i).sum().item() / pred.size(0) for i in range(n_class)]
            # acc = accuracy_score(torch.argmax(pred, dim=1).cpu(), label.cpu())
            test_acc = [(test_acc[i] + acc[i]) for i in range(n_class)]
        
        test_acc = [i / test_num for i in test_acc]
    
    return test_acc

snr = 5
batchSize = 64
act_class = 3
# soil_class = 5
data_mode = 'combine'
classifer = CNN2DClassifier if data_mode=='wavelet' else CNNClassifier
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Define TestDataset """
print("generating data...")
_, _, _, syf_dict = generate_data('E:/研一/嗑盐/土壤扰动/dataset/syf', factor=0, snr=snr)
_, _, _, syf2_dict = generate_data('E:/研一/嗑盐/土壤扰动/dataset/syf2', factor=0, snr=snr)
_, _, _, yqcc_dict = generate_data('E:/研一/嗑盐/土壤扰动/dataset/yqcc2', factor=0, snr=snr)
_, _, _, yqcc2_dict = generate_data('E:/研一/嗑盐/土壤扰动/dataset/yqcc2_md', factor=0, snr=snr)
_, _, _, zwy_dict = generate_data('E:/研一/嗑盐/土壤扰动/dataset/zwy', factor=0, snr=snr)
_, _, _, zwy2_dict = generate_data('E:/研一/嗑盐/土壤扰动/dataset/zwy_d1', factor=0, snr=snr)
_, _, _, zwy3_dict = generate_data('E:/研一/嗑盐/土壤扰动/dataset/zwy2', factor=0, by_txt=False, snr=snr)
_, _, _, zwy4_dict = generate_data('E:/研一/嗑盐/土壤扰动/dataset/zwy3', factor=0, by_txt=False, snr=snr)
_, _, _, j11_dict = generate_data('E:/研一/嗑盐/土壤扰动/dataset/j11', factor=0, by_txt=False, snr=snr)
_, _, _, j11_2_dict = generate_data('E:/研一/嗑盐/土壤扰动/dataset/j11_md', factor=0, by_txt=False, snr=snr)
_, _, _, j11_md_dict = generate_data('E:/研一/嗑盐/土壤扰动/dataset/j11_49', factor=0, by_txt=False, snr=snr)
_, _, _, zyq_dict = generate_data('E:/研一/嗑盐/土壤扰动/dataset/zyq', factor=0, by_txt=False, snr=snr)
_, _, _, zyq2_dict = generate_data('E:/研一/嗑盐/土壤扰动/dataset/zyq_d1', factor=0, by_txt=False, snr=snr)
_, _, _, j7lqc_dict = generate_data('E:/研一/嗑盐/土壤扰动/dataset/j7lqc', factor=0, by_txt=False, snr=snr)
_, _, _, sky_dict = generate_data('E:/研一/嗑盐/土壤扰动/dataset/sky', factor=0, by_txt=False, snr=snr)
print("generating data finishing...")

test_data = {'syf': syf_dict, 'syf2': syf2_dict, 'yqcc': yqcc_dict, 'yqcc2': yqcc2_dict, 
            'zwy': zwy_dict, 'zwy2': zwy2_dict, 'zwy3': zwy3_dict, 'zwy4': zwy4_dict, 'j11': j11_dict,
            'j11_2': j11_2_dict, 'j11_md': j11_md_dict, 'zyq': zyq_dict, 'zyq2': zyq2_dict, 
            'j7lqc': j7lqc_dict, 'sky': sky_dict}

act_model = classifer(n_class=act_class)
act_model.load_state_dict(torch.load('state_dicts/Act3ClassModel.pth'))
act_model = act_model.to(device)

""" Activity Test """
act_acc = []
for key, value in test_data.items():
    
    for name, data in value.items():

        dataset = SoilActDataset(data, mode=data_mode, use_soil=False)
        dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=False)
        item = {'area': key, 'name': name, 'num': len(dataset)}
        act_result = model_test(act_model, dataloader) if len(dataloader) != 0 else [0.0] * act_class
        
        for i in range(act_class):
            item['class'+str(i)] = act_result[i]

        act_acc.append(item)

df = pd.DataFrame(act_acc)
df.to_csv("each_file_result.csv")