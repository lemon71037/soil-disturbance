from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle
import numpy as np
import torch
import random
from data_process import handle_data_3dims, generate_data

class SoilActDataset(Dataset):
    """用于振动信号所属事件识别 以及 振动所在的土质/场地识别
    """
    def __init__(self, data, mode='origin', use_soil = False):
        super(SoilActDataset, self).__init__()
        self.data = data
        self.mode = mode
        self.use_soil = use_soil
        
        if mode not in {'origin', 'combine'}:
            raise ValueError("Unrecognized mode: {}".format(mode))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index] # {'data_x': .., 'data_y': .., 'data_z': .., ...}
        
        data = np.array([item['data_x'], item['data_y'], item['data_z']]) if self.mode=='origin' \
            else handle_data_3dims(item)
        
        label = item['soil'] if self.use_soil else item['label'] # use_soil表示是否要识别土壤类别
        
        return data, label

class Soil2ClassSet(Dataset):
    """用于识别两个信号是否属于同一土壤的数据集(2分类)
    """
    def __init__(self, data, mode='origin'):
        super(Soil2ClassSet, self).__init__()
        sep = len(data) // 2
        self.group1 = data[:sep]
        self.group2 = data[sep:]
        self.mode = mode
        
        if mode not in {'origin', 'combine'}:
            raise ValueError("Unrecognized mode: {}".format(mode))
    
    def __len__(self):
        return min(len(self.group1), len(self.group2))
    
    def __getitem__(self, index):
        item1, item2 = self.group1[index], self.group2[index] # {'data_x': .., 'data_y': .., 'data_z': .., ...}
        
        data1, data2 = handle_data_3dims(item1, self.mode), handle_data_3dims(item2, self.mode)
        
        label = 1 if item1['label'] == item2['label'] else 0
        return data1, data2, label

if __name__ == '__main__':
    syf_data, _, _ = generate_data('E:/研一/嗑盐/土壤扰动/dataset/syf')
    yqcc_data, _, _ = generate_data('E:/研一/嗑盐/土壤扰动/dataset/yqcc2')

    train_data = syf_data + yqcc_data
    random.shuffle(train_data)
    
    # trainset = SoilActDataset(train_data, mode='origin')
    trainset = Soil2ClassSet(train_data, mode='origin')
    trainloader = DataLoader(trainset, batch_size=2, shuffle=True)
    for d1, d2, l in trainloader:
        print(d1, d2)
        print(l)
        break
