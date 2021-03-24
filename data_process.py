import numpy as np
import os
from data_split import *
import pandas as pd
import matplotlib.pyplot as plt
import random

def cut_mean(data_list):
    return [(data - np.mean(data_list)) for data in data_list]

def cut_value(data_list, value):
    return [(data - value) for data in data_list]

def min_max_normalize(data_list):
    return [(data - np.min(data_list)) / (np.max(data_list) - np.min(data_list)) \
        for data in data_list]

def fill_data(data):
    if len(data) < 96:
        fill = np.mean(data)
        data = data + [fill] * (96-len(data))
    return data[:96]

def fill_data_with_base(data, base):
    # 用基线填充
    if len(data) < 96:
        data = data + [base] * (96-len(data))
    return data[:96]

def extract_data_from_center(data, center, base):
    # 从标签点向两边扩充，截取96长度的信号
    return fill_data_with_base(data[center-48: center+48], base)

def get_label(file_name):
    if 'dig' in file_name:
        return 0
    elif 'jump' in file_name:
        return 1
    else:
        return 2

def plot_time(signal, sample_rate):
    time = np.arange(0, len(signal)) * (1.0 / sample_rate)
    plt.figure(figsize=(20, 5))
    plt.plot(time, signal)
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')
    plt.grid()

def cal_angles(base_value):
    '''
    计算三轴与g方向的夹角
    '''
    return {'x': np.arctan(np.sqrt(base_value[1]**2 + base_value[2]**2) / base_value[0]),
            'y': np.arctan(np.sqrt(base_value[0]**2 + base_value[2]**2) / base_value[1]),
            'z': np.arctan(np.sqrt(base_value[0]**2 + base_value[1]**2) / base_value[2])}

def create_dataset_1dim(input_tensor):
    # input: n_samples * {'data_x':, 'data_y':, 'data_z':, 'label':, 'file_name':}
    dataset = []
    labels = []
    for item in input_tensor:
        data_x = np.array(cut_mean(item['data_x']), dtype=np.float64)
        data_y = np.array(cut_mean(item['data_y']), dtype=np.float64)
        data_z = np.array(cut_mean(item['data_z']), dtype=np.float64)
        
        data = np.sqrt([data_x**2 + data_y**2 + data_z**2])
        # data = np.swapaxes(data, 0, 1)
        dataset.append(data)
        labels.append(item['label'])
    dataset = np.array(dataset)
    labels = np.array(labels)
    
    return dataset, labels

def create_dataset_1dim_cut_base(input_tensor):
    # input: n_samples * {'data_x':, 'data_y':, 'data_z':, 'label':, 'file_name':, 'base_value':}
    dataset = []
    labels = []
    for item in input_tensor:
        data_x = np.array(cut_value(item['data_x'], item['base_value'][0]), dtype=np.float64)
        data_y = np.array(cut_value(item['data_y'], item['base_value'][1]), dtype=np.float64)
        data_z = np.array(cut_value(item['data_z'], item['base_value'][2]), dtype=np.float64)
        
        if len(data_x) != 96 or len(data_y) != 96 or len(data_z) != 96:
            print("yaosi!")
        data = [data_x**2 + data_y**2 + data_z**2]
        dataset.append(data)
        labels.append(item['label'])
    dataset = np.array(dataset)
    labels = np.array(labels)
    
    return dataset, labels

def create_dataset_2dims_cut_base(input_tensor, mode='former'):
    # input: n_samples * {'data_x':, 'data_y':, 'data_z':, 'label':, 'file_name':, 'base_value':, 'angle':}
    # mode: 'former' or 'after', the time for cut base
    dataset = []
    labels = []
    for item in input_tensor:
        if mode == 'former':
            data_x = np.array(cut_value(item['data_x'], item['base_value'][0]), dtype=np.float64)
            data_y = np.array(cut_value(item['data_y'], item['base_value'][1]), dtype=np.float64)
            data_z = np.array(cut_value(item['data_z'], item['base_value'][2]), dtype=np.float64)
            
            data = np.array([np.sqrt(data_x**2 + data_y**2), data_z], dtype=np.float64)
        else:
            data_x = np.array(item['data_x'], dtype=np.float64)
            data_y = np.array(item['data_y'], dtype=np.float64)
            data_z = np.array(item['data_z'], dtype=np.float64)

            data_xy = np.sqrt(data_x**2 + data_y**2)
            data = np.array([data_xy - np.sqrt(item['base_value'][0]**2 + item['base_value'][1]**2), data_z])

        dataset.append(data)
        labels.append(item['label'])
    dataset = np.array(dataset)
    labels = np.array(labels)
    
    return dataset, labels

def create_dataset_2dims_cut_base_rectify(input_tensor):
    # input: n_samples * {'data_x':, 'data_y':, 'data_z':, 'label':, 'file_name':, 'base_value':, 'angle':}
    # mode: 'former' or 'after', the time for cut base
    dataset = []
    labels = []
    for item in input_tensor:
        data_x = np.array(item['data_x'], dtype=np.float64)
        data_y = np.array(item['data_y'], dtype=np.float64)
        data_z = np.array(item['data_z'], dtype=np.float64)

        data_xy = np.sqrt(data_x**2 + data_y**2)

        sinA, cosA = np.sin(item['angle']), np.cos(item['angle'])
        data = np.array([data_xy*cosA + data_z*sinA, -data_xy*sinA + data_z*cosA])

        dataset.append(data)
        labels.append(item['label'])
    dataset = np.array(dataset)
    labels = np.array(labels)
    
    return dataset, labels

def create_dataset_3dims_cut_base_abs(input_tensor):
    # input: n_samples * {'data_x':, 'data_y':, 'data_z':, 'label':, 'file_name':, 'base_value':}
    dataset = []
    labels = []
    for item in input_tensor:
        data_x = np.abs(np.array(cut_value(item['data_x'], item['base_value'][0]), dtype=np.float64))
        data_y = np.abs(np.array(cut_value(item['data_y'], item['base_value'][1]), dtype=np.float64))
        data_z = np.abs(np.array(cut_value(item['data_z'], item['base_value'][2]), dtype=np.float64))
        
        data = np.array([data_x, data_y, data_z])
        dataset.append(data)
        labels.append(item['label'])
    dataset = np.array(dataset)
    labels = np.array(labels)
    
    return dataset, labels

def create_dataset_4dims_cut_base(input_tensor):
    # input: n_samples * {'data_x':, 'data_y':, 'data_z':, 'label':, 'file_name':, 'base_value':}
    dataset = []
    labels = []
    for item in input_tensor:
        data_x = np.array(cut_value(item['data_x'], item['base_value'][0]), dtype=np.float64)
        data_y = np.array(cut_value(item['data_y'], item['base_value'][1]), dtype=np.float64)
        data_z = np.array(cut_value(item['data_z'], item['base_value'][2]), dtype=np.float64)
        
        data = np.array([data_x, data_y, data_z, data_x**2 + data_y**2 + data_z**2])
        dataset.append(data)
        labels.append(item['label'])
    dataset = np.array(dataset)
    labels = np.array(labels)
    
    return dataset, labels

def create_dataset_5dims_cut_base(input_tensor):
    # input: n_samples * {'data_x':, 'data_y':, 'data_z':, 'label':, 'file_name':, 'base_value':}
    dataset = []
    labels = []
    for item in input_tensor:
        data_x = np.array(cut_value(item['data_x'], item['base_value'][0]), dtype=np.float64)
        data_y = np.array(cut_value(item['data_y'], item['base_value'][1]), dtype=np.float64)
        data_z = np.array(cut_value(item['data_z'], item['base_value'][2]), dtype=np.float64)
        
        data = np.array([data_x, data_y, data_z, np.sqrt(data_x**2 + data_y**2), data_x**2 + data_y**2 + data_z**2])
        dataset.append(data)
        labels.append(item['label'])
    dataset = np.array(dataset)
    labels = np.array(labels)
    
    return dataset, labels

def create_dataset_1dim_normalize(input_tensor):
    # input: n_samples * {'data_x':, 'data_y':, 'data_z':, 'label':, 'file_name':}
    dataset = []
    labels = []
    for item in input_tensor:
        data_x = np.array(cut_mean(min_max_normalize(item['data_x'])), dtype=np.float64)
        data_y = np.array(cut_mean(min_max_normalize(item['data_y'])), dtype=np.float64)
        data_z = np.array(cut_mean(min_max_normalize(item['data_z'])), dtype=np.float64)
        
        data = np.sqrt([data_x**2 + data_y**2 + data_z**2])
        # data = np.swapaxes(data, 0, 1)
        dataset.append(data)
        labels.append(item['label'])
    dataset = np.array(dataset)
    labels = np.array(labels)

    return dataset, labels

def create_dataset_1dim_by_filename(file_name, input_tensor):
    dataset = []
    labels = []
    for item in input_tensor:
        if item['file_name'] == file_name:
            data_x = np.array(cut_mean(item['data_x']), dtype=np.float64)
            data_y = np.array(cut_mean(item['data_y']), dtype=np.float64)
            data_z = np.array(cut_mean(item['data_z']), dtype=np.float64)
        
            data = np.sqrt([data_x**2 + data_y**2 + data_z**2])
            # data = np.swapaxes(data, 0, 1)
            dataset.append(data)
            labels.append(item['label'])
    
    dataset = np.array(dataset)
    labels = np.array(labels)
    
    return dataset, labels

def create_dataset_1dim_cut_base_by_filename(file_name, input_tensor):
    dataset = []
    labels = []
    for item in input_tensor:
        if item['file_name'] == file_name:
            data_x = np.array(cut_value(item['data_x'], item['base_value'][0]), dtype=np.float64)
            data_y = np.array(cut_value(item['data_y'], item['base_value'][1]), dtype=np.float64)
            data_z = np.array(cut_value(item['data_z'], item['base_value'][2]), dtype=np.float64)
            
            data = [data_x**2 + data_y**2 + data_z**2]
            dataset.append(data)
            labels.append(item['label'])
    
    dataset = np.array(dataset)
    labels = np.array(labels)
    
    return dataset, labels

def create_dataset_2dims_cut_base_by_filename(file_name, input_tensor, mode='former'):
    dataset = []
    labels = []
    for item in input_tensor:
        if item['file_name'] == file_name:
            if mode == 'former':
                data_x = np.array(cut_value(item['data_x'], item['base_value'][0]), dtype=np.float64)
                data_y = np.array(cut_value(item['data_y'], item['base_value'][1]), dtype=np.float64)
                data_z = np.array(cut_value(item['data_z'], item['base_value'][2]), dtype=np.float64)
                
                data = np.array([np.sqrt(data_x**2 + data_y**2), data_z], dtype=np.float64)
            else:
                data_x = np.array(item['data_x'], dtype=np.float64)
                data_y = np.array(item['data_y'], dtype=np.float64)
                data_z = np.array(item['data_z'], dtype=np.float64)

                data_xy = np.sqrt(data_x**2 + data_y**2)
                data = np.array([data_xy - np.sqrt(item['base_value'][0]**2 + item['base_value'][1]**2), data_z])
            
            dataset.append(data)
            labels.append(item['label'])
    
    dataset = np.array(dataset)
    labels = np.array(labels)
    
    return dataset, labels

def create_dataset_2dims_cut_base_by_filename_rectify(file_name, input_tensor):
    dataset = []
    labels = []
    for item in input_tensor:
        if item['file_name'] == file_name:
            data_x = np.array(item['data_x'], dtype=np.float64)
            data_y = np.array(item['data_y'], dtype=np.float64)
            data_z = np.array(item['data_z'], dtype=np.float64)

            data_xy = np.sqrt(data_x**2 + data_y**2)
            
            sinA, cosA = np.sin(item['angle']), np.cos(item['angle'])
            data = np.array([data_xy*cosA + data_z*sinA, -data_xy*sinA + data_z*cosA])
            
            dataset.append(data)
            labels.append(item['label'])
    
    dataset = np.array(dataset)
    labels = np.array(labels)
    
    return dataset, labels

def create_dataset_3dims_cut_base_by_filename_abs(file_name, input_tensor):
    dataset = []
    labels = []
    for item in input_tensor:
        if item['file_name'] == file_name:
            data_x = np.abs(np.array(cut_value(item['data_x'], item['base_value'][0]), dtype=np.float64))
            data_y = np.abs(np.array(cut_value(item['data_y'], item['base_value'][1]), dtype=np.float64))
            data_z = np.abs(np.array(cut_value(item['data_z'], item['base_value'][2]), dtype=np.float64))
            
            data = np.array([data_x, data_y, data_z])
            dataset.append(data)
            labels.append(item['label'])
    
    dataset = np.array(dataset)
    labels = np.array(labels)
    
    return dataset, labels

def create_dataset_4dims_cut_base_by_filename(file_name, input_tensor):
    dataset = []
    labels = []
    for item in input_tensor:
        if item['file_name'] == file_name:
            data_x = np.array(cut_value(item['data_x'], item['base_value'][0]), dtype=np.float64)
            data_y = np.array(cut_value(item['data_y'], item['base_value'][1]), dtype=np.float64)
            data_z = np.array(cut_value(item['data_z'], item['base_value'][2]), dtype=np.float64)
            
            data = np.array([data_x, data_y, data_z, data_x**2 + data_y**2 + data_z**2])
            dataset.append(data)
            labels.append(item['label'])
    
    dataset = np.array(dataset)
    labels = np.array(labels)
    
    return dataset, labels

def create_dataset_5dims_cut_base_by_filename(file_name, input_tensor):
    dataset = []
    labels = []
    for item in input_tensor:
        if item['file_name'] == file_name:
            data_x = np.array(cut_value(item['data_x'], item['base_value'][0]), dtype=np.float64)
            data_y = np.array(cut_value(item['data_y'], item['base_value'][1]), dtype=np.float64)
            data_z = np.array(cut_value(item['data_z'], item['base_value'][2]), dtype=np.float64)
            
            data = np.array([data_x, data_y, data_z, np.sqrt(data_x**2 + data_y**2), data_x**2 + data_y**2 + data_z**2])
            dataset.append(data)
            labels.append(item['label'])
    
    dataset = np.array(dataset)
    labels = np.array(labels)
    
    return dataset, labels

def create_dataset_1dim_normalize_by_filename(file_name, input_tensor):
    dataset = []
    labels = []
    for item in input_tensor:
        if item['file_name'] == file_name:
            data_x = np.array(cut_mean(min_max_normalize(item['data_x'])), dtype=np.float64)
            data_y = np.array(cut_mean(min_max_normalize(item['data_y'])), dtype=np.float64)
            data_z = np.array(cut_mean(min_max_normalize(item['data_z'])), dtype=np.float64)
        
            data = np.sqrt([data_x**2 + data_y**2 + data_z**2])
            # data = np.swapaxes(data, 0, 1)
            dataset.append(data)
            labels.append(item['label'])
    
    dataset = np.array(dataset)
    labels = np.array(labels)
    
    return dataset, labels

def create_dataset_2dims(input_tensor):
    # input: n_samples * {'data_x':, 'data_y':, 'data_z':, 'label':, 'file_name':}
    dataset = []
    labels = []
    for item in input_tensor:
        data_x = np.array(cut_mean(item['data_x']), dtype=np.float64)
        data_y = np.array(cut_mean(item['data_y']), dtype=np.float64)
        data_z = np.array(cut_mean(item['data_z']), dtype=np.float64)
        
        data = np.array([np.sqrt(data_x**2 + data_y**2), data_z], dtype=np.float64)
        # data = np.swapaxes(data, 0, 1)
        dataset.append(data)
        labels.append(item['label'])
    dataset = np.array(dataset)
    labels = np.array(labels)
    
    return dataset, labels

def create_dataset_2dims_normalize(input_tensor):
    # input: n_samples * {'data_x':, 'data_y':, 'data_z':, 'label':, 'file_name':}
    dataset = []
    labels = []
    for item in input_tensor:
        data_x = np.array(cut_mean(min_max_normalize(item['data_x'])), dtype=np.float64)
        data_y = np.array(cut_mean(min_max_normalize(item['data_y'])), dtype=np.float64)
        data_z = np.array(cut_mean(min_max_normalize(item['data_z'])), dtype=np.float64)
        
        data = np.array([np.sqrt(data_x**2 + data_y**2), data_z], dtype=np.float64)
        # data = np.swapaxes(data, 0, 1)
        dataset.append(data)
        labels.append(item['label'])
    dataset = np.array(dataset)
    labels = np.array(labels)

    return dataset, labels

def create_dataset_2dims_by_filename(file_name, input_tensor):
    dataset = []
    labels = []
    for item in input_tensor:
        if item['file_name'] == file_name:
            data_x = np.array(cut_mean(item['data_x']), dtype=np.float64)
            data_y = np.array(cut_mean(item['data_y']), dtype=np.float64)
            data_z = np.array(cut_mean(item['data_z']), dtype=np.float64)
        
            data = np.array([np.sqrt(data_x**2 + data_y**2), data_z], dtype=np.float64)
            # data = np.swapaxes(data, 0, 1)
            dataset.append(data)
            labels.append(item['label'])
    
    dataset = np.array(dataset)
    labels = np.array(labels)
    
    return dataset, labels

def create_dataset_2dims_normalize_by_filename(file_name, input_tensor):
    dataset = []
    labels = []
    for item in input_tensor:
        if item['file_name'] == file_name:
            data_x = np.array(cut_mean(min_max_normalize(item['data_x'])), dtype=np.float64)
            data_y = np.array(cut_mean(min_max_normalize(item['data_y'])), dtype=np.float64)
            data_z = np.array(cut_mean(min_max_normalize(item['data_z'])), dtype=np.float64)
        
            data = np.array([np.sqrt(data_x**2 + data_y**2), data_z], dtype=np.float64)
            # data = np.swapaxes(data, 0, 1)
            dataset.append(data)
            labels.append(item['label'])
    
    dataset = np.array(dataset)
    labels = np.array(labels)
    
    return dataset, labels

def create_dataset_3dims(input_tensor):
    # input: n_samples * {'data_x':, 'data_y':, 'data_z':, 'label':, 'file_name':}
    dataset = []
    labels = []
    for item in input_tensor:
        data_x = cut_mean(item['data_x'])
        data_y = cut_mean(item['data_y'])
        data_z = cut_mean(item['data_z'])
        data = np.array([data_x, data_y, data_z], dtype=np.float64)
        # data = np.swapaxes(data, 0, 1)
        dataset.append(data)
        labels.append(item['label'])
    dataset = np.array(dataset)
    labels = np.array(labels)
    
    return dataset, labels

def create_dataset_3dims_normalize(input_tensor):
    # input: n_samples * {'data_x':, 'data_y':, 'data_z':, 'label':, 'file_name':}
    dataset = []
    labels = []
    for item in input_tensor:
        data_x = cut_mean(min_max_normalize(item['data_x']))
        data_y = cut_mean(min_max_normalize(item['data_y']))
        data_z = cut_mean(min_max_normalize(item['data_z']))
        data = np.array([data_x, data_y, data_z], dtype=np.float64)
        # data = np.swapaxes(data, 0, 1)
        dataset.append(data)
        labels.append(item['label'])
    dataset = np.array(dataset)
    labels = np.array(labels)
    
    return dataset, labels

def create_dataset_3dims_by_filename(file_name, input_tensor):
    dataset = []
    labels = []
    for item in input_tensor:
        if item['file_name'] == file_name:
            data_x = cut_mean(item['data_x'])
            data_y = cut_mean(item['data_y'])
            data_z = cut_mean(item['data_z'])
            data = np.array([data_x, data_y, data_z], dtype=np.float64)
            # data = np.swapaxes(data, 0, 1)
            dataset.append(data)
            labels.append(item['label'])
    
    dataset = np.array(dataset)
    labels = np.array(labels)
    
    return dataset, labels

def create_dataset_3dims_normalize_by_filename(file_name, input_tensor):
    dataset = []
    labels = []
    for item in input_tensor:
        if item['file_name'] == file_name:
            data_x = cut_mean(min_max_normalize(item['data_x']))
            data_y = cut_mean(min_max_normalize(item['data_y']))
            data_z = cut_mean(min_max_normalize(item['data_z']))
            data = np.array([data_x, data_y, data_z], dtype=np.float64)
            # data = np.swapaxes(data, 0, 1)
            dataset.append(data)
            labels.append(item['label'])
    
    dataset = np.array(dataset)
    labels = np.array(labels)
    
    return dataset, labels

def handle_data_3dims(item, mode='later'):
    '''
    将单个切割出来的数据进行处理成 z, x2+y2, x2+y2+z2 三轴数据
    mode: 'former'-先减基线再合成，'later'-先合成再减基线
    '''
    base, angle = item['base_value'], item['angle'] # xyz的基线，以及其与g的夹角

    # data_x = np.array(item['data_x'], dtype=np.float64)
    # data_y = np.array(item['data_y'], dtype=np.float64)
    # data_z = np.array(item['data_z'], dtype=np.float64)
    data_x, data_y, data_z = item['data_x'], item['data_y'], item['data_z']

    data_xyz = np.sqrt((data_x-base[0])**2 + (data_y-base[1])**2 + (data_z-base[2])**2) # x2+y2+z2不论如何都减基线
    data_z_rectify = data_x * np.cos(angle['x']) + data_y * np.cos(angle['y']) + data_z * np.cos(angle['z']) # 修正过的z轴数据
    base_z = base[0] * np.cos(angle['x']) + base[0] * np.cos(angle['y']) + base[0] * np.cos(angle['z']) # 修正过的z轴baseline
    data_xy = np.sqrt((data_x-base[0])**2 + (data_y-base[1])**2)
    
    data_z_rectify = data_z_rectify - base_z # 修正过的z轴数据，并减去基线

    data = np.array([data_z_rectify, data_xy, data_xyz], dtype=np.float64)

    return data

def handle_dataset_3dims(dataset, file_name_list, mode='later'):
    '''
    对原始的数据进行处理，生成 data 与对应的 label
    detail: 若为True，则必须要输入file_name，会根据具体的文件名生成数据集
    mode: 'former'-先减基线再合成，'later'-先合成再减基线
    '''
    
    data = []
    label = []

    for item in dataset:
        if item['file_name'] in file_name_list:
            if mode == 'later' or 'former':
                data.append(handle_data_3dims(item, mode))
            else:
                data.append(np.array([item['data_x'], item['data_x'], item['data_x']]))
            label.append(item['label'])
    
    data = np.array(data, dtype=np.float64)
    label = np.array(label)
    return data, label
        

def generate_data(data_root, shuffle=True, factor=0.2):
    '''
    根据切割算法导入数据，并按文件划分训练集以及测试集
    其中训练集，测试集默认按 0.8 0.2 比例划分
    '''
    
    train_dataset = []
    test_dataset = []

    file_name_list = os.listdir(data_root)

    for file_name in file_name_list:

        file_path = data_root + '/' + file_name
        
        dataXYZ = pd.read_csv(file_path, header= 0)
        data_x, data_y, data_z = list(dataXYZ.iloc[:,0]), list(dataXYZ.iloc[:, 1]), list(dataXYZ.iloc[:, 2])
        base_value = cal_base_value(dataXYZ, 32, 16, 500)
        
        activity_list = activitySplit(dataXYZ, 32, 16, 500)
        activity_list = [int(np.mean(idx)) for idx in activity_list]

        activity_list = [{'data_x': np.array(extract_data_from_center(data_x, center, base_value[0])),
                        'data_y': np.array(extract_data_from_center(data_y, center, base_value[1])),
                        'data_z': np.array(extract_data_from_center(data_z, center, base_value[2])),
                        'label': get_label(file_name), 'file_name': file_name, 'base_value':base_value,
                        'angle': cal_angles(base_value)} for center in activity_list]
        
        if shuffle:
            random.shuffle(activity_list)
        
        test_dataset = test_dataset + activity_list[: int(factor * len(activity_list))]
        train_dataset = train_dataset + activity_list[int(factor * len(activity_list)): ]
    
    return train_dataset, test_dataset, file_name_list

def get_data(data_root, activity='dig', dis='1.0', num=1):
    """具体取出某距离上发生的某事件信号
    """
    total_activity = []
    file_name_list = [name for name in os.listdir(data_root) if activity in name and dis in name]
    for file_name in file_name_list:
        file_path = data_root + '/' + file_name
        dataXYZ = pd.read_csv(file_path, header= 0)
        data_x, data_y, data_z = list(dataXYZ.iloc[:,0]), list(dataXYZ.iloc[:, 1]), list(dataXYZ.iloc[:, 2])
        base_value = cal_base_value(dataXYZ, 32, 16, 500)
        
        activity_list = activitySplit(dataXYZ, 32, 16, 500)
        activity_list = [int(np.mean(idx)) for idx in activity_list]

        activity_list = [{'data_x': np.array(extract_data_from_center(data_x, center, base_value[0])),
                        'data_y': np.array(extract_data_from_center(data_y, center, base_value[1])),
                        'data_z': np.array(extract_data_from_center(data_z, center, base_value[2])),
                        'label': get_label(file_name), 'file_name': file_name, 'base_value':base_value,
                        'angle': cal_angles(base_value)} for center in activity_list]
        
        total_activity += activity_list

    # print(activity, dis, len(total_activity))
    return random.choices(total_activity, k=num)

def generate_data_by_txtfile(data_root, txt_root, shuffle=True, factor=0.2):
    '''
    根据打了标签的 txt 文件导入数据，并按文件来划分训练集以及测试集
    其中训练集，测试集默认按 0.8 0.2 比例划分
    '''
    
    train_dataset = []
    test_dataset = []

    file_name_list = os.listdir(data_root)

    for file_name in file_name_list:
        # print('Load:', d)
        file_path = data_root + '/' + file_name
        
        dataXYZ = pd.read_csv(file_path, header= 0)
        data_x, data_y, data_z = list(dataXYZ.iloc[:,0]), list(dataXYZ.iloc[:, 1]), list(dataXYZ.iloc[:, 2])
        base_value = cal_base_value(dataXYZ, 32, 16, 500)
        
        txt_path = txt_root + '/' + file_name[:-3] + 'txt'
        with open(txt_path, 'r') as f:
            activity_list = f.readlines()
        activity_list = [int(activity[:-1]) for activity in activity_list]

        activity_list = [{'data_x': np.array(extract_data_from_center(data_x, center, base_value[0])),
                        'data_y': np.array(extract_data_from_center(data_y, center, base_value[1])),
                        'data_z': np.array(extract_data_from_center(data_z, center, base_value[2])),
                        'label': get_label(file_name), 'file_name': file_name, 'base_value':base_value,
                        'angle': cal_angles(base_value)} for center in activity_list]
        
        if shuffle:
            random.shuffle(activity_list)
        
        test_dataset = test_dataset + activity_list[: int(factor * len(activity_list))]
        train_dataset = train_dataset + activity_list[int(factor * len(activity_list)): ]
    
    return train_dataset, test_dataset, file_name_list


def generate_data_md_by_txtfile(data_root, txt_root, shuffle=True, factor=0.2):
    train_dataset = []
    test_dataset = []

    file_name_list = os.listdir(data_root)

    for file_name in file_name_list:
        # print('Load:', d)
        file_path = data_root + '/' + file_name
        
        dataXYZ = pd.read_csv(file_path, header= 0)
        data_x, data_y, data_z = list(dataXYZ.iloc[:,0]), list(dataXYZ.iloc[:, 1]), list(dataXYZ.iloc[:, 2])
        base_value = cal_base_value(dataXYZ, 32, 16, 500)
        
        txt_path = txt_root + '/' + file_name[:-3] + 'txt'
        with open(txt_path, 'r') as f:
            activity_list = f.readlines()
        activity_list = [int(activity[:-1]) for activity in activity_list]

        total_activity = []
        for center in activity_list:
            temp_x = np.array(extract_data_from_center(data_x, center, base_value[0]))
            temp_y = np.array(extract_data_from_center(data_y, center, base_value[0]))
            
            for i in range(12):
                route_angle = np.pi * i / 6
                total_activity.append({'data_x': temp_x * np.cos(route_angle) - temp_y * np.sin(route_angle),
                                        'data_y': temp_x * np.sin(route_angle) + temp_y * np.cos(route_angle),
                                        'data_z': np.array(extract_data_from_center(data_z, center, base_value[2])),
                                        'label': get_label(file_name), 'file_name': file_name, 'base_value':base_value,
                                        'angle': cal_angles(base_value), 'route_angle': route_angle})

        if shuffle:
            random.shuffle(total_activity)
        
        test_dataset = test_dataset + total_activity[: int(factor * len(total_activity))]
        train_dataset = train_dataset + total_activity[int(factor * len(total_activity)): ]
    
    return train_dataset, test_dataset, file_name_list

def make_place_data(root, place='zwy_d1', num=10):
    """为一个场地构建实验需要的数据集
    """
    total_data = {}
    data_root = root + place + '/data'

    for activity in ['dig', 'jump', 'walk']:
        dis_list = ['0.5', '1.5', '2.5'] if place=='yqcc2' and activity=='walk' else ['1.0', '3.0', '5.0']
        for dis in dis_list:
            total_data[activity + dis] = get_data(data_root, activity, dis, num)

    return total_data

if __name__ == "__main__":
    # data_root = 'E:/研一/嗑盐/土壤扰动/dataset/zwy_d1/data'
    # data = get_data(data_root, num=10)
    # print(len(data))
    # print(data[0]['data_z'])
    root = 'E:/研一/嗑盐/土壤扰动/dataset/'
    zwy_data = make_place_data(root, 'zwy_d1', 10)
    print(zwy_data)