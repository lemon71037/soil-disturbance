import torch
import random
import numpy as np
from dataset import Soil2ClassSet
from models import Soil2ClassModel
from data_process import generate_data
from torch.utils.data import Dataset, DataLoader
from utils import LambdaLR, Logger, weights_init_normal
from sklearn.metrics import accuracy_score

d_model = 64        # 扩张维度数
nhead = 4           # 多头数
dim_feedforward = 256 # 前馈神经网络数
lr = 0.002          # learning rate
epoch = 0
n_epochs = 50
decay_epoch = 25
batchSize = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Define Dataset """
syf_train, syf_test, _ = generate_data('E:/研一/嗑盐/土壤扰动/dataset/syf')
yqcc_train, yqcc_test, _ = generate_data('E:/研一/嗑盐/土壤扰动/dataset/yqcc2')
zwy_train, zwy_test, _ = generate_data('E:/研一/嗑盐/土壤扰动/dataset/zwy')

train_data = syf_train + yqcc_train + zwy_train
test_data = syf_test + yqcc_test + zwy_test

random.shuffle(train_data)
trainset = Soil2ClassSet(train_data, mode='origin')
testset = Soil2ClassSet(test_data, mode='origin')
trainloader = DataLoader(trainset, batch_size=batchSize, shuffle=True)
testloader = DataLoader(testset, batch_size=batchSize, shuffle=False)

""" Define Model """
model = Soil2ClassModel(d_model, nhead, dim_feedforward)
model.apply(weights_init_normal)
model = model.to(device)

""" Define Optim and Criterion"""
optim = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)

""" Train and Eval """
logger = Logger(n_epochs, len(trainloader))
max_test_score = 0.0

for epoch in range(n_epochs):
    
    test_acc = 0.0
    test_num = len(testloader)

    model.train()
    for i, (sig1, sig2, label) in enumerate(trainloader):
        sig1, sig2, label = sig1.float().to(device), \
            sig2.float().to(device), label.long().to(device)
        
        pred = model(sig1, sig2)
        loss = criterion(pred, label)
        acc = accuracy_score(torch.argmax(pred, dim=1).cpu(), label.cpu())

        optim.zero_grad()
        loss.backward()
        optim.step()

        logger.log({"loss": loss, "acc": acc})
        lr_scheduler.step()
    
    model.eval()
    with torch.no_grad():
        for i, (sig1, sig2, label) in enumerate(testloader):
            sig1, sig2, label = sig1.float().to(device), \
                sig2.float().to(device), label.long().to(device)
        
            pred = model(sig1, sig2)
            # loss = criterion(pred, label)
            acc = accuracy_score(torch.argmax(pred, dim=1).cpu(), label.cpu())
            test_acc += acc
        
        test_acc = test_acc / test_num
        print("Epoch: {}/{} \t test acc: {}".format(epoch, n_epochs, test_acc.item()))

    if test_acc.item() > max_test_score:
        max_val_score = logger.metric_list['acc'][-1]
        torch.save(model.state_dict(), 'state_dicts/Soil2ClassModel.pth')

