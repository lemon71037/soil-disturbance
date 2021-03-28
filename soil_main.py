import torch
import random
import numpy as np
import torch.nn.functional as F
from dataset import SoilActDataset
from models import CNNClassifier
from data_process import generate_data
from torch.utils.data import Dataset, DataLoader
from utils import LambdaLR, Logger, weights_init_normal
from sklearn.metrics import accuracy_score

# d_model = 64        # 扩张维度数
# nhead = 4           # 多头数
# dim_feedforward = 256 # 前馈神经网络数
lr = 0.005            # learning rate
epoch = 0
n_epochs = 100
decay_epoch = 30
batchSize = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Define Dataset """
syf_train, syf_test, _ = generate_data('E:/研一/嗑盐/土壤扰动/dataset/syf')
yqcc_train, yqcc_test, _ = generate_data('E:/研一/嗑盐/土壤扰动/dataset/yqcc2')
zwy_train, zwy_test, _ = generate_data('E:/研一/嗑盐/土壤扰动/dataset/zwy')

train_data = syf_train + yqcc_train + zwy_train
test_data = syf_test + yqcc_test + zwy_test

# random.shuffle(train_data)
# random.shuffle(test_data)

trainset = SoilActDataset(train_data, mode='origin', use_soil=False)
testset = SoilActDataset(test_data, mode='origin', use_soil=False)
trainloader = DataLoader(trainset, batch_size=batchSize, shuffle=True)
testloader = DataLoader(testset, batch_size=batchSize, shuffle=False)

""" Define Model """
model = CNNClassifier()
model.apply(weights_init_normal)
model = model.to(device)
# torch.save(model.state_dict(), 'state_dicts/OriginModel.pth')
# origin = Soil2ClassModel(d_model, nhead, dim_feedforward)
# origin.load_state_dict(torch.load('state_dicts/OriginModel.pth'))

# for name, p in model.named_parameters():
#     print(name, p.size())

# model.eval()
# test_acc = 0.0
# test_num = len(testloader)
# with torch.no_grad():
#     # print(model.named_parameters() == origin.named_parameters())
#     for i, (sig1, sig2, label) in enumerate(testloader):
#         sig1, sig2, label = sig1.float().to(device), \
#             sig2.float().to(device), label.long().to(device)
    
#         pred = model(sig1, sig2)
#         acc = accuracy_score(torch.argmax(pred, dim=1).cpu(), label.cpu())
#         test_acc += acc
    
#     test_acc = test_acc / test_num
#     print("Origin test acc: {}".format(test_acc.item()))

""" Define Optim and Criterion"""
optim = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.001)
criterion = torch.nn.NLLLoss()
# lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, 20, 0.5)

""" Train and Eval """
logger = Logger(n_epochs, len(trainloader))
max_test_score = 0.0

for epoch in range(n_epochs):
    
    test_acc = 0.0
    test_num = len(testloader)

    model.train()
    for i, (sig, label) in enumerate(trainloader):
        sig, label = sig.float().to(device), label.long().to(device)
        
        pred = model(sig)
        loss = criterion(pred, label)

        optim.zero_grad()
        loss.backward()
        optim.step()
        # for x in optim.param_groups[0]['params']:
        acc = accuracy_score(torch.argmax(pred, dim=1).cpu(), label.cpu())
        logger.log({"loss": loss, "acc": acc})
        
    lr_scheduler.step()

    model.eval()
    with torch.no_grad():
        # print(model.named_parameters() == origin.named_parameters())
        for i, (sig, label) in enumerate(testloader):
            sig, label = sig.float().to(device), label.long().to(device)
        
            pred = model(sig)
            acc = accuracy_score(torch.argmax(pred, dim=1).cpu(), label.cpu())
            test_acc += acc
        
        test_acc = test_acc / test_num
        print("Epoch: {}/{} \t test acc: {}".format(epoch, n_epochs, test_acc.item()))

    if test_acc.item() > max_test_score:
        print("saved new model!")
        max_test_score = test_acc.item()
        torch.save(model.state_dict(), 'state_dicts/Act3ClassModel.pth')
    # elif test_acc.item() == max_test_score:
    #     print("never change")
    # else:
    #     print("???")
