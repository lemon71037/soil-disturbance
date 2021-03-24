from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import torch


def str_to_int_list(s):
    return [int(x) for x in s.split()]


def str_list_to_int_list(slist):
    return [str_to_int_list(s) for s in slist]


def batch_padding(q, q_len):
    max_len = np.max(q_len)
    for i in range(len(q)):
        q[i] = [0] * (max_len - q_len[i]) + q[i]
    return np.array(q)


def my_collate(batch):
    q1 = [item[0] for item in batch]
    q2 = [item[1] for item in batch]
    q1_len = np.array([item[2] for item in batch])
    q2_len = np.array([item[3] for item in batch])

    q1 = torch.from_numpy(batch_padding(q1, q1_len)).long()
    q2 = torch.from_numpy(batch_padding(q2, q2_len)).long()

    if len(batch[0]) == 5:
        a = torch.from_numpy(np.array([[item[4]] for item in batch])).float()
        return [q1, q2, a]

    return [q1, q2]


class OppoQuerySet(Dataset):
    def __init__(self, df, dataset='train'):

        q1 = str_list_to_int_list(df['q1'].values)
        q2 = str_list_to_int_list(df['q2'].values)
        self.dataset = dataset

        if dataset is not 'test':
            a = df['label'].values
            self.data = zip(q1, q2, a)
        else:
            self.data = zip(q1, q2)

        self.data = list(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        q1, q2 = self.data[index][:2]
        q1_len = len(q1)
        q2_len = len(q2)
        if self.dataset == 'test':
            return q1, q2, q1_len, q2_len
        else:
            return q1, q2, q1_len, q2_len, self.data[index][2]


if __name__ == '__main__':
    df_train = pd.read_table("./baseline_tfidf_lr/oppo_breeno_round1_data/gaiic_track3_round1_train_20210228.tsv",
                             names=['q1', 'q2', 'label']).fillna("0")
    df_test = pd.read_table('./baseline_tfidf_lr/oppo_breeno_round1_data/gaiic_track3_round1_testA_20210228.tsv',
                            names=['q1', 'q2']).fillna("0")

    max_value = -1
    min_value = 999999
    max_len = -1

    trainset = OppoQuerySet(df_train, dataset='train')
    trainloader = DataLoader(trainset, batch_size=10, shuffle=True, collate_fn=my_collate)
    for q1, q2, a in trainloader:
        print(q1)
        print(q1.size())
        print(q2)
        print(q2.size())
        print(a)
        break

    # testset = OppoQuerySet(df_test, dataset='test')
    # # for i in range(len(trainset)):
    # #     q1, q2, a = trainset[i]
    # #     max_value = max(max_value, max(q1))
    # #     max_value = max(max_value, max(q2))
    # #     min_value = min(min_value, min(q1))
    # #     min_value = min(min_value, min(q2))
    # #     max_len = max(len(q1), max_len)
    # #     max_len = max(len(q2), max_len)
    #
    # # for i in range(len(testset)):
    # #     q1, q2 = testset[i]
    # #     max_value = max(max_value, max(q1))
    # #     max_value = max(max_value, max(q2))
    # #     min_value = min(min_value, min(q1))
    # #     min_value = min(min_value, min(q2))
    # #     max_len = max(len(q1), max_len)
    # #     max_len = max(len(q2), max_len)
    #
    # q1, q2, a = trainset[231]
    #
    # print(q1.size(), q2.size(), a)
