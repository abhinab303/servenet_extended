from __future__ import division
from __future__ import print_function

import pdb
import pickle
import pandas as pd
import numpy as np
import json
import networkx as nx

import scipy.sparse as sp
import torch
import math

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.nn as nn
import torch.nn.functional as F

import time
import argparse

import torch.optim as optim

import dgl
import torch as th
from dgl.nn import GraphConv
import random

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

test_path = "/home/aa7514/PycharmProjects/servenet_extended/data/50/test.csv"
train_path = "/home/aa7514/PycharmProjects/servenet_extended/data/50/train.csv"
graph_path = "/home/aa7514/PycharmProjects/servenet_extended/files/graph.pickle"
feature_path = "/home/aa7514/PycharmProjects/servenet_extended/files/feature_matrix.pickle"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
train_df["ServiceDescription"] = train_df["ServiceDescription"].str.replace(r'[^A-Za-z .]+', '', regex=True)
test_df["ServiceDescription"] = test_df["ServiceDescription"].str.replace(r'[^A-Za-z .]+', '', regex=True)
api_dataframe = pd.concat([train_df, test_df], axis=0)
api_dataframe.reset_index(inplace=True, drop=True)

from sklearn.model_selection import train_test_split

# Train = []
# Test = []

# for c in set(api_dataframe['ServiceClassification']):
#     C_data = api_dataframe[api_dataframe['ServiceClassification'] == c]
#     # print(C_data.shape)
#     C_Train, C_Test = train_test_split(C_data, test_size=0.2, random_state=0)
#     Train.append(C_Train)
#     Test.append(C_Test)

# Train_C = pd.concat(Train)
# Test_C = pd.concat(Test)

Train_C = api_dataframe.iloc[0:len(train_df)]
Test_C = api_dataframe.iloc[len(train_df):]

print(Train_C.shape)
print(Test_C.shape)
# pdb.set_trace()
Trainlabelcount = Train_C['ServiceClassification'].value_counts()
trainP = Trainlabelcount / Trainlabelcount.sum()
Testlabelcount = Test_C['ServiceClassification'].value_counts()
Testlabelcount = Testlabelcount[Trainlabelcount.index]
TestP = Testlabelcount / Testlabelcount.sum()
comparedf = pd.DataFrame({'Training Set': trainP, 'Test Set': TestP})
# print(comparedf)


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def encode_onehot(labels):
    classes = sorted(list(set(labels)), key=str.lower)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def evaluateTop1(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def evaluateTop5(output, labels):
    y_resize = labels.view(-1, 1)
    _, pred = output.topk(5, 1, True, True)
    correct = torch.eq(pred, y_resize).sum()
    return correct / len(labels)


def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device


def relu_evidence(y):
    return F.relu(y)


def kl_divergence(alpha, num_classes, device=None):
    if not device:
        device = get_device()
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div


def edl_log_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.log, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss


class GCN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConv(nfeat, nhid1, norm='both', weight=True, bias=True)
        self.gc2 = GraphConv(nhid1, nhid2, norm='both', weight=True, bias=True)
        # self.gc3 = GraphConvolution(nhid2, nclass)
        self.dropout = dropout
        self.final_liner = nn.Linear(in_features=nhid2, out_features=nclass)

    def forward(self, g, x):
        x = F.relu(self.gc1(g, x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(g, x))
        x = F.dropout(x, self.dropout, training=self.training)
        output = self.final_liner(x)
        # return F.log_softmax(x, dim=1)
        # return F.log_softmax(output, dim=1)
        return output


with open(graph_path, "rb") as f:
    graph = pickle.load(f)

with open(feature_path, "rb") as f:
    feature_matrix = pickle.load(f)

training_data = Train_C
testing_data = Test_C
TrainIndex = training_data.index.values.tolist()
TestIndex = testing_data.index.values.tolist()

graph_copy = graph.copy()
for node_index, node in graph_copy.nodes(data=True):
    try:
        node["feature"] = feature_matrix[node_index]
    except Exception as ex:
        continue

features = sp.csr_matrix(feature_matrix)
labels = encode_onehot(api_dataframe['ServiceClassification'])
# features = normalize(features)
idx_train = TrainIndex
idx_val = TestIndex
idx_test = TestIndex

features = torch.FloatTensor(np.array(features.todense()))
labels = torch.LongTensor(np.where(labels)[1])

idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

g = dgl.from_networkx(graph_copy).to('cuda')
g = dgl.add_self_loop(g)

cuda = torch.cuda.is_available()
fastmode = False
seed = 42
epochs = 3000  # 2000
lr = 0.01
# weight_decay = 5e-4
weight_decay = 0
hidden = [1024, 1024]  # [2048, 1024]    # [256, 128]
dropout = 0.5  # 0.95   # 0.4
T_0 = 10  # Number of iterations for the first restart.
patience = 50

np.random.seed(seed)
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

# Load data
# adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid1=hidden[0],
            nhid2=hidden[1],
            nclass=labels.max().item() + 1,
            dropout=dropout)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),
                       lr=lr,
                       #  weight_decay=weight_decay
                       )
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0)


if cuda:
    model.cuda()
    features = features.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

val_losses = []
train_losses = []

val_acc = []
train_acc = []
val_acc_5 = []
train_acc_5 = []

last_loss = 100
trigger_times = 0


def train(epoch):
    global last_loss, trigger_times
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(g, features)
    # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    # loss_train = criterion(output[idx_train], labels[idx_train])
    loss_train = edl_log_loss(output[idx_train], labels[idx_train], epoch, 50, 10)
    acc_train = accuracy(output[idx_train], labels[idx_train])
    top5_train = evaluateTop5(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    # scheduler.step()

    if not fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(g, features)

    # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    # loss_val = criterion(output[idx_val], labels[idx_val])
    loss_val = edl_log_loss(output[idx_val], labels[idx_val], epoch, 50, 10)
    acc_val = accuracy(output[idx_val], labels[idx_val])
    top5_val = evaluateTop5(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'acc_top5: {:.4f}'.format(top5_val.item()),
          'time: {:.4f}s'.format(time.time() - t),
          # f'LR: {scheduler.get_last_lr()}'
          )

    train_losses.append(loss_train.item())
    val_losses.append(loss_val.item())
    train_acc.append(acc_train.item())
    val_acc.append(acc_val.item())
    train_acc_5.append(top5_train.item())
    val_acc_5.append(top5_val.item())

    current_loss = loss_val.item()
    if current_loss > last_loss:
        trigger_times += 1
    last_loss = current_loss


def test():
    model.eval()
    output = model(g, features)
    # loss_test = criterion(output[idx_test], labels[idx_test])
    loss_test = edl_log_loss(output[idx_test], labels[idx_test], epoch, 50, 10)
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()

for epoch in range(epochs):
    train(epoch)
    if trigger_times >= patience:
        print('Early stopping!\nStart to test process.')
        break
    else:
        trigger_times = 0
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()

# save the model:
# torch.save(model.state_dict(), "/home/aa7514/PycharmProjects/servenet_extended/files/gcn_model_not_random")
torch.save(model, "/home/aa7514/PycharmProjects/servenet_extended/files/gcn_full_model5")
pass