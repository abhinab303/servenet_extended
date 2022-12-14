
from __future__ import division
from __future__ import print_function
import pickle
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

import scipy.sparse as sp
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn import GraphConv

from transformers import BertModel
from model.multi_head import weighted_sum3, MutliHead

from utils import load_data_train_names, load_data_test_names, evaluteTop5_names, evaluteTop1_names

import pdb

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

# pdb.set_trace()

# load gcn model:

Train_C = api_dataframe.iloc[0:len(train_df)]
Test_C = api_dataframe.iloc[len(train_df):]

print(Train_C.shape)
print(Test_C.shape)

Trainlabelcount = Train_C['ServiceClassification'].value_counts()
trainP = Trainlabelcount / Trainlabelcount.sum()
Testlabelcount = Test_C['ServiceClassification'].value_counts()
Testlabelcount = Testlabelcount[Trainlabelcount.index]
TestP = Testlabelcount / Testlabelcount.sum()
comparedf = pd.DataFrame({'Training Set': trainP, 'Test Set': TestP})

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
        # return x
        return F.log_softmax(output, dim=1)

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
# labels = encode_onehot(api_dataframe['ServiceClassification'])
# features = normalize(features)
idx_train = TrainIndex
idx_val = TestIndex
idx_test = TestIndex

features = torch.FloatTensor(np.array(features.todense()))
# labels = torch.LongTensor(np.where(labels)[1])

idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

g = dgl.from_networkx(graph_copy).to('cuda')
g = dgl.add_self_loop(g)

cuda = torch.cuda.is_available()
seed = 42
hidden = [1024, 1024]
dropout = 0.5

np.random.seed(seed)
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)


gcn_model = GCN(nfeat=features.shape[1],
                nhid1=hidden[0],
                nhid2=hidden[1],
                nclass=50,
                dropout=dropout)

if cuda:
    gcn_model.cuda()
    features = features.cuda()
    # labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


gcn_model = torch.load("/home/aa7514/PycharmProjects/servenet_extended/files/gcn_full_model4")
for param in gcn_model.parameters():
    param.requires_grad = False
gcn_model.eval()
gcn_op = gcn_model(g, features)
print(gcn_op.shape)

comb = (0.5, 0.5)

class ServeNet(torch.nn.Module):
    def __init__(self, hiddenSize, CLASS_NUM):
        super(ServeNet, self).__init__()
        self.hiddenSize = hiddenSize

        self.bert_name = BertModel.from_pretrained('bert-base-uncased')
        self.bert_description = BertModel.from_pretrained('bert-base-uncased')

        self.name_liner = nn.Linear(in_features=self.hiddenSize, out_features=1024)
        self.name_ReLU = nn.ReLU()
        self.name_Dropout = nn.Dropout(p=0.1)

        self.lstm = nn.LSTM(input_size=self.hiddenSize, hidden_size=512, num_layers=1, batch_first=True,
                            bidirectional=True)

        self.weight_sum = weighted_sum3()
        self.mutliHead = MutliHead(num_classes=CLASS_NUM)
        # self.gcn = GCN(nfeat=features.shape[1],
        #                nhid1=hidden[0],
        #                nhid2=hidden[1],
        #                nclass=labels.max().item() + 1,
        #                dropout=dropout)

    def forward(self, names, descriptions, indices):
        self.lstm.flatten_parameters()

        # name
        name_bert_output = self.bert_name(**names)
        # Feature for Name
        name_features = self.name_liner(name_bert_output[1])
        name_features = self.name_ReLU(name_features)
        name_features = self.name_Dropout(name_features)

        # description
        description_bert_output = self.bert_description(**descriptions)

        description_bert_feature = description_bert_output[0]

        # LSTM
        packed_output, (hidden, cell) = self.lstm(description_bert_feature)
        hidden = torch.cat((cell[0, :, :], cell[1, :, :]), dim=1)
        # hidden = torch.cat((hidden[0, :, :], hidden[1, :, :]), dim=1)

        # from_gcn = torch.take(gcn_op, indices)
        # from_gcn = gcn_op[torch.add(indices, 7081)]
        from_gcn = gcn_op[indices]

        # sum
        all_features = self.weight_sum(name_features, hidden, from_gcn)
        output = self.mutliHead(all_features)

        final_output = comb[0] * output + comb[1] * from_gcn

        # pdb.set_trace()

        return final_output
        # return F.log_softmax(output, dim=1)


sn_model = torch.load("/home/aa7514/PycharmProjects/servenet_extended/files/combined_model_w4")
# model = torch.nn.DataParallel(model)
for param in sn_model.parameters():
    param.requires_grad = False
sn_model.eval()

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# train_data = load_data_train_names(50)
test_data = load_data_test_names(50)

# train_dataloader = DataLoader(train_data, batch_size=56)
test_dataloader = DataLoader(test_data, batch_size=56)

print("=======>top1 acc on the test:{}".format(str(evaluteTop1_names(sn_model, test_dataloader, 50))))
print("=======>top5 acc on the test:{}".format(str(evaluteTop5_names(sn_model, test_dataloader))))

top1 = []
top5 = []
comb_list = []
for i in np.linspace(0, 1, 50):
    print(i)
    j = 1 - i
    comb = (i, j)
    top1.append(evaluteTop1_names(sn_model, test_dataloader, 50))
    top5.append(evaluteTop5_names(sn_model, test_dataloader))
    comb_list.append(f"{i}, {j}")

acc_list = pd.DataFrame(
    {'Comb': comb_list,
     'Top1': top1,
     'Top5': top5
    })

acc_list.to_csv('/home/aa7514/PycharmProjects/servenet_extended/files/aggregate_50.csv')



# pdb.set_trace()
