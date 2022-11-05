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
#
# for c in set(api_dataframe['ServiceClassification']):
#     C_data = api_dataframe[api_dataframe['ServiceClassification'] == c]
#     # print(C_data.shape)
#     C_Train, C_Test = train_test_split(C_data, test_size=0.2, random_state=0)
#     Train.append(C_Train)
#     Test.append(C_Test)
#
# Train_C = pd.concat(Train)
# Test_C = pd.concat(Test)

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
    classes = set(labels)
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
        # output = self.final_liner(x)
        # return F.log_softmax(x, dim=1)
        return x
        # return F.log_softmax(output, dim=1)


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
epochs = 40  # 2000
lr = 0.001
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
# gcn_model = GCN(nfeat=features.shape[1],
#                 nhid1=hidden[0],
#                 nhid2=hidden[1],
#                 nclass=labels.max().item() + 1,
#                 dropout=dropout)

# criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.Adam(gcn_model.parameters(),
#                        lr=lr,
#                        #  weight_decay=weight_decay
#                        )
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0)

if cuda:
    # gcn_model.cuda()
    features = features.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

# gcn_model.load_state_dict(torch.load("/home/aa7514/PycharmProjects/servenet_extended/files/gcn_full_model4"))
# gcn_model = torch.load("/home/aa7514/PycharmProjects/servenet_extended/files/gcn_full_model4")
# for param in gcn_model.parameters():
#     param.requires_grad = False
# gcn_model.eval()
# gcn_op = gcn_model(g, features)
# print(gcn_op.shape)


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
        self.gcn = GCN(nfeat=features.shape[1],
                       nhid1=hidden[0],
                       nhid2=hidden[1],
                       nclass=labels.max().item() + 1,
                       dropout=dropout)

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
        # from_gcn = gcn_op[indices]

        gcn_op = self.gcn(g, features)
        from_gcn = gcn_op[indices]

        # sum
        all_features = self.weight_sum(name_features, hidden, from_gcn)
        output = self.mutliHead(all_features)

        return output

epochs = 100
SEED = 123
LEARNING_RATE = 0.001
# LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.01
EPSILON = 1e-8
BATCH_SIZE = 56
CLASS_NUM = 50
cat_num = "50"

des_max_length=110 #110#160#200
name_max_length=10

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

train_data = load_data_train_names(CLASS_NUM)
test_data = load_data_test_names(CLASS_NUM)

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

model = ServeNet(768, CLASS_NUM)
# model.bert_description.requires_grad_(False)
# model.bert_name.requires_grad_(False)
model = torch.nn.DataParallel(model)
model = model.cuda()
model.train()

pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
pytorch_total_params_all = sum(p.numel() for p in model.parameters())
print("Trainable: ", pytorch_total_params_trainable)
print("All: ", pytorch_total_params_all)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

for epoch in range(epochs):
    print("Epoch:{},lr:{}".format(str(epoch + 1), str(optimizer.state_dict()['param_groups'][0]['lr'])))
    # scheduler.step()
    model.train()
    for data in tqdm(train_dataloader):

        # zero the parameter gradients
        optimizer.zero_grad()

        descriptions = {'input_ids': data[0].cuda(),
                        'token_type_ids': data[1].cuda(),
                        'attention_mask': data[2].cuda()
                        }

        names = {'input_ids': data[4].cuda(),
                 'token_type_ids': data[5].cuda(),
                 'attention_mask': data[6].cuda()
                 }
        indices = data[7].cuda()
        label = data[3].cuda()

        outputs = model(names, descriptions, indices)

        # outputs = model()

        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

    w11 = model.module.weight_sum.w1
    w22 = model.module.weight_sum.w2
    w33 = model.module.weight_sum.w3
    w1 = w11 / (w11 + w22 + w33)
    w2 = w22 / (w11 + w22 + w33)
    w3 = w33 / (w11 + w22 + w33)
    print("w11: ", w11)
    print("w22: ", w22)
    print("w33: ", w33)
    print("w1: ", w1)
    print("w2: ", w2)
    print("w3: ", w3)

    print("=======>top1 acc on the test:{}".format(str(evaluteTop1_names(model, test_dataloader, CLASS_NUM))))
    print("=======>top5 acc on the test:{}".format(str(evaluteTop5_names(model, test_dataloader))))

# print("=======>top1 acc on the test:{}".format(str(evaluteTop1_names(model, test_dataloader, CLASS_NUM, True))))

# ### to do :
# # check if the index corresponds to the same data in the sentence and graph as well
# torch.save(model, "/home/aa7514/PycharmProjects/servenet_extended/files/combined_model_w4") # important 73.36@epoch40
# torch.save(model, "/home/aa7514/PycharmProjects/servenet_extended/files/combined_model")

# print(model)
# print("=======>top1 acc on the test:{}".format(str(evaluteTop1_names(model, test_dataloader, CLASS_NUM))))
# print("=======>top5 acc on the test:{}".format(str(evaluteTop5_names(model, test_dataloader))))
