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

training_data = Train_C
testing_data = Test_C
TrainIndex = training_data.index.values.tolist()
TestIndex = testing_data.index.values.tolist()

idx_train = TrainIndex
idx_val = TestIndex
idx_test = TestIndex

idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

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

if cuda:
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


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

        # sum
        all_features = self.weight_sum(name_features, hidden, hidden)
        output = self.mutliHead(all_features)

        return all_features

epochs = 40
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

sn_model = torch.load("/home/aa7514/PycharmProjects/servenet_extended/files/combined_model_w4")
# model = torch.nn.DataParallel(model)
for param in sn_model.parameters():
    param.requires_grad = False
sn_model.eval()


class EmbTest(torch.nn.Module):
    def __init__(self, hiddenSize, CLASS_NUM):
        super(EmbTest, self).__init__()
        self.mutliHead = MutliHead(num_classes=CLASS_NUM)

    def forward(self, names, descriptions, indices):
        all_features = sn_model(names, descriptions, indices)
        output = self.mutliHead(all_features)
        return output


model = EmbTest(768, 50)
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

    print("=======>top1 acc on the test:{}".format(str(evaluteTop1_names(model, test_dataloader, CLASS_NUM))))
    print("=======>top5 acc on the test:{}".format(str(evaluteTop5_names(model, test_dataloader))))