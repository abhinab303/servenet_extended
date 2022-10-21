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

cuda = torch.cuda.is_available()

class ServeNet(torch.nn.Module):
    def __init__(self, hiddenSize, CLASS_NUM):
        super(ServeNet, self).__init__()
        self.hiddenSize = hiddenSize
        self.bert_description = BertModel.from_pretrained('bert-base-uncased')

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

        return output

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
