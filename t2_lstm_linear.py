import torch
import torch.nn as nn
from transformers import BertModel
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset
import numpy as np
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm
import pdb

ip_file_dir = "/home/aa7514/PycharmProjects/servenet_extended/data/"
category_num = 50
max_len = 110
BATCH_SIZE = 56
LEARNING_RATE = 0.001
epochs = 40


def encode_onehot(labels):
    # classes = set(labels)
    classes = classes = sorted(list(set(labels)), key=str.lower)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data_train():
    train_file = f"{ip_file_dir}{category_num}/train.csv"
    df = pd.read_csv(train_file)
    values = np.array(df.ServiceClassification)
    label_encoder = LabelEncoder()
    # integer_encoded2 = label_encoder.fit_transform(values)
    # integer_encoded = encode_onehot(df.ServiceClassification).transpose(1, 0)
    integer_encoded = torch.LongTensor(np.where(encode_onehot(df.ServiceClassification))[1])
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # descriptions
    descriptions = df["ServiceDescription"].tolist()
    desc_tokens = tokenizer(descriptions, return_tensors="pt",
                            max_length=max_len,
                            padding=True,
                            truncation=True)

    # names
    names = df["ServiceName"].tolist()
    name_tokens = tokenizer(names, return_tensors="pt",
                            #  model_max_length=100,
                            max_length=max_len,
                            padding=True,
                            truncation=True)

    total_targets = torch.tensor(integer_encoded)

    desc_list = []
    for key, value in desc_tokens.items():
        desc_list.append(torch.tensor(value))

    name_list = []
    for key, value in name_tokens.items():
        name_list.append(torch.tensor(value))

    train_data = TensorDataset(*desc_list, total_targets, *name_list, torch.tensor(df.index.values))

    return train_data


def load_data_test():
    test_file = f"{ip_file_dir}{category_num}/test.csv"
    train_file = f"{ip_file_dir}{category_num}/train.csv"
    train_df = pd.read_csv(train_file)
    df = pd.read_csv(test_file)
    values = np.array(df.ServiceClassification)
    label_encoder = LabelEncoder()
    # integer_encoded = label_encoder.fit_transform(values)
    # integer_encoded = encode_onehot(df.ServiceClassification).transpose(1, 0)
    integer_encoded = torch.LongTensor(np.where(encode_onehot(df.ServiceClassification))[1])
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # descriptions
    descriptions = df["ServiceDescription"].tolist()
    desc_tokens = tokenizer(descriptions, return_tensors="pt",
                            max_length=max_len,
                            padding=True,
                            truncation=True)

    # names
    names = df["ServiceName"].tolist()
    name_tokens = tokenizer(names, return_tensors="pt",
                            max_length=max_len,
                            padding=True,
                            truncation=True)

    total_targets = integer_encoded

    desc_list = []
    for key, value in desc_tokens.items():
        desc_list.append(torch.tensor(value))

    name_list = []
    for key, value in name_tokens.items():
        name_list.append(torch.tensor(value))

    test_data = TensorDataset(*desc_list, total_targets, *name_list, torch.tensor(df.index.values + len(train_df)))

    return test_data


def eval_top1(model, dataLoader, class_num=50, per_class=False):
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(class_num))
    class_total = list(0. for i in range(class_num))
    with torch.no_grad():
        for data in dataLoader:
            descriptions = {'input_ids': data[0].cuda(),
                            'token_type_ids': data[1].cuda(),
                            'attention_mask': data[2].cuda()
                            }

            names = {'input_ids': data[4].cuda(),
                     'token_type_ids': data[5].cuda(),
                     'attention_mask': data[6].cuda()
                     }

            label = data[3].cuda()

            outputs = model(names, descriptions)

            _, predicted = torch.max(outputs, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

            # each class accuracy
            c = (predicted == label).squeeze()
            for i in range(len(label)):
                labels = label[i]
                class_correct[labels] += c[i].item()
                class_total[labels] += 1
    if per_class:
        print('each class accuracy of: ' )
        for i in range(class_num):
            #print('Accuracy of ======' ,100 * class_correct[i] / class_total[i])
            print(100 * class_correct[i] / class_total[i])

        print('total class_total: ')
        for i in range(class_num):
            print(class_total[i])

    return 100 * correct / total


def eval_top5(model, dataLoader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataLoader:
            descriptions = {'input_ids': data[0].cuda(),
                            'token_type_ids': data[1].cuda(),
                            'attention_mask': data[2].cuda()
                            }

            names = {'input_ids': data[4].cuda(),
                     'token_type_ids': data[5].cuda(),
                     'attention_mask': data[6].cuda()
                     }

            label = data[3].cuda()

            outputs = model(names, descriptions)

            maxk = max((1, 5))
            y_resize = label.view(-1, 1)
            _, pred = outputs.topk(maxk, 1, True, True)
            total += label.size(0)
            correct += torch.eq(pred, y_resize).sum().float().item()

    return 100 * correct / total


class LstmMlp(torch.nn.Module):
    def __init__(self):
        super(LstmMlp, self).__init__()
        self.linear = nn.Linear(in_features=1024, out_features=category_num)
        self.bert_description = BertModel.from_pretrained('bert-base-uncased')
        self.lstm = nn.LSTM(input_size=768, hidden_size=512, num_layers=1, batch_first=True,
                            bidirectional=True)

    def forward(self, names, descriptions):
        self.lstm.flatten_parameters()
        description_bert_output = self.bert_description(**descriptions)

        description_bert_feature = description_bert_output[0]

        # LSTM
        packed_output, (hidden, cell) = self.lstm(description_bert_feature)
        hidden = torch.cat((cell[0, :, :], cell[1, :, :]), dim=1)
        # hidden = torch.cat((hidden[0, :, :], hidden[1, :, :]), dim=1)

        output = self.linear(hidden)
        return output


train_data = load_data_train()
test_data = load_data_test()

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

model = LstmMlp()
model = torch.nn.DataParallel(model)
model = model.cuda()
model.train()

pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
pytorch_total_params_all = sum(p.numel() for p in model.parameters())
print("Trainable: ", pytorch_total_params_trainable)
print("All: ", pytorch_total_params_all)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
# optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

epoch_list = []
acc1_list = []
acc5_list = []

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
        label = data[3].cuda()

        outputs = model(names, descriptions)

        # outputs = model()

        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

    # pdb.set_trace()
    top_1_acc = eval_top1(model, test_dataloader, category_num)
    top_5_acc = eval_top5(model, test_dataloader)
    epoch_list.append(epoch+1)
    acc1_list.append(top_1_acc)
    acc5_list.append(top_5_acc)

    print("=======>top1 acc on the test:{}".format(str(top_1_acc)))
    print("=======>top5 acc on the test:{}".format(str(top_5_acc)))

    acc_list = pd.DataFrame(
        {
            'epoch': epoch_list,
            'Top1': acc1_list,
            'Top5': acc5_list
         }
    )

    acc_list.to_csv('/home/aa7514/PycharmProjects/servenet_extended/files/t2_lstm_linear.csv')

