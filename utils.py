import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

from transformers import BertTokenizer

import torch

from torch.utils.data import TensorDataset


ip_file_dir = "/home/aa7514/PycharmProjects/servenet_extended/data/"


def encode_onehot(labels):
    # classes = set(labels)
    classes = classes = sorted(list(set(labels)), key=str.lower)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def load_data_train(catagory_num):
    train_file = f"{ip_file_dir}{catagory_num}/train.csv"
    df = pd.read_csv(train_file)
    values = np.array(df.ServiceClassification)
    label_encoder = LabelEncoder()
    # integer_encoded = label_encoder.fit_transform(values)
    integer_encoded = encode_onehot(df.ServiceClassification)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # descriptions
    descriptions = df["ServiceDescription"].tolist()
    inputs = tokenizer(descriptions, return_tensors="pt",
                       #  model_max_length=100,
                       max_length=100,
                       padding=True,
                       truncation=True)
    total_targets = torch.tensor(integer_encoded)

    input_list = []
    for key, value in inputs.items():
        input_list.append(torch.tensor(value))

    train_data = TensorDataset(*input_list, total_targets)

    return train_data


def load_data_test(catagory_num):
    test_file = f"{ip_file_dir}{catagory_num}/test.csv"
    df = pd.read_csv(test_file)
    values = np.array(df.ServiceClassification)
    label_encoder = LabelEncoder()
    # integer_encoded = label_encoder.fit_transform(values)
    integer_encoded = encode_onehot(df.ServiceClassification)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # descriptions
    descriptions = df["ServiceDescription"].tolist()
    inputs = tokenizer(descriptions, return_tensors="pt",
                       #  model_max_length=100,
                       max_length=100,
                       padding=True,
                       truncation=True)
    total_targets = torch.tensor(integer_encoded)

    input_list = []
    for key, value in inputs.items():
        input_list.append(torch.tensor(value))

    test_data = TensorDataset(*input_list, total_targets)

    return test_data


def load_data_train_names(catagory_num, max_len=100):
    train_file = f"{ip_file_dir}{catagory_num}/train.csv"
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


def load_data_test_names(catagory_num, max_len_desc=100, max_len_name=10,):
    test_file = f"{ip_file_dir}{catagory_num}/test.csv"
    train_file = f"{ip_file_dir}{catagory_num}/train.csv"
    train_df = pd.read_csv(train_file)
    df = pd.read_csv(test_file)
    values = np.array(df.ServiceClassification)
    label_encoder = LabelEncoder()
    # integer_encoded = label_encoder.fit_transform(values)
    # integer_encoded = encode_onehot(df.ServiceClassification).transpose(1, 0)
    integer_encoded = integer_encoded = torch.LongTensor(np.where(encode_onehot(df.ServiceClassification))[1])
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # descriptions
    descriptions = df["ServiceDescription"].tolist()
    desc_tokens = tokenizer(descriptions, return_tensors="pt",
                            max_length=max_len_desc,
                            padding=True,
                            truncation=True)

    # names
    names = df["ServiceName"].tolist()
    name_tokens = tokenizer(names, return_tensors="pt",
                            #  model_max_length=100,
                            max_length=max_len_name,
                            padding=True,
                            truncation=True)

    total_targets = torch.tensor(integer_encoded)

    desc_list = []
    for key, value in desc_tokens.items():
        desc_list.append(torch.tensor(value))

    name_list = []
    for key, value in name_tokens.items():
        name_list.append(torch.tensor(value))

    test_data = TensorDataset(*desc_list, total_targets, *name_list, torch.tensor(df.index.values + len(train_df)))

    return test_data


def evaluteTop1(model, dataLoader, class_num=50, per_class=False):
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(class_num))
    class_total = list(0. for i in range(class_num))
    with torch.no_grad():
        for data in dataLoader:
            input = {'input_ids': data[0].cuda(),
                     'token_type_ids': data[1].cuda(),
                     'attention_mask': data[2].cuda()
                     }
            label = data[3].cuda()

            outputs = model(input)

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


def evaluteTop5(model, dataLoader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataLoader:

            input = {'input_ids': data[0].cuda(),
                     'token_type_ids': data[1].cuda(),
                     'attention_mask': data[2].cuda()
                     }
            label = data[3].cuda()

            outputs = model(input)

            maxk = max((1, 5))
            y_resize = label.view(-1, 1)
            _, pred = outputs.topk(maxk, 1, True, True)
            total += label.size(0)
            correct += torch.eq(pred, y_resize).sum().float().item()

    return 100 * correct / total


def evaluteTop1_names(model, dataLoader, class_num=50, per_class=False):
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
            indices = data[7].cuda()
            label = data[3].cuda()

            outputs = model(names, descriptions, indices)

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


def evaluteTop5_names(model, dataLoader):
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
            indices = data[7].cuda()
            label = data[3].cuda()

            outputs = model(names, descriptions, indices)

            maxk = max((1, 5))
            y_resize = label.view(-1, 1)
            _, pred = outputs.topk(maxk, 1, True, True)
            total += label.size(0)
            correct += torch.eq(pred, y_resize).sum().float().item()

    return 100 * correct / total
