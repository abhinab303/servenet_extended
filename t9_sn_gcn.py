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
from dgl.nn import GraphConv
import torch.nn.functional as F
import pickle

ip_file_dir = "/home/aa7514/PycharmProjects/servenet_extended/data/"
CLASS_NUM = category_num = 50
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

    total_targets = integer_encoded

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
    test_df = pd.read_csv(test_file)
    # df = pd.concat([train_df, test_df], axis=0)
    df = test_df
    df.reset_index(inplace=True, drop=True)
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


def load_all_data():
    test_file = f"{ip_file_dir}{category_num}/test.csv"
    train_file = f"{ip_file_dir}{category_num}/train.csv"

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    df = pd.concat([train_df, test_df], axis=0)
    # df = test_df
    df.reset_index(inplace=True, drop=True)
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

    test_data = TensorDataset(*desc_list, total_targets, *name_list, torch.tensor(df.index.values))

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


class MultiHead(nn.Module):
    def __init__(self,
                 feat_dim=1024,
                 #  num_classes=250,
                 num_classes=50,
                 use_effect=True,
                 num_head=2,  # 2, 4
                 tau=16.0,  # 16, 32
                 alpha=0,  # 0, 1, 1.5, 3
                 gamma=0.03125):
        super(MultiHead, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(num_classes, feat_dim), requires_grad=True)
        self.scale = tau / num_head
        self.norm_scale = gamma
        self.alpha = alpha
        self.num_head = num_head
        self.head_dim = feat_dim // num_head
        self.use_effect = use_effect

        self.MU = 1.0 - (1 - 0.9) * 0.02

        self.causal_embed = nn.Parameter(torch.FloatTensor(1, feat_dim).fill_(1e-10), requires_grad=False)

        self.reset_parameters(self.weight)

    def reset_parameters(self, weight):
        nn.init.normal_(weight, 0, 0.01)

    def get_cos_sin(self, x, y):
        cos_val = (x * y).sum(-1, keepdim=True) / torch.norm(x, 2, 1, keepdim=True) / torch.norm(y, 2, 1, keepdim=True)
        sin_val = (1 - cos_val * cos_val).sqrt()
        return cos_val, sin_val

    def multi_head_call(self, func, x, weight=None):
        assert len(x.shape) == 2
        x_list = torch.split(x, self.head_dim, dim=1)
        if weight:
            y_list = [func(item, weight) for item in x_list]
        else:
            y_list = [func(item) for item in x_list]
        assert len(x_list) == self.num_head
        assert len(y_list) == self.num_head
        return torch.cat(y_list, dim=1)

    def l2_norm(self, x):
        normed_x = x / (torch.norm(x, 2, 1, keepdim=True) + 1e-8)
        return normed_x

    def causal_norm(self, x, weight):
        norm = torch.norm(x, 2, 1, keepdim=True)
        normed_x = x / (norm + weight)
        return normed_x

    def init_weights(self):
        self.reset_parameters(self.weight)

    def forward(self, x):
        normed_w = self.multi_head_call(self.causal_norm, self.weight, weight=self.norm_scale)
        normed_x = self.multi_head_call(self.l2_norm, x)
        y = torch.mm(normed_x * self.scale, normed_w.t())

        return y


class WeightedSum(nn.Module):
    def __init__(self):
        super(WeightedSum, self).__init__()
        self.w1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)

    def forward(self, input1, input2):
        return input1 * self.w1 + input2 * self.w2


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

        self.weight_sum = WeightedSum()
        self.multi_head = MultiHead(num_classes=CLASS_NUM)

    def forward(self, names, descriptions):
        self.lstm.flatten_parameters()
        name_bert_output = self.bert_name(**names)
        name_features = self.name_liner(name_bert_output[1])
        name_features = self.name_ReLU(name_features)
        name_features = self.name_Dropout(name_features)

        description_bert_output = self.bert_description(**descriptions)
        description_bert_feature = description_bert_output[0]

        packed_output, (hidden, cell) = self.lstm(description_bert_feature)
        hidden = torch.cat((cell[0, :, :], cell[1, :, :]), dim=1)

        all_features = self.weight_sum(name_features, hidden)
        output = self.multi_head(all_features)
        return all_features


sn_model = torch.load("/home/aa7514/PycharmProjects/servenet_extended/files/snlt_best")
# model = torch.nn.DataParallel(model)
for param in sn_model.parameters():
    param.requires_grad = False
sn_model.eval()

# loop through all the datasets, generate the embedding and save it in a file.
feature_list = []

all_data = load_all_data()
all_data_loader = DataLoader(all_data, batch_size=1024)

for data in tqdm(all_data_loader):
    descriptions = {'input_ids': data[0].cuda(),
                    'token_type_ids': data[1].cuda(),
                    'attention_mask': data[2].cuda()
                    }

    names = {'input_ids': data[4].cuda(),
             'token_type_ids': data[5].cuda(),
             'attention_mask': data[6].cuda()
             }
    label = data[3].cuda()

    outputs = sn_model(names, descriptions)
    feature_list.extend(outputs.cpu().numpy())
    # pdb.set_trace()

pdb.set_trace()


class GCN(torch.nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConv(nfeat, nhid1, norm='both', weight=True, bias=True)
        self.gc2 = GraphConv(nhid1, nhid2, norm='both', weight=True, bias=True)
        # self.gc2 = GraphConv(nhid1, nclass, norm='both', weight=True, bias=True)
        self.dropout = dropout
        self.final_liner = nn.Linear(in_features=nhid2, out_features=nclass)
        self.multiHead = MultiHead(num_classes=CLASS_NUM)

    def forward(self, g, x):
        x = F.relu(self.gc1(g, x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(g, x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(g, x)
        x = self.final_liner(x)
        # return F.log_softmax(x, dim=1)
        # return F.log_softmax(output, dim=1)
        return x

test_path = "/home/aa7514/PycharmProjects/servenet_extended/data/50/test.csv"
train_path = "/home/aa7514/PycharmProjects/servenet_extended/data/50/train.csv"
graph_path = "/home/aa7514/PycharmProjects/servenet_extended/files/graph.pickle"
feature_path = "/home/aa7514/PycharmProjects/servenet_extended/files/feature_matrix.pickle"

with open(graph_path, "rb") as f:
    graph = pickle.load(f)

with open(feature_path, "rb") as f:
    feature_matrix = pickle.load(f)



train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
train_df["ServiceDescription"] = train_df["ServiceDescription"].str.replace(r'[^A-Za-z .]+', '', regex=True)
test_df["ServiceDescription"] = test_df["ServiceDescription"].str.replace(r'[^A-Za-z .]+', '', regex=True)

api_dataframe = pd.concat([train_df, test_df], axis=0)
api_dataframe.reset_index(inplace=True, drop=True)

Train_C = api_dataframe.iloc[0:len(train_df)]
Test_C = api_dataframe.iloc[len(train_df):]

training_data = Train_C
testing_data = Test_C
TrainIndex = training_data.index.values.tolist()
TestIndex = testing_data.index.values.tolist()

train_data = load_data_train()
test_data = load_data_test()

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

graph_copy = graph.copy()

f2 = feature_matrix[len(feature_list):]
f1 = np.array(feature_list)
dd = feature_list[0].shape[0] - feature_matrix.shape[1]
f2p = np.pad(f2, ((0, 0), (0, dd)), 'constant')
new_emb = np.concatenate((f1, f2p), axis=0)

# features = sp.csr_matrix(feature_matrix)
features = sp.csr_matrix(new_emb)

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
    loss_train = criterion(output[idx_train], labels[idx_train])
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
    loss_val = criterion(output[idx_val], labels[idx_val])
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
    loss_test = criterion(output[idx_test], labels[idx_test])
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
torch.save(model, "/home/aa7514/PycharmProjects/servenet_extended/files/gcn_sn_ip")
pass