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
import torch.nn.functional as F

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


class MultiHead(nn.Module):
    def __init__(self,
                 feat_dim=768,
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


def one_hot_embedding(labels, num_classes=50):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes)
    return y[labels]


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
    # kl_div = 1 * kl_divergence(kl_alpha, num_classes, device=device)
    # return A + kl_div
    return A


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


def loglikelihood_loss(y, alpha, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div
    # return loglikelihood


def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device)
    )
    return loss


class BertMultiHead(torch.nn.Module):
    def __init__(self):
        super(BertMultiHead, self).__init__()
        self.bert_description = BertModel.from_pretrained('bert-base-uncased')
        self.multiHead = MultiHead(num_classes=category_num)

    def forward(self, names, descriptions):
        description_bert_output = self.bert_description(**descriptions)

        description_bert_feature = description_bert_output[1]

        output = self.multiHead(description_bert_feature)
        return output


train_data = load_data_train()
test_data = load_data_test()

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

model = BertMultiHead()
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

a_param = 10
loss_func = edl_log_loss
# loss_func = edl_mse_loss

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

        # loss = criterion(outputs, label)
        # pdb.set_trace()
        loss = loss_func(outputs, one_hot_embedding(label, 50), epoch + 1, 50, a_param)
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

    acc_list.to_csv('/home/aa7514/PycharmProjects/servenet_extended/files/t11_bert_multi_head_evd.csv')

