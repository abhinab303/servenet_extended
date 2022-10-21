import torch
import torch.nn as nn
import pdb


class weighted_sum(nn.Module):
    def __init__(self):
        super(weighted_sum, self).__init__()
        self.w1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)

    def forward(self, input1, input2):
        return input1 * self.w1 + input2 * self.w2
        # return input1 * self.w1 + input2 * (1 - self.w1)
        # w1 = self.w1/(self.w1 + self.w2)
        # w2 = self.w2/(self.w1 + self.w2)
        # return input1 * w1 + input2 * w2


class flex_ws(nn.Module):
    def __init__(self):
        super(flex_ws, self).__init__()
        self.w1 = nn.Parameter(torch.FloatTensor(1, 50), requires_grad=True)
        self.w2 = nn.Parameter(torch.FloatTensor(1, 50), requires_grad=True)

    def forward(self, input1, input2):
        # pdb.set_trace()
        return input1 * self.w1.repeat(input1.shape[0], 1) + input2 * self.w2.repeat(input2.shape[0], 1)


class weighted_sum3(nn.Module):
    def __init__(self):
        super(weighted_sum3, self).__init__()
        self.w1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w3 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.w4 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)

    def forward(self, input1, input2, input3):
        # x = input1 * self.w1 + input2 * self.w2
        # return x * self.w3 + input3 * self.w4
        # return input1 * 0 + input2 * 0 + input3 * 1
        # return input1 * self.w1 + input2 * self.w2 + input3 * 0
        return input1 * self.w1 + input2 * self.w2
        # return input1 * self.w1 + input2 * self.w2 + input3 * self.w3


class MutliHead(nn.Module):

    def __init__(self,
                 feat_dim=1024,
                 #  num_classes=250,
                 num_classes=50,
                 use_effect=True,
                 num_head=2,  # 2, 4
                 tau=16.0,  # 16, 32
                 alpha=0,  # 0, 1, 1.5, 3
                 gamma=0.03125):
        super(MutliHead, self).__init__()

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