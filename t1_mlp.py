import torch
import torch.nn as nn
from transformers import BertModel

class Mlp(torch.nn.Module):
    def __init__(self, hiddenSize, CLASS_NUM=50):
        super(Mlp, self).__init__()
        self.linear = nn.Linear(in_features=768, out_features=CLASS_NUM)
        self.bert_description = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, names, descriptions):
        self.lstm.flatten_parameters()
        description_bert_output = self.bert_description(**descriptions)

        description_bert_feature = description_bert_output[1]
        output = self.linear(description_bert_feature)
        return output



