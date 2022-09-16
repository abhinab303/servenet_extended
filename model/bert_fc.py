import torch
import torch.nn as nn
from transformers import BertModel


class ServeNet(torch.nn.Module):
    def __init__(self, hiddenSize, CLASS_NUM):
        super(ServeNet, self).__init__()
        self.hiddenSize = hiddenSize
        self.bert_description = BertModel.from_pretrained('bert-base-uncased')
        self.final_linear = nn.Linear(in_features=768, out_features=CLASS_NUM)

    # def forward(self, names, descriptions):
    def forward(self, input):

        description_bert_output = self.bert_description(**input)
        hidden = description_bert_output[1]

        output = self.final_linear(hidden)

        return output