import os
from tqdm import tqdm

from torch.utils.data import DataLoader
import torch
from torch.optim import lr_scheduler

from utils import load_data_train_names, load_data_test_names, eval_top1_sn, eval_top5_sn
from model.servenetlt import ServeNet

epochs = 40
SEED = 123
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
EPSILON = 1e-8
BATCH_SIZE = 64
CLASS_NUM = 50
cat_num = "50"

des_max_length = 110  # 110#160#200
name_max_length = 10

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

if __name__ == "__main__":
    train_data = load_data_train_names(CLASS_NUM)
    test_data = load_data_test_names(CLASS_NUM)

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

    model = ServeNet(768, CLASS_NUM)
    # model.bert_description.requires_grad_(False)
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

        print("=======>top1 acc on the test:{}".format(str(eval_top1_sn(model, test_dataloader, CLASS_NUM))))
        print("=======>top5 acc on the test:{}".format(str(eval_top5_sn(model, test_dataloader))))

    # print("=======>top1 acc on the test:{}".format(str(eval_top1_sn(model, test_dataloader, CLASS_NUM, True))))

    torch.save(model, "/home/aa7514/PycharmProjects/servenet_extended/files/snlt")
