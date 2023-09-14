import os
import sys
import json
import random
import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils, models
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from model import  DANNmodel
from torch.utils.tensorboard import SummaryWriter

# from test import test2

def main():
    # seed setting
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(10)
    torch.cuda.manual_seed(10)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 判断是使用GPU还是CPU？
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    # 加载数据
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])}


    train_dataset = datasets.ImageFolder(root="/home/xiaoxie/data/tzc/domain_adaptation/dataC/OCT/val",
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    batch_size = 32

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=4)

    validate_dataset = datasets.ImageFolder(root="/home/xiaoxie/data/tzc/domain_adaptation/dataC/OCT/train",
                                            transform=data_transform["val"])

    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=False,
                                                  num_workers=4)

    # test_dataset = datasets.ImageFolder(root=  r"/home/xiaoxie/data/tzc/domain_adaptation/dataA/OCT/OCTA",
    #                                         transform=data_transform["val"])
    # test_num = len(test_dataset)
    # test_loader = torch.utils.data.DataLoader(test_dataset,
    #                                               batch_size=4, shuffle=False,
    #                                               num_workers=4)

    print("using {} images for training, {} images for validation, {} images for test.".format(train_num,
                                                                           val_num, 0 ))

    net = DANNmodel()
    # net.load_state_dict(torch.load("/home/xiaoxie/data/tzc/domain_adaptation/dataA/DSAN_A.pth"))

    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    # pata = list(net.parameters())
    #optimizer = optim.Adam(net.parameters(), lr=1e-4)
    optimizer = optim.SGD(net.parameters(), momentum=0.9,lr=1e-4)

    epochs = 50

    best_acc = 0.0
    save_path = './random.pth'
    # 绘制损失曲线
    writer = SummaryWriter()

    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_acc = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            train_acc += torch.eq(predict_y, labels.to(device)).sum().item()
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        #     # writer.add_scalar("Train Loss/Step", loss, (epoch * train_steps)+step)
        #
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        train_accurate = train_acc / train_num
        writer.add_scalar("Train Loss/Epoch", running_loss, epoch)
        writer.add_scalar("Train Acc/Epoch", train_accurate, epoch)
        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        writer.add_scalar("Val Acc/Epoch", val_accurate, epoch)
        # print(' train_loss: %.3f  val_accuracy: %.3f' %
        #       (running_loss / train_steps, val_accurate))
        print('   val_accuracy: %.3f' %( val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

        #net.load_state_dict(torch.load('./DSAN_A.pth'))

    #     acc_test = test2(net, test_loader)
    #     print('  test_accuracy: %.3f' %
    #           (  acc_test))
    #     writer.add_scalars(" acc/epoch", {"val_acc": val_accurate, "test_acc": acc_test}, epoch)
    # writer.close()
    print('Finished Training')
    print("best val acc:{}".format(best_acc))


if __name__ == '__main__':
    #torch.manual_seed(10)
    main()
