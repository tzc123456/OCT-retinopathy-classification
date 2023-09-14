"""
此代码是微调
"""
import random
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import numpy as np
import torch.optim as optim
import argparse
from tqdm import tqdm
from model import DANNmodel
from torch.utils.tensorboard import SummaryWriter


def main():
    # 设置随机数种子
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(10)
    torch.cuda.manual_seed(10)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 判断是使用GPU还是CPU？
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    # 数据预处理
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                     ]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                   ])}
    # 读取训练数据，10%数据做微调
    train_dataset = datasets.ImageFolder(root="/home/xiaoxie/data/tzc/domain_adaptation/dataB/OCT/val",
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.bs, shuffle=True,
                                               num_workers=args.num_works)
    # 读取验证数据
    validate_dataset = datasets.ImageFolder(root="/home/xiaoxie/data/tzc/domain_adaptation/dataB/OCT/train",
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=args.vs, shuffle=False,
                                                  num_workers=args.num_works)
    # 输出训练和验证的数目
    print("using {} images for training, {} images for validation.".format(train_num,val_num))
    # 定义模型
    net = DANNmodel()
    # net.load_state_dict(torch.load("/home/xiaoxie/data/tzc/domain_adaptation/dataA/DSAN_A.pth"))
    net.to(device)


    # 定义损失函数
    loss_function = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    # 定义优化器
    optimizer = optim.SGD(net.parameters(), momentum=args.momentum, lr=args.lr)
    best_acc = 0.0
    # 保存路劲
    save_path = './ImageNet.pth'
    # 绘制损失曲线
    writer = SummaryWriter()
    for epoch in range(args.epochs):
        # 训练模型
        net.train()
        # 设定初始值
        running_loss = 0.0
        train_acc = 0.0
        # 记时器
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            # 网络输出
            outputs = net(images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            # 统计计算正确的数量
            train_acc += torch.eq(predict_y, labels.to(device)).sum().item()
            # 计算硬Loss
            loss = loss_function(outputs, labels.to(device))
            # 反向传播
            loss.backward()
            # 更新网络参数
            optimizer.step()
            # 统计总的loss
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,args.epochs,loss)
        # 统计一次epoch里面的acc
        train_accurate = train_acc / train_num
        writer.add_scalar("Train Loss/Epoch", running_loss, epoch)
        writer.add_scalar("Train Acc/Epoch", train_accurate, epoch)
        # 验证模型，此时不更新网络的参数
        net.eval()
        val_acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                val_acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
        val_accurate = val_acc / val_num
        writer.add_scalar("Val Acc/Epoch", val_accurate, epoch)
        print('val_accuracy: {}'.format(val_accurate))
        # 保存模型在验证集上准确率更高的模型参数
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
    print('Finished Training')
    print("best val acc:{}".format(best_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-tb","--train_batch_size",
        type=int,
        default=32
    )
    parser.add_argument(
        "-vs", "--val_batch_size",
        type=int,
        default=4
    )
    parser.add_argument(
        "-vs", "--val_batch_size",
        type=int,
        default=4
    )
    parser.add_argument(
        "--num_works",
        type=int,
        default=4
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9
    )
    args = parser.parse_args()
    main(args)
