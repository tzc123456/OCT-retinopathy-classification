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

from model import AlexNet, DANNmodel
from torch.utils.tensorboard import SummaryWriter

from test import test2

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
    data_root = r"/home/xiaoxie/data/tzc/domain_adaptation/dataB"
    image_path = os.path.join(data_root, "OCT")
    #data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    #image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=2)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    #nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    #print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=4)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=False,
                                                  num_workers=4)

    test_dataset = datasets.ImageFolder(root=  r"/home/xiaoxie/data/tzc/domain_adaptation/dataA/OCT/OCTA",
                                            transform=data_transform["val"])
    test_num = len(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=4, shuffle=False,
                                                  num_workers=4)

    print("using {} images for training, {} images for validation, {} images for test.".format(train_num,
                                                                           val_num, test_num ))
    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()
    #
    # def imshow(img):
    #     img = img / 2 + 0.5  # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #
    # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(test_image))

    #net = AlexNet(num_classes=5, init_weights=True)
    #net = LeNet()
    #net = models.resnet18(pretrained=True)
    #net.fc = nn.Linear(512,5)
    #print(net)
    net = DANNmodel()
    #net.load_state_dict(torch.load("/home/xiaoxie/data/tzc/domain_adaptation/dataA/DSAN_A.pth"))
    #print(net)

    #net.classifier[6] = nn.Linear(4096,5)
    #print(net)

    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    # pata = list(net.parameters())
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    epochs = 20
    save_path = './DSAN_B.pth'
    best_acc = 0.0
    train_steps = len(train_loader)
    # 绘制损失曲线
    writer = SummaryWriter()

    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            writer.add_scalar("Train Loss/Step", loss, (epoch * train_steps)+step)

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        writer.add_scalar("Train Loss/Step", running_loss, epoch)
        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        writer.add_scalar("val acc/epoch", val_accurate, epoch)
        # print(' train_loss: %.3f  val_accuracy: %.3f' %
        #       (running_loss / train_steps, val_accurate))
        print('   val_accuracy: %.3f' %( val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

        #net.load_state_dict(torch.load('./DSAN_A.pth'))
        acc_test = test2(net, test_loader)
        print('  test_accuracy: %.3f' %
              (  acc_test))
        writer.add_scalars(" acc/epoch", {"val_acc":val_accurate,"test_acc":acc_test}, epoch)
    writer.close()
    print('Finished Training')


if __name__ == '__main__':
    #torch.manual_seed(10)
    main()
