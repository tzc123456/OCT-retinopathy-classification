import os
import json
import sys
import torch
from PIL import Image
from torchvision import transforms, models, datasets
import matplotlib.pyplot as plt
from tqdm import tqdm
#from model import AlexNet, DANNmodel
import torchmetrics
import models
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test1_acc = torchmetrics.Accuracy()
    test1_recall = torchmetrics.Recall(average="macro", num_classes=3)
    test1_precision = torchmetrics.Precision(average="macro", num_classes=3)
    test1_auc = torchmetrics.AUROC(average="macro", num_classes=3)

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 128

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    data_root = r"/home/xiaoxie/data/tzc/domain_adaptation/DA/data/C"
    test_dataset = datasets.ImageFolder(root=data_root, transform= data_transform)
    test_num = len(test_dataset)
    print(test_num)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = 4
    )
    # create model
    #model = AlexNet(num_classes=5).cpu()
    # model = models.resnet18(pretrained=True)
    # model.fc = torch.nn.Linear(512,5)
    #model.to(device)

    DAmodel = models.TransferNet(3, device=device, base_net="resnet50", transfer_loss='lmmd')

    # load model weights
    weights_path = "./DSANbc.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    DAmodel.load_state_dict(torch.load(weights_path))
    model = DAmodel.to(device)

    model.eval()
    with torch.no_grad():
        acc = 0.0
        predict = []
        true = []
        test_bar = tqdm(test_loader, file=sys.stdout)
        for test_data in test_bar:
            test_images, test_labels = test_data

            outputs = model.predict(test_images.to(device))
            outputs = outputs.cpu()
            predict_y = torch.max(outputs, dim=1)[1]
            predict_y = predict_y.cpu()
            acc += torch.eq(predict_y, test_labels).sum().item()
            predict.extend(list(predict_y.numpy()))
            true.extend(list(test_labels.numpy()))

            test1_acc(predict_y, test_labels)
            test1_auc.update(outputs, test_labels)
            test1_recall(predict_y, test_labels)
            test1_precision(predict_y, test_labels)

        test_acc = 100. * acc / test_num
        print(test_acc)

        total_acc = test1_acc.compute()
        total_recall = test1_recall.compute()
        total_precision = test1_precision.compute()
        total_auc = test1_auc.compute()
        print(
              f"torch metrics acc: {(100 * total_acc):>0.1f}%\n")
        print("recall of every test dataset class: ", total_recall)
        print("precision of every test dataset class: ", total_precision)
        print("auc:", total_auc.item())

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import recall_score
    import matplotlib.pyplot as plt

    # guess = [1, 0, 1, 2, 1, 0, 1, 0, 1, 0]
    # fact = [0, 1, 0, 1, 2, 1, 0, 1, 0, 1]
    classes = list(set(true))
    classes.sort()
    confusion = confusion_matrix(predict, true)
    plt.imshow(confusion, cmap=plt.cm.Blues)
    indices = range(len(confusion))
    plt.xticks(indices, classes)
    plt.yticks(indices, classes)
    plt.colorbar()
    font1 = {"family": "Times New Roman",
             "weight": "normal",
             "size": 20}
    font2 = {"family": "Times New Roman",
             "weight": "normal",
             "size": 20}
    plt.xlabel('Predict', font1)
    plt.ylabel('True', font1)
    for first_index in range(len(confusion)):
        for second_index in range(len(confusion[first_index])):
            plt.text(first_index, second_index, confusion[first_index][second_index], font2,va = "center",ha = "center")
    plt.savefig('./DSAN.jpg', dpi=500)
    plt.show()


if __name__ == '__main__':
    main()
