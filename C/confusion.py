import os
import json
import sys
import torch
from PIL import Image
from torchvision import transforms, models, datasets
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import DANNmodel


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 4

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    data_root = r"/home/xiaoxie/data/tzc/domain_adaptation/dataC/OCT/OCTC"
    test_dataset = datasets.ImageFolder(root=data_root, transform= data_transform)
    test_num = len(test_dataset)
    print(test_num)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = 0
    )
    # create model
    # model = AlexNet(num_classes=5).cpu()
    # model = models.resnet18(pretrained=True)
    # model.fc = torch.nn.Linear(512,5)
    #model.to(device)
    model = DANNmodel()
    model.to(device)

    # load model weights
    #weights_path = "./DSAN_C.pth"
    # weights_path = "/home/xiaoxie/data/tzc/domain_adaptation/DA/ATOC/Proposed/ProposedAC.pth"
    # assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    # model.load_state_dict(torch.load(weights_path))

    model.eval()
    with torch.no_grad():
        acc = 0.0
        predict = []
        true = []
        test_bar = tqdm(test_loader, file=sys.stdout)
        for test_data in test_bar:
            test_images, test_labels = test_data
            outputs = model(test_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, test_labels).sum().item()
            predict.extend(list(predict_y.numpy()))
            true.extend(list(test_labels.numpy()))
        test_acc = 100. * acc / test_num
        print(test_acc)

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
             "size": 30}
    font2 = {"family": "Times New Roman",
             "weight": "normal",
             "size": 20}
    plt.xlabel('predict', font1)
    plt.ylabel('true', font1)
    for first_index in range(len(confusion)):
        for second_index in range(len(confusion[first_index])):
            plt.text(first_index, second_index, confusion[first_index][second_index], font2)

    plt.show()


if __name__ == '__main__':
    main()
