# -*-coding:utf-8-*-
import torch
import sys
from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#torch.manual_seed(10)
def test2(model, target_test_loader):
    model.eval()
    correct = 0
    #criterion = torch.nn.CrossEntropyLoss()
    len_target_dataset = len(target_test_loader.dataset)
    with torch.no_grad():
        test_bar = tqdm(target_test_loader, file=sys.stdout)
        for data, target in test_bar:
            data, target = data.to(device), target.to(device)
            s_output = model(data)
            #loss = criterion(s_output, target)
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)
    acc = correct.double() / len(target_test_loader.dataset)
    return acc