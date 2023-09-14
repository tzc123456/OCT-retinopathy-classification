# -*-coding:utf-8-*-
# -*-coding:utf-8-*-

import os
from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
import torch
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import models
import backbones
import lsoftmax
# 超参数的设置

val_batch_size = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 看看使用什么训练
print(torch.cuda.is_available())

# 超参数的设置
data_transform = {

    'val':transforms.Compose([

        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),]
    )
}

A_set = datasets.ImageFolder(

    root = r'/home/xiaoxie/data/tzc/domain_adaptation/DA/data/A',
    transform= data_transform['val']
)


A_dataloader = DataLoader(
    A_set,
    batch_size= val_batch_size,
    shuffle= False,
    num_workers=4,
)


B_set = datasets.ImageFolder(

    root = r'/home/xiaoxie/data/tzc/domain_adaptation/DA/data/B',
    transform= data_transform['val']
)


B_dataloader = DataLoader(
    B_set,
    batch_size= val_batch_size,
    shuffle= False,
    num_workers=4,
)

print(len(A_set))
print(len(B_set))

def plot_embedding_2D(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(int(label[i])),
                 color=plt.cm.Set1(int(label[i])),
                 fontdict={'weight': 'bold', 'size': 9})
        # colors = np.random.rand(1)
        # plt.scatter(data[i,0],data[i,1],c = 0.5,s = 20)
    plt.xticks([])
    plt.yticks([])

    l = ['A', 'B']
    color = [plt.cm.Set1(0), plt.cm.Set1(1)]
    patches = [mpatches.Patch(color=color[i], label='{:s}'.format(l[i])) for i in range(2)]
    plt.legend(handles=patches)


    plt.title(title)
    plt.savefig('./Proposed.jpg', dpi=500)
    return fig




# 开始训练网络
def main():
    DEVICE = torch.device('cuda')
    model = models.TransferNet(3,device= DEVICE,base_net="resnet50", transfer_loss='lmmd')
    model.load_state_dict(torch.load(
        r'/home/xiaoxie/data/tzc/domain_adaptation/DA/ATOB/Proposed/Proposed.pth'))
    model = model.cpu()


    data_A = np.zeros((1,256))
    data_B = np.ones((1,256))

    # la = np.zeros((31712,))
    for epoch in range(1):
        model.eval()
        with torch.no_grad():
            for step,val_data in enumerate(A_dataloader):
                val_images,val_labels = val_data
                output = model.base_network(val_images)
                output = model.bottleneck_layer(output)
                output = output.numpy()
                output = np.asarray(output)
                data_A = np.concatenate((data_A, output), axis=0)

            for step,val_data in enumerate(B_dataloader):
                val_images,val_labels = val_data
                output = model.base_network(val_images)
                output = model.bottleneck_layer(output)
                output = output.numpy()
                output = np.asarray(output)
                data_B = np.concatenate((data_B, output), axis=0)

            print('Begining......')
            # data_A = data_A[:50]
            # data_B = data_B[:50]
            data_ab = np.concatenate((data_A,data_B),axis=0)

            tsne_2D = TSNE(n_components=2, init='pca', random_state=501)
            result_2D = tsne_2D.fit_transform(data_ab)
            # la = np.zeros((2801,))
            # lb = np.ones((2801,))
            la = np.zeros((len(A_set)+1,))
            lb = np.ones((len(B_set)+1,))



            # la = np.zeros((193,))
            # lb = np.ones((193,))
            lab = np.concatenate((la,lb))

            print('Finished......')

            fig1 = plot_embedding_2D(result_2D, lab, 'Proposed')
            plt.show()



if __name__ == '__main__':

    main()


