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
import random
# 超参数的设置

val_batch_size = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

def plot_embedding_2D(S_data, Slabel,T_data, Tlabel, title):
    Sx_min, Sx_max = np.min(S_data, 0), np.max(S_data, 0)
    Sdata = (S_data - Sx_min) / (Sx_max - Sx_min)
    fig = plt.figure()
    for i in range(Sdata.shape[0]):
        plt.text(Sdata[i, 0], Sdata[i, 1], str(int(Slabel[i])),
                 color=plt.cm.Set1(int(Slabel[i])),
                 fontdict={'weight': 'bold', 'size': 9},
                 )

    Tx_min, Tx_max = np.min(T_data, 0), np.max(T_data, 0)
    Tdata = (T_data - Tx_min) / (Tx_max - Tx_min)
    for i in range(Tdata.shape[0]):
        plt.text(Tdata[i, 0], Tdata[i, 1], str(int(Tlabel[i])),
                 color=plt.cm.Set1(int(Tlabel[i])+3),
                 fontdict={'weight': 'bold', 'size': 9})


    plt.xticks([])
    plt.yticks([])

    l = ['S_AMD', 'S_DME', "S_NORMAL", "T_AMD", "T_DME", "T_NORMAL"]
    color = [plt.cm.Set1(0), plt.cm.Set1(1),plt.cm.Set1(2),plt.cm.Set1(3),plt.cm.Set1(4),plt.cm.Set1(5)]
    patches = [mpatches.Patch(color=color[i], label='{:s}'.format(l[i])) for i in range(6)]
    plt.legend(handles=patches)


    plt.title(title)
    plt.savefig('./DSAN.jpg', dpi=500)
    return fig




# 开始训练网络
def main():
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    DEVICE = torch.device('cuda')
    model = models.TransferNet(3,device= DEVICE,base_net="resnet50", transfer_loss='adv')
    model.load_state_dict(torch.load(
        r'/home/xiaoxie/data/tzc/domain_adaptation/DA/ATOB/DANN/DANNab.pth'))
    model = model.cpu()


    data_A = np.zeros((1,3))
    A_label = np.zeros((1))
    #data_B = np.ones((1,256))
    data_B = np.zeros((1, 3))
    B_label = np.zeros((1))


    # la = np.zeros((31712,))
    for epoch in range(1):
        model.eval()
        with torch.no_grad():
            for step,val_data in enumerate(A_dataloader):
                val_images,val_labels = val_data
                output = model.base_network(val_images)
                output = model.bottleneck_layer(output)
                output = model.classifier_layer(output)
                output = output.numpy()
                output = np.asarray(output)
                val_labels = val_labels.numpy()
                val_labels = np.asarray(val_labels)
                data_A = np.concatenate((data_A, output), axis=0)
                A_label = np.concatenate((A_label, val_labels), axis=0)


            for step,val_data in enumerate(B_dataloader):
                val_images,val_labels = val_data
                output = model.base_network(val_images)
                output = model.bottleneck_layer(output)
                output = model.classifier_layer(output)
                output = output.numpy()
                output = np.asarray(output)
                val_labels = val_labels.numpy()
                val_labels = np.asarray(val_labels)
                data_B = np.concatenate((data_B, output), axis=0)
                B_label = np.concatenate((B_label, val_labels), axis=0)


            print('Begining......')

            data_A = data_A[1:len(A_set),]
            data_B = data_B[1:len(B_set),]
            A_label = A_label[1:]
            B_label = B_label[1:]

            #data_ab = np.concatenate((data_A,data_B),axis=0)

            tsne_2D = TSNE(n_components=2, init='pca', random_state=501) # 501
            result_2D_A = tsne_2D.fit_transform(data_A)
            result_2D_B = tsne_2D.fit_transform(data_B)
            # la = np.zeros((2801,))
            # lb = np.ones((2801,))
            # la = np.zeros((len(A_set)+1,))
            # lb = np.ones((len(B_set)+1,))



            # la = np.zeros((193,))
            # lb = np.ones((193,))
            # lab = np.concatenate((la,lb))

            print('Finished......')

            fig1 = plot_embedding_2D(result_2D_A, A_label, result_2D_B, B_label, 'DSAN')
            plt.show()



if __name__ == '__main__':

    main()


