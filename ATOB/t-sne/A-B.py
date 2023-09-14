# -*-coding:utf-8-*-
# draw distribution between A and B
import os
import numpy as np
from time import time
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
import torch
from torchvision import transforms,datasets
import numpy as np
from torch.utils.data import DataLoader

def get_A():
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_set = datasets.ImageFolder(

        root = r'/home/xiaoxie/data/tzc/domain_adaptation/DA/data/A',
        transform= transform
    )
    print("the number of A is : %d" %len(train_set))
    A_data = []
    A_label = []
    for i in train_set:
        A_data.append(i[0].numpy())  # 3,224,224
        A_label.append(i[1])

    return np.asarray(A_data), np.asarray(A_label)

def get_B():

    pre_process = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    )

    train_set = datasets.ImageFolder(
        root = r'/home/xiaoxie/data/tzc/domain_adaptation/DA/data/B',
        transform=pre_process
    )

    print("the number of B is : %d" %len(train_set))
    B_data = []
    B_label = []
    for i in train_set:


        B_data.append(i[0].numpy())
        B_label.append(i[1])

    return np.asarray(B_data),np.asarray(B_label)



def plot_embedding_2D(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])

    l = ['A', 'B']
    color = [plt.cm.Set1(0), plt.cm.Set1(1)]
    patches = [mpatches.Patch(color=color[i], label='{:s}'.format(l[i])) for i in range(2)]
    plt.legend(handles=patches)

    plt.title(title)
    plt.savefig('./distributionA-B.jpg', dpi=500)
    return fig

def main():

    A_data, A_label = get_A()
    B_data ,B_label = get_B()
    # resize --> er wei
    new_A_data = A_data.reshape(A_data.shape[0],-1)
    new_B_data = B_data.reshape(B_data.shape[0],-1)

    AA = new_A_data
    BB = new_B_data
    data = np.concatenate((AA,BB),axis=0)

    aa = np.zeros_like(A_label)
    bb = np.ones_like(B_label)
    new_label = np.concatenate((aa, bb))

    print('Begining......')
    tsne_2D = TSNE(n_components=2, init='pca', random_state=501)
    result_2D = tsne_2D.fit_transform(data)
    print('Finished......')

    fig1 = plot_embedding_2D(result_2D, new_label,'t-sne')
    plt.show()

if __name__ == '__main__':
    torch.manual_seed(10)
    main()