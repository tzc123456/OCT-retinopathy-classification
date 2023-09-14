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

        root = r'/home/xiaoxie/data/tzc/domain_adaptation/DA/data/B',
        transform= transform
    )
    print("the number of A is : %d" %len(train_set))
    A_data = []
    A_label = []
    for i in train_set:
        A_data.append(i[0].numpy())  # 3,224,224
        A_label.append(i[1])

    return np.asarray(A_data), np.asarray(A_label), len(train_set)

def get_B():

    pre_process = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    )

    train_set = datasets.ImageFolder(
        root = r'/home/xiaoxie/data/tzc/domain_adaptation/DA/data/C',
        transform=pre_process
    )

    print("the number of B is : %d" %len(train_set))
    B_data = []
    B_label = []
    for i in train_set:


        B_data.append(i[0].numpy())
        B_label.append(i[1])

    return np.asarray(B_data),np.asarray(B_label),len(train_set)



def plot_embedding_2D( S_data, Slabel, title):
    # gui yi hua
    Sx_min, Sx_max = np.min(S_data, 0), np.max(S_data, 0)
    Sdata = (S_data - Sx_min) / (Sx_max - Sx_min)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    form = ["o", "v","*","o","v","*"]
    color = ["r","y","g","b","c","m"]
    l = ['S_AMD', 'S_DME', "S_NORMAL", "T_AMD", "T_DME", "T_NORMAL"]

    for i in range(Sdata.shape[0]):
        if i < 4254:
            if Slabel[i] == 0:
                ax1 = plt.scatter(S_data[i, 0], S_data[i, 1], c=color[int(Slabel[i])], marker=form[int(Slabel[i])], s = 5)
            elif Slabel[i] == 1:
                ax2 = plt.scatter(S_data[i, 0], S_data[i, 1], c=color[int(Slabel[i])], marker=form[int(Slabel[i])], s = 5)
            elif Slabel[i] == 2:
                ax3 = plt.scatter(S_data[i, 0], S_data[i, 1], c=color[int(Slabel[i])], marker=form[int(Slabel[i])], s = 5)


        else:
            if Slabel[i] == 0:
                ax4 = plt.scatter(S_data[i, 0], S_data[i, 1], c=color[int(Slabel[i])+3], marker=form[int(Slabel[i])], s = 5)
            elif Slabel[i] == 1:
                ax5 = plt.scatter(S_data[i, 0], S_data[i, 1], c=color[int(Slabel[i])+3], marker=form[int(Slabel[i])], s = 5)
            elif Slabel[i] == 2:
                ax6 = plt.scatter(S_data[i, 0], S_data[i, 1], c=color[int(Slabel[i])+3], marker=form[int(Slabel[i])], s = 5)

    plt.xticks([])
    plt.yticks([])

    plt.legend((ax1, ax2, ax3, ax4, ax5, ax6 ),('S_AMD', 'S_DME', "S_NORMAL", "T_AMD", "T_DME", "T_NORMAL"),loc = "upper right")

    plt.title(title)
    # plt.show()
    plt.savefig('./TSNEBC.jpg', dpi=500)
    return fig

def main():

    A_data, A_label, numA = get_A()
    B_data ,B_label, numB = get_B()
    # resize --> er wei
    new_A_data = A_data.reshape(A_data.shape[0],-1)
    new_B_data = B_data.reshape(B_data.shape[0],-1)

    AA = new_A_data
    BB = new_B_data
    data = np.concatenate((AA,BB),axis=0)


    new_label = np.concatenate((A_label, B_label))

    print('Begining......')
    tsne_2D = TSNE(n_components=2, init='pca', random_state=501)
    result_2D = tsne_2D.fit_transform(data)
    print('Finished......')

    fig1 = plot_embedding_2D(result_2D, new_label,'t-sne')
    plt.show()

if __name__ == '__main__':
    torch.manual_seed(10)
    main()