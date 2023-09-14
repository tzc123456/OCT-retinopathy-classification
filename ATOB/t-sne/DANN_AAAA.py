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

batch_size = 128
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
    batch_size= batch_size,
    shuffle= False,
    num_workers=4,
)


B_set = datasets.ImageFolder(

    root = r'/home/xiaoxie/data/tzc/domain_adaptation/DA/data/B',
    transform= data_transform['val']
)


B_dataloader = DataLoader(
    B_set,
    batch_size= batch_size,
    shuffle= False,
    num_workers=4,
)

print("The number of source domain: {}".format(len(A_set)))
print("The number of target domain: {}".format(len(B_set)))

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
        if i < len(A_set):
            if Slabel[i] == 0:
                ax1 = plt.scatter(S_data[i, 0], S_data[i, 1], c=color[int(Slabel[i])], marker=form[int(Slabel[i])], s = 5)
            elif Slabel[i] == 1:
                ax2 = plt.scatter(S_data[i, 0], S_data[i, 1], c=color[int(Slabel[i])], marker=form[int(Slabel[i])], s = 5)
            elif Slabel[i] == 2:
                ax3 = plt.scatter(S_data[i, 0], S_data[i, 1], c=color[int(Slabel[i])], marker=form[int(Slabel[i])], s = 5)

            # plt.text(Sdata[i, 0], Sdata[i, 1], str(int(Slabel[i])),
            #          color=plt.cm.Set1(int(Slabel[i])),
            #          fontdict={'weight': 'bold', 'size': 9},
            #          )
            #ax1 = plt.scatter(S_data[i,0], S_data[i,1],c = color[int(Slabel[i])], marker=form[int(Slabel[i])])
        else:
            if Slabel[i] == 0:
                ax4 = plt.scatter(S_data[i, 0], S_data[i, 1], c=color[int(Slabel[i])+3], marker=form[int(Slabel[i])], s = 5)
            elif Slabel[i] == 1:
                ax5 = plt.scatter(S_data[i, 0], S_data[i, 1], c=color[int(Slabel[i])+3], marker=form[int(Slabel[i])], s = 5)
            elif Slabel[i] == 2:
                ax6 = plt.scatter(S_data[i, 0], S_data[i, 1], c=color[int(Slabel[i])+3], marker=form[int(Slabel[i])], s = 5)
            #ax1 = plt.scatter(S_data[i, 0], S_data[i, 1], c=color[int(Slabel[i])+3], marker=form[int(Slabel[i])] )
            # plt.text(Sdata[i, 0], Sdata[i, 1], str(int(Slabel[i])),
            #          color=plt.cm.Set1(int(Slabel[i])+3),
            #          fontdict={'weight': 'bold', 'size': 9},
            #          )

    # for i in range(Sdata.shape[0]):
    #     plt.text(Sdata[i, 0], Sdata[i, 1], str(int(Slabel[i])),
    #              color=plt.cm.Set1(int(Slabel[i])),
    #              fontdict={'weight': 'bold', 'size': 9},
    #              )
    #
    # Tx_min, Tx_max = np.min(T_data, 0), np.max(T_data, 0)
    # Tdata = (T_data - Tx_min) / (Tx_max - Tx_min)
    # Tdata = (T_data - Sx_min) / (Sx_max - Sx_min)
    # for i in range(Tdata.shape[0]):
    #     plt.text(Tdata[i, 0], Tdata[i, 1], str(int(Tlabel[i])),
    #              color=plt.cm.Set1(int(Tlabel[i])+3),
    #              fontdict={'weight': 'bold', 'size': 9})


    plt.xticks([])
    plt.yticks([])

    # l = ['S_AMD', 'S_DME', "S_NORMAL", "T_AMD", "T_DME", "T_NORMAL"]
    #color = [plt.cm.Set1(0), plt.cm.Set1(1),plt.cm.Set1(2),plt.cm.Set1(3),plt.cm.Set1(4),plt.cm.Set1(5)]
    #patches = [mpatches.Patch(color=color[i], label='{:s}'.format(l[i])) for i in range(6)]

    #patches = [mpatches.Patch(color=color[i],linestyle="-", label='{:s}'.format(l[i])) for i in range(6)]
    #plt.legend(handles=patches)
    #plt.legend()
    #plt.legend(handles = ax1.legend_elements()[0], title = "classes")
    # plt.rcParams.update({"font.size": 20})
    plt.legend((ax1, ax2, ax3, ax4, ax5, ax6 ),('S_AMD', 'S_DME', "S_NORMAL", "T_AMD", "T_DME", "T_NORMAL"),loc = "upper right")






    plt.title(title)
    # plt.show()
    plt.savefig('./Proposed.jpg', dpi=500)
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
    model = models.TransferNet(3,device= DEVICE,base_net="resnet50", transfer_loss='lmmd')
    model.load_state_dict(torch.load(
        r'/home/xiaoxie/data/tzc/domain_adaptation/DA/ATOB/Proposed/Proposed.pth'))
    model.to(DEVICE)




    for epoch in range(1):
        model.eval()
        # A_data = []
        # A_label = []
        # B_data = []
        # B_label = []

        A_data = np.zeros((1, 256))
        A_label = np.zeros((1))
        # data_B = np.ones((1,256))
        B_data = np.zeros((1, 256))
        B_label = np.zeros((1))
        with torch.no_grad():
            for step,val_data in enumerate(A_dataloader):
                val_images,val_labels = val_data
                # output = model.predict(val_images.to(DEVICE))
                output = model.base_network(val_images.to(DEVICE))
                output = model.bottleneck_layer(output.to(DEVICE))
                output = output.cpu()
                # A_data.append(output.numpy())
                # A_label.append(val_labels.numpy())
                A_data = np.concatenate((A_data, output), axis=0)
                A_label = np.concatenate((A_label, val_labels), axis=0)






            for step,val_data in enumerate(B_dataloader):
                val_images,val_labels = val_data
                #output = model.predict(val_images.to(DEVICE))
                output = model.base_network(val_images.to(DEVICE))
                output = model.bottleneck_layer(output.to(DEVICE))
                output = output.cpu()
                # B_data.append(output.numpy())
                # B_label.append(val_labels.numpy())
                B_data = np.concatenate((B_data, output), axis=0)
                B_label = np.concatenate((B_label, val_labels), axis=0)



            print('Begining......')

            A_data = np.asarray(A_data)
            B_data = np.asarray(B_data)
            A_label = np.asarray(A_label)
            B_label = np.asarray(B_label)

            # A_data = A_data.reshape(A_data.shape[0],-1)
            # B_data = B_data.reshape(B_data.shape[0],-1)



            data_ab = np.concatenate((A_data, B_data),axis=0)
            label_ab = np.concatenate((A_label, B_label), axis=0)

            tsne_2D = TSNE(n_components=2, init='pca', random_state=501) # 501
            # result_2D_A = tsne_2D.fit_transform(A_data)
            # result_2D_B = tsne_2D.fit_transform(B_data)
            result_2D_AB = tsne_2D.fit_transform(data_ab)

            # la = np.zeros((2801,))
            # lb = np.ones((2801,))
            # la = np.zeros((len(A_set)+1,))
            # lb = np.ones((len(B_set)+1,))



            # la = np.zeros((193,))
            # lb = np.ones((193,))
            # lab = np.concatenate((la,lb))

            print('Finished......')

            #fig1 = plot_embedding_2D(result_2D_AB, A_label, result_2D_B, B_label, 'DSAN')
            fig1 = plot_embedding_2D(result_2D_AB, label_ab, 'Proposed')

            plt.show()



if __name__ == '__main__':

    main()


