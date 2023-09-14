import torch.nn as nn
from torchvision import models
import semodel
import torch
resnet_dict = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
}

def get_backbone(name):
    if "resnet" in name.lower():
        return ResNetBackbone(name)
    elif "alexnet" == name.lower():
        return AlexNetBackbone()
    elif "dann" == name.lower():
        return DaNNBackbone()
    elif "cbamalexnet" == name.lower():
        return alexnet2()
    elif "CBAMalexnet" == name.lower():
        return cbam_alexnet()
    elif "vgg16" == name.lower():
        return VGGBackbone()
    elif "vgg11" == name.lower():
        return vgg_11()
    elif "se" in name.lower():
        return seResNetBackbone()


class DaNNBackbone(nn.Module):
    def __init__(self, n_input=224*224*3, n_hidden=256):
        super(DaNNBackbone, self).__init__()
        self.layer_input = nn.Linear(n_input, n_hidden)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self._feature_dim = n_hidden

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x

    def output_num(self):
        return self._feature_dim
    
# convnet without the last layer
class AlexNetBackbone(nn.Module):
    def __init__(self):
        super(AlexNetBackbone, self).__init__()
        model_alexnet = models.alexnet(pretrained=True)
        self.features = model_alexnet.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module(
                "classifier"+str(i), model_alexnet.classifier[i])
        self._feature_dim = model_alexnet.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256*6*6)
        x = self.classifier(x)
        return x

    def output_num(self):
        return self._feature_dim

class ResNetBackbone(nn.Module):
    def __init__(self, network_type):
        super(ResNetBackbone, self).__init__()
        resnet = resnet_dict[network_type](pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self._feature_dim = resnet.fc.in_features
        del resnet
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
    
    def output_num(self):
        return self._feature_dim


class seResNetBackbone(nn.Module):
    def __init__(self):
        super(seResNetBackbone, self).__init__()
        resnet = semodel.se_resnet_50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self._feature_dim = resnet.fc.in_features
        del resnet

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self._feature_dim


class VGGBackbone(nn.Module):
    def __init__(self):
        super(VGGBackbone, self).__init__()
        model_VGG = models.vgg16(pretrained=True)
        self.features = model_VGG.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module(
                "classifier"+str(i), model_VGG.classifier[i])
        self._feature_dim = model_VGG.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 25088)
        x = self.classifier(x)
        return x

    def output_num(self):
        return self._feature_dim

# 自己加的model
class SElayer(nn.Module):
    def __init__(self, channel, reduction = 16):
        super(SElayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias= False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias= False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x*y.expand_as(x)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        # print('outchannels:{}'.format(out.shape))
        out = self.spatial_attention(out) * out
        return out

# 注意力机制+alexnet
class alexnet2(nn.Module):
    def __init__(self):
        super(alexnet2,self).__init__()
        alexnet = models.alexnet(pretrained= True)
        self.features = nn.Sequential(
            alexnet.features[0],
            SElayer(64),
            #CBAM(64),
            alexnet.features[1],
            alexnet.features[2],
            alexnet.features[3],
            SElayer(192),
            #CBAM(192),
            alexnet.features[4],
            alexnet.features[5],
            alexnet.features[6],
            SElayer(384),
            #CBAM(384),
            alexnet.features[7],
            alexnet.features[8],
            SElayer(256),
            #CBAM(256),
            alexnet.features[9],
            alexnet.features[10],
            SElayer(256),
            #CBAM(256),
            alexnet.features[11],
            alexnet.features[12],
        )
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module(
                "classifier"+str(i), alexnet.classifier[i])
        self._feature_dim = alexnet.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256*6*6)
        x = self.classifier(x)
        return x

    def output_num(self):
        return self._feature_dim



#vgg16
class vgg_16(nn.Module):
    def __init__(self):
        super(vgg_16,self).__init__()
        vgg = models.vgg16(pretrained= True)
        self.features = nn.Sequential(
            vgg.features[0],
            SElayer(64),
            vgg.features[1],
            vgg.features[2],
            SElayer(64),
            vgg.features[3],
            vgg.features[4],
            vgg.features[5],
            SElayer(128),
            vgg.features[6],
            vgg.features[7],
            SElayer(128),
            vgg.features[8],
            vgg.features[9],
            vgg.features[10],
            SElayer(256),
            vgg.features[11],
            vgg.features[12],
            SElayer(256),
            vgg.features[13],
            vgg.features[14],
            SElayer(256),
            vgg.features[15],
            vgg.features[16],
            vgg.features[17],
            SElayer(512),
            vgg.features[18],
            vgg.features[19],
            SElayer(512),
            vgg.features[20],
            vgg.features[21],
            SElayer(512),
            vgg.features[22],
            vgg.features[23],
            vgg.features[24],
            SElayer(512),
            vgg.features[25],
            vgg.features[26],
            SElayer(512),
            vgg.features[27],
            vgg.features[28],
            SElayer(512),
            vgg.features[29],
            vgg.features[30],
        
        )
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module(
                "classifier"+str(i), vgg.classifier[i])
        self._feature_dim = vgg.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 25088)
        x = self.classifier(x)
        #x = nn.DataParallel(x,device_ids = [0,1,2,3])
        return x

    def output_num(self):
        return self._feature_dim


#vgg16  无注意力机制
class vgg_17(nn.Module):
    def __init__(self):
        super(vgg_17,self).__init__()
        vgg = models.vgg16(pretrained= True)
        self.features = nn.Sequential(
            vgg.features[0],
            #SElayer(64),
            vgg.features[1],
            vgg.features[2],
            #SElayer(64),
            vgg.features[3],
            vgg.features[4],
            vgg.features[5],
            #SElayer(128),
            vgg.features[6],
            vgg.features[7],
            #SElayer(128),
            vgg.features[8],
            vgg.features[9],
            vgg.features[10],
            #SElayer(256),
            vgg.features[11],
            vgg.features[12],
            #SElayer(256),
            vgg.features[13],
            vgg.features[14],
            #SElayer(256),
            vgg.features[15],
            vgg.features[16],
            vgg.features[17],
            #SElayer(512),
            vgg.features[18],
            vgg.features[19],
            #SElayer(512),
            vgg.features[20],
            vgg.features[21],
            #SElayer(512),
            vgg.features[22],
            vgg.features[23],
            vgg.features[24],
            #SElayer(512),
            vgg.features[25],
            vgg.features[26],
            #SElayer(512),
            vgg.features[27],
            vgg.features[28],
            #SElayer(512),
            vgg.features[29],
            vgg.features[30],
        
        )
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module(
                "classifier"+str(i), vgg.classifier[i])
        self._feature_dim = vgg.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 25088)
        x = self.classifier(x)
        #x = nn.DataParallel(x,device_ids = [0,1,2,3])
        return x

    def output_num(self):
        return self._feature_dim



#vgg11
class vgg_11(nn.Module):
    def __init__(self):
        super(vgg_11,self).__init__()
        vgg = models.vgg11(pretrained= True)
        self.features = nn.Sequential(
            vgg.features[0],
            #SElayer(64),
            vgg.features[1],
            vgg.features[2],
            vgg.features[3],
            #SElayer(64),
            vgg.features[4],
            vgg.features[5],
            
            vgg.features[6],
            #SElayer(128),
            vgg.features[7],
            
            vgg.features[8],
            #SElayer(256),
            vgg.features[9],
            vgg.features[10],
            
            vgg.features[11],
            #SElayer(256),
            vgg.features[12],
            
            vgg.features[13],
            #SElayer(512),
            vgg.features[14],
            
            vgg.features[15],
            vgg.features[16],
            #SElayer(512),
            vgg.features[17],
            
            vgg.features[18],
            #SElayer(512),
            vgg.features[19],
            
            vgg.features[20],
           
        
        )
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module(
                "classifier"+str(i), vgg.classifier[i])
        self._feature_dim = vgg.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 25088)
        x = self.classifier(x)
        #x = nn.DataParallel(x,device_ids = [0,1,2,3])
        return x

    def output_num(self):
        return self._feature_dim


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        # print(avgout.shape)
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)



class cbam_alexnet(nn.Module):
    def __init__(self):
        super(cbam_alexnet,self).__init__()
        alexnet = models.alexnet(pretrained= True)
        self.features = nn.Sequential(
            alexnet.features[0],
            #CBAM(64),
            alexnet.features[1],
            alexnet.features[2],
            alexnet.features[3],
            #CBAM(192),
            alexnet.features[4],
            alexnet.features[5],
            alexnet.features[6],
            #CBAM(384),
            alexnet.features[7],
            alexnet.features[8],
            #CBAM(256),
            alexnet.features[9],
            alexnet.features[10],
            #CBAM(256),
            alexnet.features[11],
            alexnet.features[12],
        )
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module(
                "classifier"+str(i), alexnet.classifier[i])
        self._feature_dim = alexnet.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256*6*6)
        x = self.classifier(x)
        return x

    def output_num(self):
        return self._feature_dim

