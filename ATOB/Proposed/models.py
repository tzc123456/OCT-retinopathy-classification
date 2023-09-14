import torch
import torch.nn as nn
from transfer_losses import TransferLoss
import backbones

class TransferNet(nn.Module):
    def __init__(self, num_class, base_net='resnet50', transfer_loss='mmd', use_bottleneck=True, bottleneck_width=256, max_iter=1000, **kwargs):
        super(TransferNet, self).__init__()
        self.num_class = num_class


        self.base_network = backbones.get_backbone(base_net)
        self.use_bottleneck = use_bottleneck
        self.transfer_loss = transfer_loss
        if self.use_bottleneck:
            bottleneck_list = [
                nn.Linear(self.base_network.output_num(), bottleneck_width),
                nn.ReLU()
            ]
            self.bottleneck_layer = nn.Sequential(*bottleneck_list)
            #self.bottleneck_layer = nn.DataParallel(nn.Sequential(*bottleneck_list), device_ids = [0,1,2,3])
            feature_dim = bottleneck_width
        else:
            feature_dim = self.base_network.output_num()
        
        self.classifier_layer = nn.Linear(feature_dim, num_class)
     
        transfer_loss_args = {
            "loss_type": self.transfer_loss,
            "max_iter": max_iter,
            "num_class": num_class
        }
      
        self.adapt_loss = TransferLoss(**transfer_loss_args)
        self.criterion = torch.nn.CrossEntropyLoss()
        

    def reset_parameters(self):
        self.lsoftmax_linear.reset_parameters()
    def forward(self, source, target, source_label,e):
        source = self.base_network(source)
        target = self.base_network(target)
        if self.use_bottleneck:
            source = self.bottleneck_layer(source)
            target = self.bottleneck_layer(target)
        # classification
        source_clf = self.classifier_layer(source)
        #.....................................
        target_clf = self.classifier_layer(target)


        #with torch.enable_grad():
        "pseudo label"
        p = torch.nn.functional.softmax(target_clf,dim=-1)  # softmax
        # EM_LOSS = -1 * torch.sum(p * torch.nn.functional.log_softmax(target_clf, dim=-1)) / target_clf.size()[0]


        # 设定阈值，返回伪标签样本，e为迭代的轮次

        if e < 10:
            a = 0
        elif 10<= e < 60:
            # print("kai shi wei biao qian")
            a = ((e - 10) / (60 - 10)) * 0.3
            # print(a)
        else:
            a = 0.3
            # print(a)

        sample = []
        for hang in range(p.size(0)):
            for col in range(2):
                if p[hang, col] > 0.99:
                    sample.append(hang)
        target_sample = target_clf[sample, :]

        if  min(target_sample.shape) == 0:
            clf_loss = self.criterion(source_clf, source_label)
            #print(666)
        else:
            #print("you wei biao qian la !!")
            target_label = torch.nn.functional.softmax(target_sample, dim=-1)

            _ , target_label = torch.max(target_label, dim=1)
            #print(target_label.size())

            clf_loss =  self.criterion(source_clf, source_label) + a * self.criterion(target_sample, target_label)

        #clf_loss = self.criterion(source_clf, source_label)
        #...............................................
        # if e > 30:
        #     clf_loss = self.criterion(source_clf, source_label)  + 0.3 * EM_LOSS
        # else:
        #     clf_loss = self.criterion(source_clf, source_label)

        # y_output  =torch.log_softmax(self.output(source),dim=1)
        # loss_cls = self.loss_fn_cls(y_output,source_label)
        # source_label1 = source_label.float()
        # loss_center = self.c_net(source,source_label1)
        # clf_loss = loss_cls + loss_center

        # transfer
        kwargs = {}
        
        if self.transfer_loss == "lmmd":
            kwargs['source_label'] = source_label
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        elif self.transfer_loss == "daan":
            source_clf = self.classifier_layer(source)
            kwargs['source_logits'] = torch.nn.functional.softmax(source_clf, dim=1)
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        elif self.transfer_loss == 'bnm':
            tar_clf = self.classifier_layer(target)
            target = nn.Softmax(dim=1)(tar_clf)

       

        transfer_loss = self.adapt_loss(source, target, **kwargs) 
        return clf_loss, transfer_loss
    
    def get_parameters(self, initial_lr=1.0):
        params = [
            {'params': self.base_network.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.classifier_layer.parameters(), 'lr': 1.0 * initial_lr},
            #{'params': self.adv_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr},
            #{'params': self.lsoftmax_linear.parameters(), 'lr': 1.0 * initial_lr},
            #{'params': self.output.parameters(), 'lr': 1.0 * initial_lr},
        ]
        if self.use_bottleneck:
            params.append(
                {'params': self.bottleneck_layer.parameters(), 'lr': 1.0 * initial_lr}
            )
        # Loss-dependent
        if self.transfer_loss == "adv":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
        elif self.transfer_loss == "daan":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
            params.append(
                {'params': self.adapt_loss.loss_func.local_classifiers.parameters(), 'lr': 1.0 * initial_lr}
            )
        return params

    def predict(self, x):
        features = self.base_network(x)
        x = self.bottleneck_layer(features)
        clf = self.classifier_layer(x)
        #clf = self.lsoftmax_linear(x)
        #clf = self.output(x)
        return clf

    def epoch_based_processing(self, *args, **kwargs):
        if self.transfer_loss == "daan":
            self.adapt_loss.loss_func.update_dynamic_factor(*args, **kwargs)
        else:
            pass
