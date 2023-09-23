# OCT-retinopathy-classification
A Pytorch implementation of "OCT retinopathy classification via a semi-supervised domain adaptation and fine-tuning method". The contribution of this paper are summarized as follows:
1. A modified deep subdomain adaptation network with pseudo-labels (DSAN-PL) was proposed to firstly align the feature spaces of a public domain (labeled) and a private domain (unlabeled).
2. The DSAN-PL model was then fine-tuned using the provided few labeled OCT data in the private domain.

# Usage
1. Modify the configuration file in the corresponding directories
2. Run the main.py with specified config, for example, python main.py --config DAN/DAN.yaml
   
# Algorithm overview
![image](https://github.com/tzc123456/OCT-retinopathy-classification/assets/82322328/d936543e-5ad2-45ab-b3d7-a6b183e04bec)

# Datasets
In this paper, we used three public OCT retinopathy datasets to demonstrate the effectiveness of the proposed algorithm. The first dataset (denoted as Dataset A) was acquired from 45 subjects in different locations of USA, which includes 723 AMD, 1101 DME and 1407 normal images [1]. The second dataset (denoted as Dataset B) was obtained at Noor Eye Hospital in Tehran, Iran [2]. The third dataset was collected from 6 different hospitals in USA and China, which includes 37206 CNV, 11349 DME, 8617 drusen and 51140 normal images from 4686 subjects [3]. 

# Results

# Domain bias experiment results
In order to illustrate the differences among the different domains in the three datasets, three DL models were trained. Specifically, we trained the model with 90% labeled OCT images on the Dataset A, which was called as Model A, and then tested Model A on the rest of the Dataset A, full Dataset B and full Dataset C, respectively. This modeling process was repeated for Dataset B and Dataset C, which was named as Model B and Model C, respectively.

![image](https://github.com/tzc123456/OCT-retinopathy-classification/assets/82322328/39ea646a-314f-4877-853f-29dcceff97df)

# Unsupervised deep subdomain adaptation with pseudo-label experiment results
In Step 1, a neural network was trained using the proposed domain adaptation algorithm. To demonstrate the superior performance of the proposed domain adaptation method, we compared it with some popular and state-of-the-art domain adaptation methods, including DAN [4], DANN [5], DeepCoral [6] and DSAN [7]. Specifically, DAN, DeepCoral and DSAN are statistic moment matching-based methods, while DANN is an adversarial-based method.
For fair comparison, we have performed three domain adaptation tasks, i.e. A to B, A to C and B to C. For A to B, the labeled domain is Dataset A, while the unlabeled domain is Dataset B. The scenarios of A to C and B to C are similarly defined.

![image](https://github.com/tzc123456/OCT-retinopathy-classification/assets/82322328/72ce6b76-db02-4828-952e-9d51df5bf82a)

# Semi-supervised transfer-learning results
We further tested the scenario where there was a small percentage (10%) of data with true labels existing in the private domain and how domain adaptation could help improve the classification performance. Following the previous experiment, four different experiments were conducted on Dataset B and Dataset C as the private domain where 10% data was labeled. The first method is the basic one without transfer-learning (No-TL), i.e., the model trained using 10% of the private OCT dataset first and then tested on the remaining 90% data, with random initialization. The second one is the transfer-learning with ImageNet (TL-ImageNet). This model was obtained by fine-tuning the basic ResNet-50 model pre-trained on the ImageNet dataset using 10% of the private OCT dataset. The third one is similar to the TL-ImageNet, with the difference that we further fine-tuned the model using a public OCT dataset (here indicating Dataset A) and it was named TL-OCT. The last one is the transfer-learning with the subdomain adaptation model proposed in this study (TLDA). 

![image](https://github.com/tzc123456/OCT-retinopathy-classification/assets/82322328/1d3d0abd-c435-4670-8597-1abcc9485f7d)






# Reference
[1] Srinivasan PP, Kim LA, Mettu PS, et al (2014) Fully automated detection of diabetic macular edema and dry age-related macular degeneration from optical coherence tomography images. Biomed Opt Express. 5(10): 3568-3577.

[2] Rasti R, Rabbani H, Mehridehnavi A, et al (2017) Macular OCT classification using a multi-scale convolutional neural network ensemble. IEEE Trans Med Imaging. 37(4): 1024-1034.

[3] Kermany DS, Goldbaum M, Cai W, et al (2018) Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. Cell. 172(5): 1122-1131.

[4] Long M, Cao Y, Wang J, et al (2015) Learning transferable features with deep adaptation networks. Proc Int Conf Mach Learn. 37: 97-105.

[5] Ganin Y, Lempitsky V (2015) Unsupervised domain adaptation by backpropagation. Proc Int Conf Mach Learn. 37: 1180-1189.

[6] Sun B, Saenko K (2016) Deep coral: Correlation alignment for deep domain adaptation. Proc Eur Conf Comput Vis. 443-450.

[7] Zhu Y, Zhuang F, Wang J, et al (2020) Deep subdomain adaptation network for image classification. IEEE Trans Neural Netw Learn Syst. 32(4): 1713-1722.

@Misc{deepda,
howpublished = {\url{https://github.com/jindongwang/transferlearning/tree/master/code/DeepDA}},   
title = {DeepDA: Deep Domain Adaptation Toolkit},  
author = {Wang, Jindong and Hou, Wenxin}
}  
