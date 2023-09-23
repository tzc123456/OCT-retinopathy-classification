# OCT-retinopathy-classification
A Pytorch implementation of "OCT retinopathy classification via a semi-supervised domain adaptation and fine-tuning method". The contribution of this paper are summarized as follows:
1. A modified deep subdomain adaptation network with pseudo-labels (DSAN-PL) was proposed to firstly align the feature spaces of a public domain (labeled) and a private domain (unlabeled).
2. The DSAN-PL model was then fine-tuned using the provided few labeled OCT data in the private domain.
# Algorithm overview
![image](https://github.com/tzc123456/OCT-retinopathy-classification/assets/82322328/d936543e-5ad2-45ab-b3d7-a6b183e04bec)
# Deep subdomain adaptation with pseudo-label
![image](https://github.com/tzc123456/OCT-retinopathy-classification/assets/82322328/88a7bb72-0cb9-4767-ad3c-ca2cac0856a4)
# Reaults
![image](https://github.com/tzc123456/OCT-retinopathy-classification/assets/82322328/0a536a33-bf1a-43e1-91e3-87f85470c1fd)
![image](https://github.com/tzc123456/OCT-retinopathy-classification/assets/82322328/02872b60-f446-4776-aa21-ee349b563844)
![image](https://github.com/tzc123456/OCT-retinopathy-classification/assets/82322328/9a82bb7d-46bf-4db3-a18e-287bc7293864)

# Reference
@Misc{deepda,
howpublished = {\url{https://github.com/jindongwang/transferlearning/tree/master/code/DeepDA}},   
title = {DeepDA: Deep Domain Adaptation Toolkit},  
author = {Wang, Jindong and Hou, Wenxin}
}  
