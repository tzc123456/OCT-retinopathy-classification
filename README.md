# OCT-retinopathy-classification
A Pytorch implementation of "OCT retinopathy classification via a semi-supervised domain adaptation and fine-tuning method". The contribution of this paper are summarized as follows:
1. A modified deep subdomain adaptation network with pseudo-labels (DSAN-PL) was proposed to firstly align the feature spaces of a public domain (labeled) and a private domain (unlabeled).
2. The DSAN-PL model was then fine-tuned using the provided few labeled OCT data in the private domain.
# Reference
@Misc{deepda,
howpublished = {\url{https://github.com/jindongwang/transferlearning/tree/master/code/DeepDA}},   
title = {DeepDA: Deep Domain Adaptation Toolkit},  
author = {Wang, Jindong and Hou, Wenxin}
}  
