# SLTRL
This is the source code of our SPL paper:

<Similarity Learning with Top-heavy Ranking Loss for Person Re-identification>

The package contains the following components:

(1)/code

the source code for the proposed SLTRL method. It should be noted that the pca toolbox is downloaded from the homepage of Prof. Deng Cai.

(2)/demo

a demo for viper dataset. Please run LOMO_SLTRL.m to see the result.

(3)/cache

the extracted feature for each dataset. Here, we use LOMO [1] feature proposed by Prof. Shengcai Liao to conduct the experiment, please direct to his homepage to download the source code. (http://www.cbsr.ia.ac.cn/users/scliao/projects/lomo_xqda/)

(4)/mat

train/test partition files for each dataset. The split is generated randomly, and the final reported result is the average of 10 random splits.

Version: 1.0
Date: 2015-11-17

Author: Jin Wang
Institute: Huazhong University of Science and Techonology

Email: jinw@hust.edu.cn

References:
[1] Shengcai Liao, Yang Hu, Xiangyu Zhu, and Stan Z. Li. “Person re-identification by local maximal occurrence representation and metric learning.” In IEEE Conference on Computer Vision and Pattern Recognition, 2015.
