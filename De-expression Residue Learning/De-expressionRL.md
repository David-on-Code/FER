### Facial Expression Recognition by De-expression Residue Learning  
#### Abstract
一个人的面部表情由表情成分和中性成分组成。这篇文章中，我们提出通过de-expression learning procedure提取表情成分信息的面部表情识别方法，叫做残余表情识别算法（De-expression Residue Learning，DeRL）。首先，通过cGAN训练一个生成模型。这个模型对于任何输入的人脸图像生成大致相当的中性脸。我们称这个过程为de-expression,因为这个生成模型筛选表情信息；然而，表情信息仍然记录在中间层。给定中性脸，不像以前使用像素级或者特征级不同来面部表情分类，我们的新方法学习生成模型的中间层残留得到残余表情。这种剩余表情很重要，因为它包含任何输入表情图片冗余在生成模型的表情成分。7种公开面部表情数据集在我们的实验中使用。两个数据集作为预训练（BU-4DFE 和 BP4D-spontaneous），DeRL方法在五个数据集上评估，CK+，Oulu-CASIA, MMI, BU-3DFE, and BP4D+.实验结果表明提出的方法表现优越。  
#### 1. Introduction  
FER的研究在各种成像条件下的姿势和自发的面部表情
在包括各种头部姿势，照明条件，分辨率和遮挡在内的各种成像条件下，已经对姿势和自发的面部表情进行了（FER）的研究。
