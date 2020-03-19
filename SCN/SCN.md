
## Suppressing Uncertainties for Large-Scale Facial Expression Recognition   
## Abstract  
注释qualitative大规模面部表情数据集十分困难，由歧义的面部表情、低质量面部图像和注释者的主观性造成的不确定性。为了解决这个问题，本文提出一个简单而高效的自修复网络Self-Cure Network(SCN)，有效的抑制了不确定性并阻止深度网络过拟合不确定性面部图像。Specifically，SCN从两方面抑制不确定性：  
⑴在小批量上的自注意机制，通过排名规则化对每个训练样本进行加权；  
⑵重新贴标签机制，在排名最低的组中修改这些样本的标签。  
实验在人造FER数据集和自己收集的WebEmotion数据集上验证方法的有效性。  
RAF-DB 88.14%  
Affect-Net 60.23%   
FERPlus 89.35%  
## 1.Introduction   
面部表情是人类最自然、有力、普遍的表达情绪状态和意图的信号。laboratory or in the wild数据集CK+, MMI, Oulu- CASIA, SFEW/AFEW, FERPlus, AffectNet, EmotioNet, RAF-DB   
![F1]()
从高质量和明显的面部表情到低质量和微表情不确定性增加。不确定性导致反常标签和错误标签，抑制大规模FER进程，especially for 数据驱动的深度学习。Generally,视同不确定FER训练可能导致以下问题。First,错误标注的歧义样本导致过拟合。Second, 对模型学习有用的面部表情特征有害。Third,高比例的错误标签甚至使模型在优化的早期就不融合。  
To address these issues, Self-Cure Network (SCN)。SCN由是三个关键模型组成：自我注意重要性加权（self-attention importance weighting）, 排名正则化（ranking regularization）, and 噪声重新标记（noise relabeling）。一批图像，主干CNN用来提取面部特征。Then自注意重要性权重模型为每张图像学习一个权重来为损失权重获取样本重要性。不确定面部图像被分配低重要性权重。Further,排名正则化模块降序排列这些权重，分成两组（如高重要性权重和低重要性权重），通过强制两组平均权重之间的边距来规范化两组。正则化由损失函数实现，termed as Rank Regularization loss (RR-Loss).ranking regularization module确保第一个模块学习到有意义权重来增强确定的样本（如值得信任的标注）并抑制不确定样本（如模糊标注）。最后的模块是谨慎再标记模块，试图比较最大预测的可能性和给定标签的可能性，重新标记底部组的样本。如果最大预测可能性比已给标签高于一定阈值，样本被指派为假标签。In addition, WebEmotion（来自Internet的极端噪声FER数据集）。  
Overall, contributions总结如下，  
创新地提出FER中的不确定问题，并提出Self-Cure Network来减少不确定性的影响。  
周详地设计分级正则化来监督SCN以学习有意义众艳星权值，为重新标记模块提供参考。  
全面地验证SCN在合成FER数据和新的真实世界的不确定表情数据集(WebEmotion)。在RAF-DB表现88.14%，60.23% on AffectNet, and 89.35% on FERPlus，创造新记录。  
## 2. Related Work  
### 2.1. Facial Expression Recognition  
Generally,一个FER系统主要由三阶段组成，face detection, feature extraction, and expression recognition。  
### 2.2. Learning with Uncertainties  
Particularly,   
## 3. Self-Cure Network  
SCN基于传统CNNs并由三个关键模块组成：  
 i) self-attention importance weight- ing,   
 ii) ranking regularization, and   
 iii) relabeling,  
### 3.1. Overview of Self-Cure Network  
