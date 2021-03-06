
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
![F1](https://github.com/David-on-Code/FER/blob/master/SCN/F1.png)
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
![F2](https://github.com/David-on-Code/FER/blob/master/SCN/F2.png)
Self-Cure Network流程。首先面部图像喂给主干CNN来提取特征。
self-attention importance weighting module 从面部特征学习样本权重得到损失权重。
rank regularization module以样本权重为输入并用排序操作和margin-based的损失函数约束它们。
relabeling module寻找可信赖的样本,通过比较最大预测概率与给定标签的概率。
贴错标签的样本用红色实线矩形标记，模糊的样品用绿色间断线标记。值得注意的是，SCN主要依靠重新加权操作来抑制这些不确定性，并且仅修改一些不确定性样本。  
给定一批带有一些不确定样本的面部图像，首先提取深度特征。self-attention importance weighting module为每个图像分配重要权重使用全连接层（FC）和sigmoid函数。这些权重乘以样本重新加权方案的对数。rank regularization module 调整权重。在这个模块，排序学习的权重并分为两组，high和low。通过margin-based loss约束这些组的平均权重，称为rank regularization loss (RR-Loss)。relabeling module修正low组中的不确定样本。此重新标记操作旨在收集更多干净的样本，然后增强最终模型。可以以端到端的方式训练整个SCN，并轻松地将其添加到任何CNN主干中。  
### 3.2. Self-Attention Importance Weighting  
$F=[x_{1},x_{2},...,x_{N}]\in$$R^{D* N}$表示N张图像的面部特征。F作为self-attention importance weighting module的输入，输出每个特征的重要性权重。该module由线性全连接层和sigmoid激活函数构成，
$$\alpha=\sigma(W_a^Tx_i)$$
$\alpha_i$是第$i$个样本的重要性权重，$W_a$是用于attention的FC层参数，$\sigma$是sigmoid函数。该模块还为其他两个模块提供参考。  
 Logit-Weighted Cross-Entropy Loss.
For a multi-class Cross-Entropy loss, we call our weighted loss as Logit-Weighted Cross-Entropy loss (WCE-Loss), 
（多类别交叉熵损失）
![2](https://github.com/David-on-Code/FER/blob/master/SCN/2.png) $W_j$是第$j$个分类器。$\mathcal{L}$与$\alpha$正相关。  
### 3.3. Rank Regularization
self-attention weights归于（0，1）。在rank regularization module，首先将学习的注意力权重按降序排列，然后以比例$\beta$将其分为两组。排序正则化确保高重要性组的平均注意权重高于低重要性组一个margin。定义rank regularization loss (RR-Loss)，
![34](https://github.com/David-on-Code/FER/blob/master/SCN/3.png)
$\delta_1$是一个margin，可以是固定的超参数或者学习到的参数，$\alpha_H$和$\alpha_L$分别是高重要性组$\beta* N=M$个样本和低重要性组$N-M$个样本。训练时，整个损失函数是$\mathcal{L}_ {all}=\gamma\mathcal{L}_ {RR}+(1-\gamma)\mathcal{L}_ {WCE}$，$gamma$是权衡比例。  
### 3.4. Relabeling  
在rank regularization module，每个mini-batch分为两组，the high-importance 和 the low- importance groups.
修改这些标注的主要挑战是知道哪个标注是错误的。
Specifically, relabeling module近考虑在low-importance  group中的样本并执行Softmax概率。对于每个样本，我们将最大预测概率与给定标签的概率进行比较。如果最大预测概率高于给定标签的阈值，则将样本分配给新的伪标签。Formally,relabeling module定义为，
![5](https://github.com/David-on-Code/FER/blob/master/SCN/5.png)
$y'$表示新标签，$\delta_{2}$是下限，$P_{max}$是最大的预测可能性，$P_{gtInd}$是给定标签的预测可能性。$l_{org}$和$l_{max}$分别是起始给定的标签和最大预测的索引。  
在我们的系统中，不确定的样本有望获得较低的重要权重，从而随着重新加权而降低其负面影响，然后落入低重要性组，最后可以通过重新标记将其校正为确定样本。这些校正后的样本可能在下一个epoch获得较高的重要权重。我们希望可以通过重新加权或重新标记自身来修复网络，这就是为什么我们将我们的方法称为自修复网络的原因。  
### 3.5. Implementation
 Pre-processing and facial features. 在SCN，面部图像检测和对齐通过MTCNN并resized到224x224大小。SCN由Pytorch和ResNet18实现。ResNet-18在MS-Celeb-1M面部识别数据集上预训练，面部特征从最后池化层抽取。  
 Training。
$<I_{input},I_{target}>$,用来训练cGANs,$I_{input}$
