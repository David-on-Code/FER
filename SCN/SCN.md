
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
面部表情是人类最自然、有力、普遍的表达情绪状态和意图的信号。Automatically
