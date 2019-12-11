### Deep Facial Expression Recognition: A Survey [addr](https://arxiv.org/pdf/1804.08348.pdf)  
#### cs.CV 22 Oct 2018
##### aboratory-controlled->in-the-wild conditions
面部表情是人类传达其情绪状态和意图的最有力，最自然和普遍的信号之一。由于自动面部表情分析在社交机器人，医学治疗，驾驶员疲劳监测以及许多其他人机交互系统中的实用重要性，因此已经进行了许多研究。早在20世纪，Ekman和Friesen在跨文化研究的基础上定义了六种基本情感，这表明无论文化如何，人们都以相同的方式感知某些基本情感。这些典型的面部表情是愤怒anger，厌恶disgust，恐惧fear，幸福happiness，悲伤sadness和惊奇surprise。随后加入了鄙视contempt，这是基本情绪之一。但是，最近，有关神经科学和心理学的高级研究认为，六种基本情绪的模型是特定于文化的，而不是通用的。  
尽管基于基本情感的情感模型在表达我们日常情感展示的复杂性和微妙性的能力方面受到限制，并且其他情绪描述模型，例如面部动作编码系统（FACS）和andthe continuous model using affect dimensions 被认为代表更广泛的情感，由于其开创性的研究以及对面部表情的直接直观定义，用离散基本情感描述情感的分类模型仍然是FER最受欢迎的观点。    
根据特征表示，FER系统可分为两大类：静态图像FER和动态序列FER。在静态方法中，特征表示仅使用当前单个图像中的空间信息进行编码，基于动态的方法，考虑了输入面部表情序列中连续帧之间的时间关系。  
The majority of the traditional methods have used handcraftedfeatures or shallow learning (e.g., local binary patterns (LBP),LBP  on  three  orthogonal  planes  (LBP-TOP),  non-negativematrix factorization (NMF)and sparse learning for FER.但是，自2013年以来，诸如FER2013和Wild情感识别（EmotiW)，等情感识别竞赛从具有挑战性的现实场景中收集了相对足够的训练数据，这相当促进从ab-controlled to in-the-wild环境的FER。同时，由于芯片处理能力的显着提高和精心设计的网络架构，各个领域的研究已开始转移到深度学习方法，这些方法已经达到了最新的识别精度，并且大大超过了以前的结果。图1在算法和数据集方面说明了FER的这种演变。 ![](https://github.com/David-on-Code/Facial-expression-recognizition/blob/master/Datasets/Figure1.png)  
尽管深度学习有强大的特征学习能力，但是应用于FER时仍然存在问题。首先深度神经网络需要大量的训练数据来避免过拟合。然而，现存的表情数据库不足够用众所周知的深度架构的神经网络训练实现更好的结果在目标识别任务。此外主体间存在很大的差异由于不同的人的属性，如年龄、性别、种族背景和表达水平。除了目标个体差异，姿势、光照和遮挡也常见于不受约束的面部表情方案。这些因素与面部表情非线性耦合，因此加强了深度网络的需求，以应对较大的类内变化并学习有效的表情特定表示

深度面部表情识别  
pre-processing,  
背景、光照、头姿势。需要进行预处理以对齐和标准化人脸传达的视觉语义信息
deep feature learing, deep feature classification
![overview of datasets](https://github.com/David-on-Code/Facial-expression-recognizition/blob/master/Datasets/WeChatee3dfadaf56be8150efceab7948523e7.png)
[CK+](http://www.pitt.edu/~emotion/ck-spread.htm)  
[MMI](https://mmifacedb.eu/)  
[JAFFE](http://www.kasrl.org/jaffe.html)  
[FER-2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)  
[AFEW7.0](https://sites.google.com/site/emotiwchallenge/)  
[SFEW2.0](https://cs.anu.edu.au/few/emotiw2015.html)  
[Multi-PIE](http://www.flintbox.com/public/project/4742/)  
[BU-3DFE](http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html)  
[Oulu-CASIA](http://www.cse.oulu.fi/CMV/Downloads/Oulu-CASIA)  
[RaFD](http://www.socsci.ru.nl:8180/RaFD2/RaFD)  
[KDEF](http://www.emotionlab.se/kdef/)  
[EmotioNet](http://cbcsl.ece.ohio-state.edu/dbform_emotionet.html)  
[RAF-DB](http://www.whdeng.cn/RAF/model1.html)  
[AffectNet](http://mohammadmahoor.com/databases-codes/)  
[ExpW](http://mmlab.ie.cuhk.edu.hk/projects/socialrelation/index.html)  
[公开人脸数据集](https://blog.csdn.net/lilai619/article/details/51178971)  
