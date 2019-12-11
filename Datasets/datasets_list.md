### Deep Facial Expression Recognition: A Survey [addr](https://arxiv.org/pdf/1804.08348.pdf)  
#### cs.CV 22 Oct 2018
##### aboratory-controlled->in-the-wild conditions
面部表情是人类传达其情绪状态和意图的最有力，最自然和普遍的信号之一。由于自动面部表情分析在社交机器人，医学治疗，驾驶员疲劳监测以及许多其他人机交互系统中的实用重要性，因此已经进行了许多研究。早在20世纪，Ekman和Friesen在跨文化研究的基础上定义了六种基本情感，这表明无论文化如何，人们都以相同的方式感知某些基本情感。这些典型的面部表情是愤怒anger，厌恶disgust，恐惧fear，幸福happiness，悲伤sadness和惊奇surprise。随后加入了鄙视contempt，这是基本情绪之一。但是，最近，有关神经科学和心理学的高级研究认为，六种基本情绪的模型是特定于文化的，而不是通用的。  
尽管基于基本情感的情感模型在表达我们日常情感展示的复杂性和微妙性的能力方面受到限制，并且其他情绪描述模型，例如面部动作编码系统（FACS）和andthe continuous model using affect dimensions 被认为代表更广泛的情感，由于其开创性的研究以及对面部表情的直接直观定义，用离散基本情感描述情感的分类模型仍然是FER最受欢迎的观点。    
根据特征表示，FER系统可分为两大类：静态图像FER和动态序列FER。在静态方法中，特征表示仅使用当前单个图像中的空间信息进行编码，基于动态的方法，考虑了输入面部表情序列中连续帧之间的时间关系。  
The majority of the traditional methods have used handcraftedfeatures or shallow learning (e.g., local binary patterns (LBP),LBP  on  three  orthogonal  planes  (LBP-TOP),  non-negativematrix factorization (NMF)and sparse learning for FER.但是，自2013年以来，诸如FER2013和Wild情感识别（EmotiW)，等情感识别竞赛从具有挑战性的现实场景中收集了相对足够的训练数据，这相当促进从ab-controlled to in-the-wild环境的FER。同时，由于芯片处理能力的显着提高和精心设计的网络架构，各个领域的研究已开始转移到深度学习方法，这些方法已经达到了最新的识别精度，并且大大超过了以前的结果。图1在算法和数据集方面说明了FER的这种演变。 ![](https://github.com/David-on-Code/Facial-expression-recognizition/blob/master/Datasets/Figure1.png)  
尽管深度学习有强大的特征学习能力，但是应用于FER时仍然存在问题。首先深度神经网络需要大量的训练数据来避免过拟合。然而，现存的表情数据库不足够用众所周知的深度架构的神经网络训练实现更好的结果在目标识别任务。此外主体间存在很大的差异由于不同的人的属性，如年龄、性别、种族背景和表达水平。除了目标个体差异，姿势、光照和遮挡也常见于不受约束的面部表情方案。这些因素与面部表情非线性耦合，因此加强了深度网络的需求，以应对较大的类内变化并学习有效的表情特定表示。

### 深度面部表情识别  
#### pre-processing  
##### face alignment  
背景、光照、头姿势。需要进行预处理以对齐和标准化人脸传达的视觉语义信息。  
尽管人脸检测是实现特征学习的唯一必不可少的过程，但使用局部界标的坐标进行进一步的人脸对齐可以显着提高FER性能。此步骤至关重要，因为它可以减少面部比例和面内旋转度的变化。表2研究了深度FER中广泛使用的面部界标检测算法，并从效率和性能方面对它们进行了比较。![表2](https://github.com/David-on-Code/Facial-expression-recognizition/blob/master/Datasets/Table2.jpg)    
总的来说，级联回归已经成为最流行的，最先进的人脸对齐方法，因为它具有很高的速度和准确性。  
与仅使用一个检测器进行面部对齐相比，一些方法提出了在挑战性不受约束的环境中处理面部时，将多个检测器组合在一起以进行更好的地标估计的方法。Yu等串联了三个不同的标注探测器，以相互补充。Kim等考虑了不同的输入（原始图像和直方图均衡化的图像）和不同的面部检测模型（V＆J和MoT），因此选择了由Intraface提供的具有最高置信度的标注集。  
##### Data augmentation  
深度神经网络需要足够的训练数据以确保可通用性达到给定的识别任务。然而，大多数FER数据集没有足够的数据集用来训练。因此，数据增强是深度FER的重要步骤。数据增强技术可以分为两类：动态数据增强和离线数据增强。  
通常，实时数据扩充被嵌入在深度学习工具包中，以减轻过度拟合的风险。在训练步骤中，从图像的四个角和中心随机裁剪输入样本，然后将其水平翻转，从而可以得到比原始训练数据大十倍的数据集。在测试过程中采用两种常见的预测模式：仅使用面部的中心补丁进行预测，或者预测值是所有十种crops的平均值。  
除了基本的实时数据扩充外，还设计了各种离线数据扩充操作，以进一步扩展数据的大小和分散性。最常用的操作包括随机扰动和变换，例如旋转，移位，偏斜，缩放，噪声，对比度和色彩抖动。例如，采用常见的噪声模型，盐和胡椒和斑点噪声和高斯噪声来扩大数据大小。为了进行对比度转换，每个像素的饱和度和值（HSV色彩空间的S和V分量）都会更改，以进行数据增强。多种操作的结合可以生成更多看不见的训练样本，并使网络对于偏斜和旋转的脸部更加健壮。此外，基于深度学习的技术可以应用于数据扩充。例如，具有3D卷积神经网络（CNN）的合成数据生成系统，以创建具有不同饱和度表情的人脸。生成对抗网络（GAN）也可以通过生成姿势和表情各异的appearance来应用于增强数据。
##### Face normalization  
光照和头部姿势的变化可能会导致图像发生较大变化，从而损害FER性能。因此，我们引入了两种典型的人脸归一化方法来改善这些变化：光照归一化和姿势归一化（正面化）  
Illumination  normalizatio：即使来自同一个人的相同表情，照明和对比度也会在不同的图像中变化，尤其是在不受约束的环境中，这会导致较大的类内差异。深度FER文献中的许多研究都使用直方图均衡化来提高图像的整体对比度，从而进行预处理。当背景和前景的亮度相似时，此方法有效。但是，直接应用直方图均衡化可能会过分强调局部对比。。为了解决这个问题，提出了一种加权求和方法，用于组合直方图均衡和线性映射。全局对比度归一化（GCN），局部归一化和直方图均衡化。报告显示，GCN和直方图均衡化分别在训练和测试步骤中获得了最佳的准确性。  
##### Pose normalization  
在不受约束的环境中，相当大的姿势变化是另一个常见且棘手的问题。最近，提出了一系列基于GAN的深度模型用于前视图合成（例如FF-GAN，TP-GAN和DR-GAN），并报告了令人鼓舞的性能。
#### deep feature learing
深度学习近来已成为研究的热点，并已针对各种应用实现了最先进的性能。深度学习尝试通过多个非线性变换和表示的层次结构来捕获高级抽象。在本节中，我们简要介绍了已应用于FER的一些深度学习技术。这些深度神经网络的传统架构如图2所示。  
![图2](https://github.com/David-on-Code/Facial-expression-recognizition/blob/master/Datasets/Fig2.png)  
##### Convolutional neural network (CNN)  
CNN已广泛用于包括FER在内的各种计算机视觉应用中。在21世纪初，FER文献的多项研究发现，CNN能够有效应对面部位置变化和尺度变化，并且在以前看不见的面部姿势变化情况下，其表现优于多层感知器（MLP）。使用CNN解决了面部表情识别中的主体独立性以及平移，旋转和尺度不变性的问题。  
CNN由三种类型的异构层组成：卷积层，池化层和全连接层。卷积层具有一组可学习的过滤器，用于对整个输入图像进行卷积并生成各种特定类型的激活特征图。卷积操作具有三个主要优点：局部连接，它学习相邻像素之间的相关性；同一特征图中的权重共享，大大减少了要学习的参数数量；池化层跟随卷积层，用于减少特征图的空间大小和网络的计算成本。平均池化和最大池化是平移不变性最常用的两种非线性下采样策略。全连接层通常包含在网络末端，以确保该层中的所有神经元都完全连接到上一层中的激活使2D特征图将转换为一维特征图以进行进一步的特征表示和分类。  
我们在表3中列出了一些已应用于FER的著名CNN模型的配置和特征。![图3](https://github.com/David-on-Code/Facial-expression-recognizition/blob/master/Datasets/Table3.png)  
除了这些网络之外，还存在一些众所周知的派生框架。基于区域的region-based CNN（R-CNN）用于学习FER的功能。Faster R-CNN通过生成高质量的区域提议来识别面部表情。此外，3D CNN捕获通过多个相邻帧编码的运动信息，以通过3D卷积进行动作识别。提出了精心设计的C3D，该C3D利用3D卷积在大型监督训练数据集中学习时空特征。许多相关研究已将此网络用于涉及图像序列的FER。  
#####  Deep belief network (DBN)  
由Hinton等人提出的DBN是学习提取训练数据的深层次表示的图形模型。传统的DBN是由一堆受限的Boltzmann机器（RBM）构建的，这是由可见单元层和隐藏单元层组成的两层生成随机模型。RBM中的这两层必须形成没有横向连接的二部图。在DBN中，训练高层中的单元以学习相邻较低层中的单元之间的条件依赖性，除了顶层两层具有无定向的连接之外。DBN的训练包含两个阶段：预训练和微调。首先，一种有效的逐层贪婪学习策略用于以无人监督的方式初始化深层网络，这可以在一定程度上防止局部最优结果的劣势，而无需大量标记数据。在此过程中，使用对比散度训练DBN中的RBM，以估计对数似然的近似梯度。然后，在监督下以简单的梯度下降对网络的参数和所需的输出进行微调。  
#####  Deep autoencoder (DAE)  
DAE学习有效的降维编码。与之前提到的训练目标值的网络相反，DAE通过最小化重构误差来优化其输入。 存在DAE的变体，例如降噪自动编码器，它可以从部分损坏的数据中恢复原始的未失真输入。 稀疏自动编码器网络（DSAE），在学习的特征表示上实施稀疏性； 压缩自编码器（CAE1），它添加了一个依赖于活动的正则化来诱导局部不变的特征； 卷积自动编码器（CAE2），它对网络中的隐藏层使用卷积（和可选的池化）层； 以及变分自动编码器（VAE），它是一种定向图形模型，具有某些类型的潜在变量，可以设计复杂的数据生成模型。  
#####  Recurrent neural network (RNN)  
RNN是一种捕获时间信息的连接主义模型，更适合于具有任意长度的顺序数据预测。除了以单一前馈方式训练深度神经网络外，RNN还包括跨越相邻时间步长并在所有步长上共享相同参数的递归边。经典的时间反向传播（BPTT）用于训练RNN。 Hochreiter＆Schmidhuber引入的长期短期记忆（LSTM）是传统RNN的一种特殊形式，用于解决训练RNN时常见的梯度消失和爆炸问题。 LSTM中的细胞状态由三个门控制和控制：一个输入门允许或阻止输入信号改变细胞状态，一个输出门使细胞状态能够影响其他神经元，或阻止该状态影响其他神经元。调制单元的自循环连接以累积或忘记其先前状态。通过结合这三个门，LSTM可以按顺序对长期依赖性进行建模，并已广泛用于基于视频的表情识别任务。  
#####  Generative Adversarial Network (GAN）  
![](https://github.com/David-on-Code/Facial-expression-recognizition/blob/master/Datasets/GAN.png)  
#### deep feature classification  
在学习了深度的特征之后，FER的最后一步是将给定的面孔分类为基本的情感类别之一。  
与传统方法不同，传统方法的特征提取步骤和特征分类步骤是独立的，而深度网络可以以端到端的方式执行FER。具体来说，在网络末端增加一个损失层，以调节反向传播误差。然后，每个样本的预测概率可以由网络直接输出。在CNN中，softmaxloss是最常用的函数，它可以最大程度地减少估计的类概率和地面真相分布之间的交叉熵。或者，证明了使用线性支持向量机（SVM）进行端到端训练的好处，该方法可最大程度地减少基于边际的损失而不是交叉熵。同样，研究了深度神经森林（NFs）的适应性，该神经森林用NFs取代了softmax损失层，并获得了竞争性结果。  
除了端到端的学习方式外，另一种选择是采用深度神经网络（尤其是CNN）作为特征提取工具，然后应用其他独立的分类。例如支持向量机或随机森林，以提取出的表示。此外，在DCNN特征上计算的协方差描述符和在对称正定（SPD）流形上用高斯核进行分类比使用softmax层进行标准分类更有效。  
#### THE STATE OF THE ART  
对于每个最频繁评估的数据集，表4显示了该领域中最新的方法，这些方法是在独立于人员的协议中明确进行的（训练和测试集中的主题已分离）  
![表4](https://github.com/David-on-Code/Facial-expression-recognizition/blob/master/Datasets/Table4.png)
####  Diverse network input  
传统做法通常使用RGB图像的整个对齐面作为网络的输入，以学习特征。然而，这些原始数据缺乏重要的信息，例如均匀或规则的纹理以及在图像缩放，旋转，遮挡和照明方面的不变性，这可能代表FER的混杂因素。一些方法采用了各种手工制作的功能及其扩展作为网络输入，以缓解此问题。  
低级表示对给定RGB图像中的小区域的特征进行编码，然后使用局部直方图对这些特征进行聚类和合并，这对于照明变化和较小的配准误差均很鲁棒。提出了一种新颖的映射LBP特征（见图5）用于照度不变的FER。针对多视图FER任务采用了针对图像缩放和旋转具有鲁棒性的缩放不变特征变换（SIFT）特性。将轮廓，纹理，角度和颜色的不同描述符组合为输入数据也可以帮助增强深层网络的性能。  
基于零件的表示会根据目标任务提取特征，这些特征会从整个图像中删除非关键部分，并利用对任务敏感的关键部分。三个感兴趣的区域（眉毛，眼睛和嘴巴）与面部表情变化密切相关，并裁剪了这些区域作为DSAE的输入。其他研究建议自动学习面部表情的关键部分。例如，使用深层多层网络来检测显着性图，该显着性图将强度加在需要视觉注意的零件上。并且应用了邻居中心差向量（NCDV）来获得具有更多固有信息的特征。  
#### Auxiliary blocks & layers  
基于CNN的基础架构，一些研究提出了添加精心设计的辅助块或层以增强与表情相关的特征学习能力。  
一种新颖的CNN架构,HoloNet是为FER设计的，其中CReLU与强大的残差结构结合使用可在不降低效率的情况下增加网络深度，并且具有初始残差块，是专为FER设计的，用于学习多尺度特征以捕获表达式中的变化。引入了另一种CNN模型，即监督评分系统（SSE），以提高FER的监督程度，在主流CNN的早期隐藏层中嵌入了三种类型的监督块，分别用于浅，中和深层监督。 见图6（a））  
CNN中的传统softmax损失层只是迫使不同类别的特征保持分离，但是现实世界中的FER不仅遭受类别间相似性高而且类别内部变异高的困扰。因此，一些工作提出了用于FER的新颖的损耗层。受中心损失的启发，该损失惩罚了深部特征与其对应的类中心之间的距离，提出了两种变体来协助对softmax损失进行监督，以对FER进行更多区分：（1）正式确定了岛损失以进一步增加成对将不同类别中心之间的距离（见图6（b））和（2）保留位置的损失（LP损失）形式化，以将同一类别的局部邻近特征拉在一起，以便每个类别的内部类集群类很紧凑。此外，基于三重态损失[169]，这需要一个正例比一个固定间隔的负例更接近锚点，提出了两种变体来替代或帮助监督softmax损失：（1）指数基于三重态的损失[145]被正式确定为在更新网络时给困难的样本更大的权重，（2）（N + M）-元组聚类损失[77]被正式确定为减轻锚点选择和阈值验证的难度身份不变FER的三元组丢失（有关详细信息，请参见图6（c））。此外，提出了特征损失，以在早期训练阶段为深度特征提供补充信息。  
![图6](https://github.com/David-on-Code/Facial-expression-recognizition/blob/master/Datasets/Fig6.png)  

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
