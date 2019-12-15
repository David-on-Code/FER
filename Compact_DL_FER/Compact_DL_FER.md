### A Compact Deep Learning Model for Robust Facial Expression Recognition
2018 CVPR  
### 紧凑的深度学习模型，用于鲁棒的面部表情识别   
#### 摘要：
我们提出了一种基于框架的紧凑型面部表情识别框架，用于面部表情识别，相比于最新方法具有非常有竞争力的性能，同时使用的参数更少。提出的框架扩展到frame-to-sequence方法,通过利用有gated recurrent units（门控循环单元）的时间信息。此外，我们开发了一种光照增强方案，以缓解使用混合数据源训练深度网络时的过拟合问题。最后，我们使用提出的技术在一些公共数据集上证明性能的改进。  
#### 1.Introduction  
随着深度学习和人机交互的进步，从图像中了解人的情感变得越来越重要。人的情感表达方式有多种。研究表明，非姿势表达的分析必须依靠其他生理信号，例如温度变化和心率。遗憾的是，这些生理方法在实际中通常不可用或不可行，这使得研究结果仅限于实验室环境中。  
由于易于获取数据，基于视频的方法最常用于表情识别。具有严格限制的数据库通常用于面部表情识别的性能基准测试。传统的基于图像的面部表情识别方法采用了手工特征，比如LBP，BoW,HoG,SIFT,并且它们在几个数据库上都显示了相当不错的结果。此外，基于序列的方法还对时态情绪变化建模，从视频中提取出时间时间手工特征。   
最近，自然环境下的表情识别引起了广泛的关注。这种问题很具有挑战性，因为收集的面部图像通常是从互联网获取的在不同的照明条件和头部姿势下。诸如EmotioNet之类的研究还表明，将下载的图像用于训练集对于提高模型训练的通用性非常有用。这激励我们进一步研究表情识别任务如何从不受约束的环境中获取的面部图像数据集的模型训练中受益。  
在本文中，我们介绍了一种新的卷积神经网络（CNN）架构，以通过适当的深层设计来提高性能和泛化能力网络。标准数据库上的实验还显示所提出的CNN模型适用于面部具有紧凑网络参数的表情识别与相关的基于深度学习的模型相比。 此外，我们将几个不同类型的数据集包含到训练数据集以提高学习到的CNN模型的泛化能力。除此之外，我们研究了一种光照增强方案，以提高训练的CNN模型的鲁棒性。本文的主要贡献可以总结如下：  
•提出了一种用于面部表情识别的紧凑型CNN模型，以在识别精度和模型大小之间折衷。  
•在两个标准数据库上评估了我们的网络模型，并证明了所提出的方法优于最新方法。  
•收集了三个不同场景的数据集可用于评估跨域性能。  
•我们提出的“一站式（leave-one-set-out）”实验表明提议的光照增强策略减轻了模型训练中不同来源的图像的过度拟合问题。  
#### 2. Related Work  
BDBN表明，特征提取和选择与统一的增强型深度置信网络相结合可获得更好的性能。 STM-ExpLet 使用expressionlet-based 时空流形为表情视频剪辑建模。Exemplar-HMMs结合HMMs和SVMs在基于模型的相似性框架中。LOMo结合不同类型的complimentary特征，比如面部标志，LBP，SIFT，和几何特征，FER。  
在最近几年，自从CNN在许多计算机视觉任务中展现出其空前的能力以来，深度学习就变得非常流行。已经针对不同的图像分类任务提出了各种CNN模型。 但是，这些深度网络不适用于小型表情识别数据库。  
联合微调方法采用具有7个不同旋转角度的数据扩充策略，以获得14倍的数据。根据外观和几何特征训练两个不同的网络，并通过联合微调来组合预训练的网络。研究人员还表明，将CNN与递归神经网络（RNN）结合使用对于从视频中进行表情识别非常有效。  
最近，通过将从大规模人脸识别数据库中学习到的知识进行转移，峰值引导方法成功地将GoogLeNet应用于表情识别。 他们的结果还表明，基于图像的方法的准确性与基于序列的方法的准确性相当。  
#### 3. Proposed Framework   
所提出的深度学习方法的总体流程如图1所示。我们的框架由两个模块组成：人脸预处理和CNN分类。为了确保我们的框架可以扩展到不同的场景，我们不采用任何时间标准化
方法。  ![Fig1](https://github.com/David-on-Code/Facial-expression-recognizition/blob/master/Compact_DL_FER/Fig1.png)  
##### 3.1. Preprocessing   
我们首先根据IntraFace检测到的landmarks裁剪面部区域。这些landmarks可用于提取眉毛，眼睛，鼻子和嘴巴的轮廓。大crop可以保留更多信息，而小crop可以减少背景噪声或头部轮廓。裁剪后的图像尺寸$L=\alpha \times \max \left(d_{v}, d_{h}\right)$其中$d_{v}$是the uppermost landmark point和the lowermost landmark point之间的距离, $d_{h}$和the the leftmost landmark point和the rightmost landmark point之间的水平距离, α是用于控制面部区域大小的常量。对于所有实验，我们将α设置为1.05。  
一旦确定了裁剪的大小L，我们在鼻子的landmark上裁剪脸部区域中心，并获得适度的脸部图像以进行模型训练。将裁剪后的图像调整为固定尺寸120×120，然后将其发送到CNN分类器进行表情识别。  
##### 3.2. The CNN Model   
我们的CNN模型的架构如图2所示。 ![图二](https://github.com/David-on-Code/Facial-expression-recognizition/blob/master/Compact_DL_FER/Fig2.png) 模型由两个卷积和池化块，然后是两个全连接层，使用ReLU作为每个卷积层的激活函数。在全连接层后应用下采样以防止过拟合。注意，该模型仅将调整后的面部图像的中心96×96部分用作输入。有关模型训练的详细信息将在下一部分中描述。  
提议的CNN结构可以视为DTAN的改进版本。他们的实验已经表明，该简单模型可以在表情识别任务中取得良好的效果。为了进一步提高模型的识别能力，我们在最大池化之前堆叠了两个连续的卷积层。我们还使用了较大的卷积滤波器，从而使模型中的神经元具有更大的感受野。经过这种修改后，第一个完全连接层中每个神经元的感受野将变为36×36，约为输入96×96图像的14％，而原来的DTGAN为16×16，仅占输入大小为64×64的6%。  
另一个重要的修改是减少完全连接的神经元的数量。我们相信，只要我们对感受野进行适当的设计，就可以通过适度的模型大小来学习人脸的表情。本文后面的实验表明，合适的轻量级全连接网络不仅模型参数紧凑，而且对于面部表情识别也很准确。  
#### 3.3. The Frame-to-Sequence Model  
标准面部表情数据库中的图像序列通常以中性表情开始，然后逐渐发展为峰值表情。我们可以用一个模型$S(x)$来近似这个转换过程，使用一系列图像$x_{i}^{t}, t=1, \ldots, T$作为输入，并且将每个图像序列映射到其ground truth$y_{i}$尽可能近：  
$
y_{i} \cong \widetilde{Y}_ {i}=S\left(x_{i}^{1}, \ldots, x_{i}^{T} ; \theta\right), --(1)
$   
T是图像序列的长度，$\theta$是一组模型参数。$p$表示由序列模型产生的每个表情的概率，序列建模问题可以表述为在给定训练序列的情况下最大化模型的对数似然性，即，  
$$
\hat{\theta}=\underset{\theta}{\arg \max } \frac{1}{N} \sum_{i=1}^{N} \log p\left(\widetilde{Y}_ {i} | x_{i}^{1}, \ldots, x_ {i}^{T} ; \theta\right), --(2)
$$  
这个问题很难直接解决，因此我们使用预先训练的CNN作为特征提取器。先前的基于帧的方法可以视为映射函数$F(x)$，映射每个$x_{i}^{t}$到概率分布 
$\{p_{1}^{t}(j), j=1, \dots, m\} $所以最大概率索引$p_{i}^{t}(j), \widetilde{y}_ {i}$像正确类别$y_{i}$一样,如  
$$y_{i} \cong \widetilde{y}_ {i}=\underset{j}{\arg \max } p_{i}^{t}(j)=F\left(x_{i}\right), --(3)$$  
$$p_{i}^{t}=F\left(x_{i}\right)=\left[p_{i}^{t}(1), p_{i}^{t}(2), \ldots, p_{i}^{t}(m)\right]$$  
这里，我们使用从CNNs计算出的概率分布序列进行基于帧的表情识别，而不是使用图像作为表情识别的输入,这意味着  
$y_{i} \cong \widetilde{Y}_ {i}=S\left(F\left(x_{i}^{1}\right), \ldots, F\left(x_{i}^{T}\right) ; \theta\right)$ --(4)  
用Gated Recurrent Neural Network建模S(x)。由于我们将概率分布用于基于帧的分类作为特征表示，我们期望S(x)可以由浅层结构很好地建模。我们的帧到序列模型的架构由具有128个隐藏层的单个门控循环单元（GRU）层和一个softmax层组成。 总体框架如图3所示。
![图3](https://github.com/David-on-Code/Facial-expression-recognizition/blob/master/Compact_DL_FER/Fig3.png)  
##### 3.4. Model Training  
小的面部表情识别数据集通常有几百图像序列，在模型训练时很容易导致过拟合问题。对于模型训练，我们使用水平反转和随机移动(random shifting)的在线数据增强。在文章中，设置最大迭代次数2000epoches并报告用于训练的CNN模型的最佳验证准确性。对于帧到序列模型，使用ADAM优化器作为模型训练并使用48batch大小和固定学习率0.01迭代10000次。  
#### 4. Experiments on Standard Databases  
标准面部表情识别数据库一般包含从中性表情到峰值表情的视频序列。对于基于帧的方法，我们只使用峰值图像作为训练和验证。我们首先在两个知名标准数据库上评估提出的架构：the Extended Cohn-Kanade (CK+) database and the Oulu-CASIA database。CK+数据库由327个标记有7种情感的图像序列组成：anger,contempt, disgust, fear, happiness, sadness, and surprise。Oulu-CASIA数据库包含480个有六种情感标签的图像序列组成：anger,disgust, fear, happiness,sadness, surprise。 CK+ and Oulu-CASIA数据的大小分别是640x490和320x320.这些数据库的详细信息在表4.对于CNN模型的训练，所有加权层通过xavier初始化，学习率固定为0.001，动量设为0.9 。权重衰减方法也用于正则化，系数为0.001 。  
##### 4.1. Frame-based Approach  
为了避免受试者同时出现在训练和测试集中，我们按照ID升序将其分为10个子集，这与10折交叉验证方案相同。该实验方案用于本文实验比较中包含的所有方法。表1显示了CK +数据库上10折交叉验证的总体准确性。我们基于框架的方法的准确性优于大多数基于序列的方法，仅次于峰引导方法。然而，在[47]中，它们使用CK+ 数据库一千倍大小的500K张图像预训练CNN模型。结果表明提出的CNN模型非常适合在小的数据库上学习面部表情。  








[47]X. Zhao, X. Liang, L. Liu, T. Li, Y. Han, N. Vasconcelos,and S. Yan. Peak-piloted deep network for facial expression recognition. In European Conference on Computer Vision,pages 425–442. Springer, 2016.
