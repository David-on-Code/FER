## Spontaneous Facial Micro-Expression Recognition using 3D Spatiotemporal Convolutional Neural Networks  
### Abstract  
Fake facial ex- pressions are difficult to be recognized even by humans.  
Facial micro-expressions generally represent the actual emotion of a person, as it is a spontaneous reaction expressed through human face.  
Despite of a few attempts made for recognizing micro-expressions, still the problem is far from being a solved problem, which is depicted by the poor rate of ac- curacy shown by the state-of-the-art methods.  
A few CNN based approaches are found in the literature to recognize micro-facial expressions from still images.   
This paper proposes two 3D-CNN methods: MicroExpSTCNN and MicroExpFuseNet, for spontaneous facial micro-expression recognition by exploiting the spatiotemporal information in CNN framework. The MicroExpSTCNN considers the full spatial information, whereas the MicroExpFuseNet is based on the 3D-CNN feature fusion of the eyes and mouth regions.   
The experiments are performed over CAS(ME)^2 and SMIC micro- expression databases.   
### Introduction  
LBP, LBP-TOP,directional mean optical flow feature
CNN, RNN, combinations of CNN and RNN.
### LITERATURE REVIEW  
### MicroExpSTCNN considers full face as the input  
![F1_MicroExpSTCNN](https://github.com/David-on-Code/Facial-expression-recognizition/blob/master/3D-CNN/F1_MicroExpSTCNN.png)
![Table1](https://github.com/David-on-Code/Facial-expression-recognizition/blob/master/3D-CNN/Table_MicroExpSTCNN.jpeg)
![code](https://github.com/David-on-Code/Facial-expression-recognizition/blob/master/3D-CNN/Code_MicroExpSTCNN.png)
### MicroExpFuseNet considers the eyes and mouth regions as the inputs  
In MicroExpFuseNet model, only eyes and mouth regions of the face are used as input to two separate 3D spatio-temporal CNNs.  Based on the different fusion strategies (i.e., at different stages), we propose two versions of MicroExpFuseNet models: Intermediate MicroExpFuseNet and Late MicroExpFuseNet.
#### 1) Intermediate MicroExpFuseNet Model  
![F2](https://github.com/David-on-Code/Facial-expression-recognizition/blob/master/3D-CNN/F2_Immediate_MicroExpFuseNet.png)
Table II reports the network architecture of the proposed Intermediate MicroExp- FuseNet model in terms of the filter dimension and the output dimension of different layers. In Table II, the input dimension is considered for the CAS(ME)^2 dataset.  
![Tabel2](https://github.com/David-on-Code/Facial-expression-recognizition/blob/master/3D-CNN/Table2.png)
#### 2) Late MicroExpFuseNet Model
![F3](https://github.com/David-on-Code/Facial-expression-recognizition/blob/master/3D-CNN/F3_Immediate_MicroExpFuseNet.png)
Table III presents the network architecture of the proposed Late MicroExpFuseNet model in terms of the filter dimension and the output dimension of different layers. In Table III, the input dimension is considered for the CAS(ME)?2 dataset.  
![Table3](https://github.com/David-on-Code/Facial-expression-recognizition/blob/master/3D-CNN/Table3.png)
### Samples
Table IV shows the expression levels and the number of video samples present in the respective datasets.  
![Table4](https://github.com/David-on-Code/Facial-expression-recognizition/blob/master/3D-CNN/Table4.png)
### COMPARISON
![Table5](https://github.com/David-on-Code/Facial-expression-recognizition/blob/master/3D-CNN/Table5.png)
### Experimental Results
![Table6_7](https://github.com/David-on-Code/Facial-expression-recognizition/blob/master/3D-CNN/Table6_7.png)
### Accuracy Standard Deviation Analysis
Table VIII reports the mean and standard deviation in validation accuracies between epochs 91 to 100 for MicroExpSTCNN and MicroExpFuseNet models over CAS(ME)^2 and SMIC datasets.
![Table8](https://github.com/David-on-Code/Facial-expression-recognizition/blob/master/3D-CNN/Table8.png)
### Impact of 3D Kernel
![Table9](https://github.com/David-on-Code/Facial-expression-recognizition/blob/master/3D-CNN/Table9.png)
### CONCLUSION
Other than eyes and mouth regions, some other salient facial regions can contribute to micro-expression recognition. 
