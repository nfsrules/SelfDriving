# SelfDriving
Some work on comma.ai self driving cars dataset.


<p align="center">![Alt Text](https://github.com/nfsrules/SelfDriving/blob/master/images/sample_output.gif)</p>



## Model Architecture
The proposed driving architecture is a slightly modified version of the PilotNet published by [Nvidia](https://www.youtube.com/watch?v=ccShIHBCx4g).
Batch normalization after each convolution layer and Dropout after the first fully connected layer allows to improve learning and generalization.

![alt text](https://github.com/nfsrules/SelfDriving/blob/master/images/model_inet_2.png)




