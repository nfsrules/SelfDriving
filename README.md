# SelfDriving
Some work on comma.ai self driving cars dataset.


<p align="center">![Alt Text](https://github.com/nfsrules/SelfDriving/blob/master/images/sample_output.gif) </p>



## Autopilot Architecture
The proposed architecture is a slightly modified version of the PilotNet published by [Nvidia](https://www.youtube.com/watch?v=ccShIHBCx4g). The following improvements are proposed to ameliorate training and generalization.

1. <strong>Batch normalization after each convolutive layer.</strong>
2. <strong>Activation function type 'elu'. For more information read this [article](http://image-net.org/challenges/posters/JKU_EN_RGB_Schwarz_poster.pdf) </strong>
3. <strong>Dropout before output layer.</strong>

![alt text](https://github.com/nfsrules/SelfDriving/blob/master/images/model_inet_2.png)




