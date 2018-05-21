# Behavioural Cloning for Self Driving Cars
Some work on the [comma.ai](https://comma.ai) self driving cars dataset. 


![Alt Text](https://github.com/nfsrules/SelfDriving/blob/master/images/sample_output.gif)



## Autopilot Architecture
The proposed architecture is a slightly modified version of the PilotNet published by [Nvidia](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). The following improvements are proposed to ameliorate training and generalization capabilities.

1. <strong>Batch normalization after each convolutive layer.</strong>
2. <strong>Activation function type 'elu'. For more information read this [article.](http://image-net.org/challenges/posters/JKU_EN_RGB_Schwarz_poster.pdf) </strong>
3. <strong>Dropout before the output layer.</strong>

![alt text](https://github.com/nfsrules/SelfDriving/blob/master/images/model_inet_2.png)

## Visualization
Some nice functions to annotate input images and render .AVI videos with the autopilot predictions are provided. These functions were used to generate the gif showed above.

1. <strong>Annotate input images with vehicle's CAN readings.</strong>
2. <strong>Generate video with the autopilot predictions and target angles.</strong>


## Licence
Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License
